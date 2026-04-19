import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// 64x64 Output Tile, 32 K-dimension Tile
const int BM = 64;
const int BN = 64;
const int BK = 32;

// Shared memory union to reuse the 16KB space for inputs and output casting
union SharedMem {
    struct {
        half As[BM * BK];
        half Bs[BK * BN];
    } inputs;
    float Cs[BM][BN];
};

__global__ void wmma_shared_matmul_kernel(
    const half* __restrict__ A, 
    const half* __restrict__ B, 
    half* __restrict__ C, 
    int M, int N, int K
) {
    // 4 warps = 128 threads
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    // Block positions
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ SharedMem smem;

    // Accumulators: each warp computes a 32x32 sub-tile (4 fragments of 16x16)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];

    // Warp offset for the C tile
    int warp_row = (warp_id / 2) * 32;
    int warp_col = (warp_id % 2) * 32;

    // Loop over the K dimension
    for (int k = 0; k < K; k += BK) {
        
        // Coalesced load 64x32 A tile (2048 elements)
        for (int i = 0; i < 2048; i += 128) {
            int idx = i + tid;
            int r = idx / BK;
            int c = idx % BK;
            int global_r = by * BM + r;
            int global_c = k + c;
            smem.inputs.As[r * BK + c] = (global_r < M && global_c < K) ? A[global_r * K + global_c] : __float2half(0.0f);
        }

        // Coalesced load 32x64 B tile (2048 elements)
        for (int i = 0; i < 2048; i += 128) {
            int idx = i + tid;
            int r = idx / BN;
            int c = idx % BN;
            int global_r = k + r;
            int global_c = bx * BN + c;
            smem.inputs.Bs[r * BN + c] = (global_r < K && global_c < N) ? B[global_r * N + global_c] : __float2half(0.0f);
        }

        // Wait for shared memory tiles to populate
        __syncthreads();

        // Compute using Tensor Cores
        for (int step = 0; step < BK / 16; ++step) {
            
            // Load 16x16 A fragments from shared memory
            wmma::load_matrix_sync(a_frag[0], &smem.inputs.As[(warp_row + 0) * BK + step * 16], BK);
            wmma::load_matrix_sync(a_frag[1], &smem.inputs.As[(warp_row + 16) * BK + step * 16], BK);

            // Load 16x16 B fragments from shared memory
            wmma::load_matrix_sync(b_frag[0], &smem.inputs.Bs[(step * 16) * BN + warp_col + 0], BN);
            wmma::load_matrix_sync(b_frag[1], &smem.inputs.Bs[(step * 16) * BN + warp_col + 16], BN);

            // Matrix Multiply Accumulate
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        
        // Wait for all warps to finish computing before overwriting shared memory in the next k-step
        __syncthreads();
    }

    // Write accumulators into the shared memory float array to safely cast
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(&smem.Cs[warp_row + i * 16][warp_col + j * 16], c_frag[i][j], BN, wmma::mem_row_major);
        }
    }

    // Synchronize to ensure all warps have written to smem.Cs
    __syncthreads();

    // Coalesced cast and write of the 64x64 float matrix (4096 elements) back to global memory C
    for (int i = 0; i < 4096; i += 128) {
        int idx = i + tid;
        int r = idx / BN;
        int c = idx % BN;
        int global_r = by * BM + r;
        int global_c = bx * BN + c;
        
        if (global_r < M && global_c < N) {
            C[global_r * N + global_c] = __float2half(smem.Cs[r][c]);
        }
    }
}

torch::Tensor custom_matmul_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    dim3 threads(128);
    
    // Safely round up to handle small inputs/warmups and prevent 'invalid configuration argument'
    dim3 blocks((N + 63) / 64, (M + 63) / 64);

    wmma_shared_matmul_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        M, N, K
    );

    return c;
}
"""

cpp_source = """
torch::Tensor custom_matmul_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c);
"""

# Compile the extension, explicitly versioned to avoid PyTorch loading previous cached binaries
matmul_extension = load_inline(
    name='custom_matmul_ext_v4',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['custom_matmul_cuda'],
    with_cuda=True,
    extra_cuda_cflags=['-O3', '-arch=sm_80']
)

def custom_kernel(data):
    a, b, c = data
    return matmul_extension.custom_matmul_cuda(a, b, c)

if __name__ == "__main__":
    from task import input_t, output_t
    from utils import make_match_reference, DeterministicContext
    
    check_implementation = make_match_reference(custom_kernel)