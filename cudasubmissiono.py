import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_M = 32;
constexpr int WARP_N = 32;
constexpr int PAD_A = 8;
constexpr int PAD_B = 8;
constexpr int NUM_THREADS = 512;

__global__ void matmul_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M, const int N, const int K)
{
    __shared__ half As[BLOCK_M][BLOCK_K + PAD_A];
    __shared__ half Bs[BLOCK_K][BLOCK_N + PAD_B];

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int warp_row = warpId >> 2;
    const int warp_col = warpId & 3;

    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int a_load_row = tid >> 2;
    const int a_load_col = (tid & 3) << 3;
    const int b_load_row = tid >> 4;
    const int b_load_col = (tid & 15) << 3;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        {
            const half* a_src = A + (block_row + a_load_row) * K + k0 + a_load_col;
            *reinterpret_cast<float4*>(&As[a_load_row][a_load_col]) =
                *reinterpret_cast<const float4*>(a_src);
        }
        {
            const half* b_src = B + (k0 + b_load_row) * N + block_col + b_load_col;
            *reinterpret_cast<float4*>(&Bs[b_load_row][b_load_col]) =
                *reinterpret_cast<const float4*>(b_src);
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K / WMMA_K; kk++) {
            #pragma unroll
            for (int wi = 0; wi < 2; wi++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag,
                    &As[warp_row * WARP_M + wi * WMMA_M][kk * WMMA_K],
                    BLOCK_K + PAD_A);

                #pragma unroll
                for (int wj = 0; wj < 2; wj++) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag,
                        &Bs[kk * WMMA_K][warp_col * WARP_N + wj * WMMA_N],
                        BLOCK_N + PAD_B);

                    wmma::mma_sync(acc[wi][wj], a_frag, b_frag, acc[wi][wj]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int wi = 0; wi < 2; wi++) {
        #pragma unroll
        for (int wj = 0; wj < 2; wj++) {
            const int c_row = block_row + warp_row * WARP_M + wi * WMMA_M;
            const int c_col = block_col + warp_col * WARP_N + wj * WMMA_N;

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
            #pragma unroll
            for (int i = 0; i < acc[wi][wj].num_elements; i++)
                c_frag.x[i] = __float2half(acc[wi][wj].x[i]);

            wmma::store_matrix_sync(
                &C[c_row * N + c_col], c_frag, N, wmma::mem_row_major);
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    dim3 block(NUM_THREADS);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    matmul_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K
    );

    return C;
}
"""

cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);
"""

module = load_inline(
    name='matmul_wmma',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['matmul_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_80'],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    return module.matmul_cuda(a, b, c)
