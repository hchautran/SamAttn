#pragma once

#include <cuda_runtime.h>
#include <cute/tensor.hpp>

namespace samattn {

// Simple GEMM kernel using CUTLASS CUTE
// C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
template <typename Element, int kTileM, int kTileN, int kTileK>
__global__ void gemm_cute_kernel(
    const Element* A,
    const Element* B,
    Element* C,
    int M, int N, int K
) {
    using namespace cute;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;

    // Compute block offsets
    int block_m = by * kTileM;
    int block_n = bx * kTileN;

    // Shared memory for tiles
    __shared__ Element sA[kTileM * kTileK];
    __shared__ Element sB[kTileK * kTileN];

    // Register accumulator
    Element acc = Element(0);

    // Compute thread's output position within tile
    int thread_m = tid / kTileN;
    int thread_n = tid % kTileN;

    // Number of K tiles
    int num_k_tiles = (K + kTileK - 1) / kTileK;

    // Loop over K dimension
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int block_k = k_tile * kTileK;

        // Cooperatively load A tile [kTileM, kTileK] into shared memory
        for (int i = tid; i < kTileM * kTileK; i += blockDim.x) {
            int m = i / kTileK;
            int k = i % kTileK;
            int gm = block_m + m;
            int gk = block_k + k;

            if (gm < M && gk < K) {
                sA[m * kTileK + k] = A[gm * K + gk];
            } else {
                sA[m * kTileK + k] = Element(0);
            }
        }

        // Cooperatively load B tile [kTileK, kTileN] into shared memory
        for (int i = tid; i < kTileK * kTileN; i += blockDim.x) {
            int k = i / kTileN;
            int n = i % kTileN;
            int gk = block_k + k;
            int gn = block_n + n;

            if (gk < K && gn < N) {
                sB[k * kTileN + n] = B[gk * N + gn];
            } else {
                sB[k * kTileN + n] = Element(0);
            }
        }

        __syncthreads();

        // Compute partial dot product for this thread
        if (thread_m < kTileM && thread_n < kTileN) {
            for (int k = 0; k < kTileK; ++k) {
                acc += sA[thread_m * kTileK + k] * sB[k * kTileN + thread_n];
            }
        }

        __syncthreads();
    }

    // Write result to global memory
    if (thread_m < kTileM && thread_n < kTileN) {
        int gm = block_m + thread_m;
        int gn = block_n + thread_n;

        if (gm < M && gn < N) {
            C[gm * N + gn] = acc;
        }
    }
}

// Host-side launcher
template <typename Element>
void gemm_cute_launch(
    const Element* A,
    const Element* B,
    Element* C,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    constexpr int kTileM = 64;
    constexpr int kTileN = 64;
    constexpr int kTileK = 16;

    dim3 grid((N + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM);
    dim3 block(256);

    gemm_cute_kernel<Element, kTileM, kTileN, kTileK><<<grid, block, 0, stream>>>(
        A, B, C, M, N, K
    );
}

// Explicit instantiations
extern template void gemm_cute_launch<float>(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream
);

extern template void gemm_cute_launch<__half>(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K, cudaStream_t stream
);

} // namespace samattn
