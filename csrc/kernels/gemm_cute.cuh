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

    // Thread and block indices
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global memory tensors
    auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, 1));
    auto gB = make_tensor(make_gmem_ptr(B), make_shape(K, N), make_stride(N, 1));
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, 1));

    // Tile the global tensors
    auto tileA = local_tile(gA, make_shape(Int<kTileM>{}, Int<kTileK>{}), make_coord(by, _));
    auto tileB = local_tile(gB, make_shape(Int<kTileK>{}, Int<kTileN>{}), make_coord(_, bx));
    auto tileC = local_tile(gC, make_shape(Int<kTileM>{}, Int<kTileN>{}), make_coord(by, bx));

    // Shared memory for tiles
    __shared__ Element sA[kTileM * kTileK];
    __shared__ Element sB[kTileK * kTileN];

    auto smemA = make_tensor(make_smem_ptr(sA), make_shape(Int<kTileM>{}, Int<kTileK>{}));
    auto smemB = make_tensor(make_smem_ptr(sB), make_shape(Int<kTileK>{}, Int<kTileN>{}));

    // Accumulator
    Element acc = Element(0);

    // Number of K tiles
    int num_k_tiles = (K + kTileK - 1) / kTileK;

    // Loop over K dimension
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // Load A tile to shared memory
        auto gA_slice = tileA(_, k_tile);
        for (int i = tid; i < kTileM * kTileK; i += blockDim.x) {
            int m = i / kTileK;
            int k = i % kTileK;
            if (by * kTileM + m < M && k_tile * kTileK + k < K) {
                smemA(m, k) = gA_slice(m, k);
            } else {
                smemA(m, k) = Element(0);
            }
        }

        // Load B tile to shared memory
        auto gB_slice = tileB(k_tile, _);
        for (int i = tid; i < kTileK * kTileN; i += blockDim.x) {
            int k = i / kTileN;
            int n = i % kTileN;
            if (k_tile * kTileK + k < K && bx * kTileN + n < N) {
                smemB(k, n) = gB_slice(k, n);
            } else {
                smemB(k, n) = Element(0);
            }
        }

        __syncthreads();

        // Compute tile
        int m = tid / kTileN;
        int n = tid % kTileN;

        if (m < kTileM && n < kTileN) {
            for (int k = 0; k < kTileK; ++k) {
                acc += smemA(m, k) * smemB(k, n);
            }
        }

        __syncthreads();
    }

    // Write result
    int m = tid / kTileN;
    int n = tid % kTileN;
    int gm = by * kTileM + m;
    int gn = bx * kTileN + n;

    if (gm < M && gn < N && m < kTileM && n < kTileN) {
        tileC(m, n) = acc;
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
