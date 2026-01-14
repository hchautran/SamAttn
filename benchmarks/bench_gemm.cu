#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../csrc/kernels/gemm_cute.cuh"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(status) << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

template <typename T>
void benchmark_gemm(int M, int N, int K, int warmup_iters, int bench_iters) {
    // Allocate memory
    T *d_A, *d_B, *d_C_cute, *d_C_cublas;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_C_cute, M * N * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_C_cublas, M * N * sizeof(T)));

    // Initialize with random data
    // (In practice, you'd want to copy actual data from host)

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        samattn::gemm_cute_launch<T>(d_A, d_B, d_C_cute, M, N, K, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark CUTLASS CUTE
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; ++i) {
        samattn::gemm_cute_launch<T>(d_A, d_B, d_C_cute, M, N, K, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float cute_time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&cute_time_ms, start, stop));
    cute_time_ms /= bench_iters;

    // Benchmark cuBLAS
    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha,
                                     d_B, N, d_A, K, &beta, d_C_cublas, N));
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha,
                                     d_B, N, d_A, K, &beta, d_C_cublas, N));
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float cublas_time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&cublas_time_ms, start, stop));
    cublas_time_ms /= bench_iters;

    // Calculate TFLOPS
    double flops = 2.0 * M * N * K;
    double cute_tflops = flops / (cute_time_ms * 1e-3) / 1e12;
    double cublas_tflops = flops / (cublas_time_ms * 1e-3) / 1e12;

    std::cout << "=== GEMM Benchmark [" << M << " x " << N << " x " << K << "] ===" << std::endl;
    std::cout << "CUTLASS CUTE: " << cute_time_ms << " ms, " << cute_tflops << " TFLOPS" << std::endl;
    std::cout << "cuBLAS:       " << cublas_time_ms << " ms, " << cublas_tflops << " TFLOPS" << std::endl;
    std::cout << "Speedup:      " << cublas_time_ms / cute_time_ms << "x" << std::endl;
    std::cout << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C_cute));
    CHECK_CUDA(cudaFree(d_C_cublas));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    std::cout << "CUTLASS CUTE GEMM Benchmark" << std::endl;
    std::cout << "============================" << std::endl << std::endl;

    int warmup_iters = 5;
    int bench_iters = 100;

    // Square matrices
    benchmark_gemm<float>(512, 512, 512, warmup_iters, bench_iters);
    benchmark_gemm<float>(1024, 1024, 1024, warmup_iters, bench_iters);
    benchmark_gemm<float>(2048, 2048, 2048, warmup_iters, bench_iters);
    benchmark_gemm<float>(4096, 4096, 4096, warmup_iters, bench_iters);

    // Rectangular matrices
    benchmark_gemm<float>(1024, 512, 2048, warmup_iters, bench_iters);
    benchmark_gemm<float>(2048, 1024, 4096, warmup_iters, bench_iters);

    return 0;
}
