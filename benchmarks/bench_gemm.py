#!/usr/bin/env python3
"""
Benchmark CUTLASS CUTE GEMM against PyTorch native GEMM
"""

import torch
import time
from typing import Tuple

try:
    import samattn
    HAS_SAMATTN = True
except ImportError:
    HAS_SAMATTN = False
    print("Warning: samattn module not found. Build the package first.")


def benchmark_torch_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    warmup_iters: int = 10,
    bench_iters: int = 100
) -> float:
    """Benchmark PyTorch native GEMM"""
    # Warmup
    for _ in range(warmup_iters):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / bench_iters * 1000  # Convert to ms


def benchmark_cute_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    warmup_iters: int = 10,
    bench_iters: int = 100
) -> float:
    """Benchmark CUTLASS CUTE GEMM"""
    if not HAS_SAMATTN:
        return float('inf')

    # Warmup
    for _ in range(warmup_iters):
        _ = samattn.gemm(A, B)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        _ = samattn.gemm(A, B)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / bench_iters * 1000  # Convert to ms


def verify_correctness(M: int, N: int, K: int, dtype: torch.dtype) -> bool:
    """Verify that CUTE GEMM produces correct results"""
    if not HAS_SAMATTN:
        return False

    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')

    torch_result = torch.matmul(A, B)
    cute_result = samattn.gemm(A, B)

    # Check relative error
    rel_error = torch.abs(torch_result - cute_result) / (torch.abs(torch_result) + 1e-5)
    max_error = rel_error.max().item()

    passed = max_error < 1e-2 if dtype == torch.float32 else max_error < 1e-1

    print(f"Correctness test [{M}x{N}x{K}]: {'PASSED' if passed else 'FAILED'}")
    print(f"  Max relative error: {max_error:.6f}")

    return passed


def run_benchmark(
    M: int, N: int, K: int,
    dtype: torch.dtype = torch.float32,
    warmup_iters: int = 10,
    bench_iters: int = 100
):
    """Run benchmark for a given shape"""
    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')

    torch_time = benchmark_torch_gemm(A, B, warmup_iters, bench_iters)

    if HAS_SAMATTN:
        cute_time = benchmark_cute_gemm(A, B, warmup_iters, bench_iters)

        # Calculate TFLOPS
        flops = 2 * M * N * K
        torch_tflops = flops / (torch_time * 1e-3) / 1e12
        cute_tflops = flops / (cute_time * 1e-3) / 1e12
        speedup = torch_time / cute_time

        print(f"\n=== GEMM Benchmark [{M} x {N} x {K}] ({dtype}) ===")
        print(f"PyTorch:      {torch_time:.3f} ms, {torch_tflops:.2f} TFLOPS")
        print(f"CUTLASS CUTE: {cute_time:.3f} ms, {cute_tflops:.2f} TFLOPS")
        print(f"Speedup:      {speedup:.2f}x")
    else:
        flops = 2 * M * N * K
        torch_tflops = flops / (torch_time * 1e-3) / 1e12
        print(f"\n=== GEMM Benchmark [{M} x {N} x {K}] ({dtype}) ===")
        print(f"PyTorch:      {torch_time:.3f} ms, {torch_tflops:.2f} TFLOPS")


def main():
    print("CUTLASS CUTE GEMM Benchmark")
    print("============================\n")

    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return

    print(f"Device: {torch.cuda.get_device_name()}\n")

    # Verify correctness
    if HAS_SAMATTN:
        print("Running correctness tests...\n")
        verify_correctness(128, 128, 128, torch.float32)
        verify_correctness(256, 256, 256, torch.float32)
        print()

    # Benchmark configurations
    configs = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        # Rectangular
        (1024, 512, 2048),
        (2048, 1024, 4096),
        # Batch-size like shapes
        (8192, 4096, 2048),
    ]

    for M, N, K in configs:
        run_benchmark(M, N, K, torch.float32)

    # FP16 benchmarks
    print("\n\n=== FP16 Benchmarks ===")
    for M, N, K in [(2048, 2048, 2048), (4096, 4096, 4096)]:
        run_benchmark(M, N, K, torch.float16)


if __name__ == "__main__":
    main()
