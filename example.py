#!/usr/bin/env python3
"""
Simple example demonstrating SamAttn usage
"""

import torch

try:
    import samattn
    print("✓ samattn module loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import samattn: {e}")
    print("\nPlease build the package first:")
    print("  pip install -e .")
    exit(1)


def main():
    if not torch.cuda.is_available():
        print("✗ CUDA is not available")
        return

    device = torch.cuda.get_device_name()
    print(f"✓ Using device: {device}\n")

    # Test FP32 GEMM
    print("=" * 50)
    print("Testing FP32 GEMM")
    print("=" * 50)

    M, N, K = 1024, 1024, 1024
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')

    print(f"Matrix shapes: A={A.shape}, B={B.shape}")

    # Run custom kernel
    C_custom = samattn.gemm(A, B)
    print(f"Custom kernel output shape: {C_custom.shape}")

    # Run PyTorch reference
    C_torch = torch.matmul(A, B)

    # Check correctness
    max_diff = (C_custom - C_torch).abs().max().item()
    rel_error = ((C_custom - C_torch).abs() / (C_torch.abs() + 1e-5)).max().item()

    print(f"\nCorrectness check:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Max relative error: {rel_error:.6e}")

    if rel_error < 1e-3:
        print("  Status: ✓ PASSED")
    else:
        print("  Status: ✗ FAILED")

    # Test FP16 GEMM
    print("\n" + "=" * 50)
    print("Testing FP16 GEMM")
    print("=" * 50)

    A_fp16 = A.half()
    B_fp16 = B.half()

    print(f"Matrix shapes: A={A_fp16.shape}, B={B_fp16.shape}")

    C_custom_fp16 = samattn.gemm(A_fp16, B_fp16)
    C_torch_fp16 = torch.matmul(A_fp16, B_fp16)

    max_diff_fp16 = (C_custom_fp16 - C_torch_fp16).abs().max().item()
    rel_error_fp16 = ((C_custom_fp16 - C_torch_fp16).abs() / (C_torch_fp16.abs() + 1e-3)).max().item()

    print(f"\nCorrectness check:")
    print(f"  Max absolute difference: {max_diff_fp16:.6e}")
    print(f"  Max relative error: {rel_error_fp16:.6e}")

    if rel_error_fp16 < 0.1:  # FP16 has lower precision
        print("  Status: ✓ PASSED")
    else:
        print("  Status: ✗ FAILED")

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
    print("\nNext steps:")
    print("  - Run benchmarks: python benchmarks/bench_gemm.py")
    print("  - Implement your own kernels in csrc/kernels/")
    print("  - Check README.md for more details")


if __name__ == "__main__":
    main()
