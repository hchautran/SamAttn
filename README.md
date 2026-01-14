# SamAttn

Custom GEMM and Flash Attention kernels using CUTLASS CUTE and PyTorch integration.

## Overview

This repository provides custom CUDA kernels implemented with NVIDIA's CUTLASS CUTE (C++ Template Abstractions for Linear Algebra) library, with seamless PyTorch integration for benchmarking and production use.

## Features

- Custom GEMM (General Matrix Multiply) kernels using CUTLASS CUTE
- PyTorch C++ extensions for easy integration
- Comprehensive benchmarking suite (both standalone CUDA and Python)
- Support for FP32 and FP16 datatypes
- CMake-based build system with automatic CUTLASS fetching

## Requirements

- CUDA Toolkit 11.8+ (12.x recommended)
- CMake 3.18+
- C++17 compatible compiler
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with compute capability 8.0+ (Ampere or newer recommended)

## Installation

### Option 1: Install as Python Package

```bash
# Install in development mode
pip install -e .

# Or build and install
pip install .
```

The setup script will:
1. Automatically fetch CUTLASS from GitHub if not found
2. Build CUDA kernels
3. Create PyTorch extension bindings
4. Install the `samattn` Python package

### Option 2: Build with CMake Directly

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
cmake --build . -j
```

This will build:
- `libsamattn_kernels.a` - Static library with CUDA kernels
- `samattn_ext.so` - PyTorch extension module (if PyTorch is found)
- `bench_gemm` - Standalone CUDA benchmark executable

### CUDA Architecture Configuration

By default, the build targets Ampere (80), Ada (86,89), and Hopper (90) architectures. To customize:

```bash
# For specific GPU (e.g., RTX 3090 = compute capability 8.6)
cmake .. -DCMAKE_CUDA_ARCHITECTURES="86"

# Multiple architectures
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;80;86"
```

Common architectures:
- `75`: Turing (RTX 20 series)
- `80`: Ampere (A100, RTX 30 series)
- `86`: Ampere (RTX 3090)
- `89`: Ada Lovelace (RTX 40 series)
- `90`: Hopper (H100)

## Project Structure

```
SamAttn/
├── CMakeLists.txt              # Main CMake configuration
├── setup.py                    # Python package setup
├── pyproject.toml              # Python project metadata
├── README.md                   # This file
├── csrc/                       # C++/CUDA source code
│   ├── kernels/                # CUDA kernel implementations
│   │   ├── gemm_cute.cuh       # GEMM kernel header
│   │   └── gemm_cute.cu        # GEMM kernel implementation
│   └── bindings/               # PyTorch C++ extensions
│       └── bindings.cpp        # Python binding interface
├── samattn/                    # Python package
│   └── __init__.py             # Package initialization
└── benchmarks/                 # Benchmarking tools
    ├── bench_gemm.cu           # Standalone CUDA benchmark
    └── bench_gemm.py           # Python benchmark script
```

## Usage

### Python Interface

```python
import torch
import samattn

# Create random matrices
M, N, K = 2048, 2048, 2048
A = torch.randn(M, K, dtype=torch.float32, device='cuda')
B = torch.randn(K, N, dtype=torch.float32, device='cuda')

# Run custom GEMM kernel
C = samattn.gemm(A, B)

# Compare with PyTorch
C_torch = torch.matmul(A, B)

# Check correctness
print(f"Max difference: {(C - C_torch).abs().max().item()}")
```

### FP16 Support

```python
# FP16 is automatically supported
A_fp16 = torch.randn(M, K, dtype=torch.float16, device='cuda')
B_fp16 = torch.randn(K, N, dtype=torch.float16, device='cuda')

C_fp16 = samattn.gemm(A_fp16, B_fp16)
```

## Benchmarking

### Python Benchmarks

```bash
python benchmarks/bench_gemm.py
```

This will:
- Run correctness tests
- Benchmark various matrix sizes
- Compare against PyTorch native GEMM
- Report performance in TFLOPS and speedup

### CUDA Benchmarks

```bash
# Build first
mkdir build && cd build
cmake .. && cmake --build . -j

# Run standalone benchmark
./bench_gemm
```

This compares against cuBLAS (the highly optimized NVIDIA BLAS library).

## Development

### Adding New Kernels

1. Create kernel implementation in `csrc/kernels/`:
   ```cpp
   // my_kernel.cuh
   #pragma once
   #include <cute/tensor.hpp>

   template <typename Element>
   void my_kernel_launch(...);
   ```

2. Add bindings in `csrc/bindings/bindings.cpp`:
   ```cpp
   torch::Tensor my_kernel_wrapper(torch::Tensor input) {
       // Implementation
   }

   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
       m.def("my_kernel", &my_kernel_wrapper);
   }
   ```

3. Export in `samattn/__init__.py`:
   ```python
   from . import _C

   def my_kernel(input):
       return _C.my_kernel(input)
   ```

### Building in Debug Mode

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

### Cleaning Build

```bash
# Clean CMake build
rm -rf build/

# Clean Python build artifacts
pip uninstall samattn
rm -rf *.egg-info build/ dist/ samattn/*.so
```

## Performance Tips

1. **Tune tile sizes**: Modify `kTileM`, `kTileN`, `kTileK` in kernel implementations for your GPU
2. **Use FP16**: Half precision can be 2x faster on modern GPUs
3. **Profile with Nsight**: Use `nsys` or `ncu` for detailed performance analysis
4. **Benchmark your shapes**: Kernel performance varies significantly with matrix dimensions

## Troubleshooting

### "CUTLASS not found" Error

The build system will automatically fetch CUTLASS. If you have a local installation:

```bash
cmake .. -DCUTLASS_DIR=/path/to/cutlass
```

### "PyTorch not found" Warning

Ensure PyTorch is installed in your Python environment:

```bash
python -c "import torch; print(torch.__version__)"
```

### CUDA Compilation Errors

Check your CUDA version:

```bash
nvcc --version
```

Ensure it's compatible with your PyTorch installation.

### Runtime Errors

Verify GPU compute capability:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Resources

- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CUTLASS CUTE Tutorial](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [PyTorch C++ Extension Guide](https://pytorch.org/tutorials/advanced/cpp_extension.html)

## License

MIT License (or your preferred license)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.
