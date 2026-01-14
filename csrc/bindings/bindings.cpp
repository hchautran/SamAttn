#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../kernels/gemm_cute.cuh"

namespace samattn {

// PyTorch interface for GEMM
torch::Tensor gemm_cute(
    torch::Tensor A,
    torch::Tensor B
) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible dimensions for matrix multiplication");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A and B must have the same dtype");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    // Ensure contiguous
    A = A.contiguous();
    B = B.contiguous();

    auto stream = at::cuda::getCurrentCUDAStream();

    if (A.scalar_type() == torch::kFloat32) {
        gemm_cute_launch<float>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K,
            stream
        );
    } else if (A.scalar_type() == torch::kFloat16) {
        gemm_cute_launch<__half>(
            reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
            M, N, K,
            stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Only float32 and float16 are supported.");
    }

    return C;
}

} // namespace samattn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_cute", &samattn::gemm_cute, "GEMM using CUTLASS CUTE");
}
