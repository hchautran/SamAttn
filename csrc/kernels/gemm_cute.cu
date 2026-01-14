#include "gemm_cute.cuh"

namespace samattn {

// Explicit instantiations
template void gemm_cute_launch<float>(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream
);

template void gemm_cute_launch<__half>(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K, cudaStream_t stream
);

} // namespace samattn
