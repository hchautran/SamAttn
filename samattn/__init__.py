"""
SamAttn: Custom GEMM and Flash Attention kernels using CUTLASS CUTE
"""

__version__ = "0.1.0"

try:
    from . import _C

    def gemm(A, B):
        """
        Matrix multiplication using CUTLASS CUTE kernel.

        Args:
            A: torch.Tensor of shape [M, K]
            B: torch.Tensor of shape [K, N]

        Returns:
            torch.Tensor of shape [M, N]
        """
        return _C.gemm_cute(A, B)

    __all__ = ["gemm"]

except ImportError as e:
    import warnings
    warnings.warn(f"Could not import native extension: {e}. You may need to build the package.")
    __all__ = []
