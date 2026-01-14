import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

        # CMake configuration arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CUDA_ARCHITECTURES=80",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
        ]

        # Build configuration
        build_args = ["--config", "Release", "--verbose"]

        # Parallel build
        if hasattr(self, "parallel") and self.parallel:
            build_args += ["-j", str(self.parallel)]
        else:
            build_args += ["-j"]

        # Create build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Run CMake
        subprocess.check_call(
            ["cmake", str(Path(__file__).parent)] + cmake_args,
            cwd=build_temp
        )

        # Build
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp
        )


setup(
    name="samattn",
    version="0.1.0",
    author="Your Name",
    description="Custom GEMM and Flash Attention kernels using CUTLASS CUTE",
    long_description=open("README.md").read() if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    packages=["samattn"],
    ext_modules=[CMakeExtension("samattn._C")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
    ],
)
