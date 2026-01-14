cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(SamAttn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# CUDA architectures - modify based on your GPU
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)  # Ampere, Ada, Hopper
endif()

# Find or fetch CUTLASS
find_package(CUTLASS QUIET)
if(NOT CUTLASS_FOUND)
  message(STATUS "CUTLASS not found, fetching from GitHub...")
  include(FetchContent)
  FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v3.5.1
  )
  FetchContent_MakeAvailable(cutlass)
  set(CUTLASS_INCLUDE_DIR ${cutlass_SOURCE_DIR}/include)
else()
  message(STATUS "Found CUTLASS: ${CUTLASS_DIR}")
endif()

# Find PyTorch (optional, for Python bindings)
execute_process(
  COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

if(TORCH_CMAKE_PREFIX_PATH)
  set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
  find_package(Torch REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  set(BUILD_PYTHON_BINDINGS ON)
  message(STATUS "Found PyTorch: ${TORCH_INSTALL_PREFIX}")
else()
  set(BUILD_PYTHON_BINDINGS OFF)
  message(WARNING "PyTorch not found, skipping Python bindings")
endif()

# CUDA flags for CUTLASS CUTE
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc
  ${CUTLASS_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

# Kernel library
file(GLOB_RECURSE KERNEL_SOURCES
  csrc/kernels/*.cu
  csrc/kernels/*.cpp
)

if(KERNEL_SOURCES)
  add_library(samattn_kernels STATIC ${KERNEL_SOURCES})
  target_include_directories(samattn_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CUTLASS_INCLUDE_DIR}
  )
  target_link_libraries(samattn_kernels PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
  )
  set_target_properties(samattn_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  file(GLOB_RECURSE BINDING_SOURCES csrc/bindings/*.cpp)
  if(BINDING_SOURCES AND KERNEL_SOURCES)
    add_library(samattn_ext SHARED ${BINDING_SOURCES})
    target_link_libraries(samattn_ext PRIVATE
      samattn_kernels
      ${TORCH_LIBRARIES}
    )
    set_target_properties(samattn_ext PROPERTIES
      PREFIX ""
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/samattn
    )
    # Platform-specific extension
    if(APPLE)
      set_target_properties(samattn_ext PROPERTIES SUFFIX ".so")
    endif()
  endif()
endif()

# Benchmarks
option(BUILD_BENCHMARKS "Build benchmark executables" ON)
if(BUILD_BENCHMARKS)
  file(GLOB BENCHMARK_SOURCES benchmarks/*.cu)
  foreach(benchmark_file ${BENCHMARK_SOURCES})
    get_filename_component(benchmark_name ${benchmark_file} NAME_WE)
    add_executable(${benchmark_name} ${benchmark_file})
    if(TARGET samattn_kernels)
      target_link_libraries(${benchmark_name} PRIVATE
        samattn_kernels
        CUDA::cudart
        CUDA::cuda_driver
      )
    else()
      target_link_libraries(${benchmark_name} PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
      )
      target_include_directories(${benchmark_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc
        ${CUTLASS_INCLUDE_DIR}
      )
    endif()
  endforeach()
endif()
