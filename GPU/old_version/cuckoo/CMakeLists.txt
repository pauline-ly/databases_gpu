cmake_minimum_required(VERSION 2.8)
project(linear)
find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61;-G;-g)
include_directories(/usr/local/cuda/samples/common/inc)

set(CMAKE_CXX_STANDARD 11)

set(D_S
        dy_hash.h
        cuckoo.cu cuckoo.h
        api.cpp api.h
        Main.cpp)
cuda_add_executable(linear ${D_S})










