cmake_minimum_required(VERSION 2.8)
project(linear)
find_package(CUDA QUIET REQUIRED)

#set(CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61;-G;-g;--ptxas-options=-v)
set(CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61;-O3;--ptxas-options=-v)
#set(CMAKE_CXX_FLAGS --gtest_filter=HashResizeTest.*)
include_directories(/usr/local/cuda/samples/common/inc)

set(CMAKE_CXX_STANDARD 11)

set(D_S
        dy_hash.h gputimer.h
        cuckoo.cu cuckoo.h
        api.cpp api.h
        Main.cpp
        mt19937ar.cpp mt19937ar.h
        random_numbers.cpp random_numbers.h
        cnmem.cpp cnmem.h)
cuda_add_executable(linear ${D_S})
target_link_libraries(linear pthread)
target_link_libraries(linear gmock)
target_link_libraries(linear gtest)









