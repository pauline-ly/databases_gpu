cmake_minimum_required(VERSION 2.8)
project(slabhash)
#SET(CMAKE_C_COMPILER “/Applications/Xcode.app/Contents/Developer/usr/bin/gcc”)
#SET(CMAKE_CXX_COMPILER “/Applications/Xcode.app/Contents/Developer/usr/bin/g++”)
message(STATUS "optional:-std=c++11")
find_package(CUDA QUIET REQUIRED)



#set(CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61;-G;-g;--ptxas-options=-v)
set(CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61;-O3;--ptxas-options=-v)
#set(CMAKE_CXX_FLAGS --gtest_filter=HashResizeTest.*)
include_directories(/usr/local/cuda/samples/common/inc)
include_directories(..)

set(CMAKE_CXX_STANDARD 11)

set(D_S
        Main.cpp
        slab_hash.cu slab_hash.h
        )
cuda_add_executable(linear ${D_S})
target_link_libraries(linear pthread)
target_link_libraries(linear gmock)
target_link_libraries(linear gtest)
#target_link_libraries(linear cudart)










