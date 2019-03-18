cmake_minimum_required(VERSION 2.8)
project(linear)
find_package(CUDA QUIET REQUIRED)
set(CMAKE_CXX_STANDARD 11)
include_directories(/usr/local/cuda/samples/common/inc)


set(D_S
        dy_hash.h
        linear_probe.cu linear_probe.h
        api.cpp api.h
        Main.cpp)
cuda_add_executable(linear ${D_S})










