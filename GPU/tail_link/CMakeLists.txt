cmake_minimum_required(VERSION 2.8)
project(tlink)
include_directories(/usr/local/cuda-8.0/samples/common/inc)

set(CMAKE_CXX_STANDARD 11)
find_package(CUDA QUIET REQUIRED)


set(SOURCE_FILES hash.cu)
cuda_add_executable(tlink ${SOURCE_FILES})


