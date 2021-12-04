cmake_minimum_required(VERSION 3.3)
project(multiply)
find_package(CUDA REQUIRED)

link_directories(
    /usr/local/cuda/lib64
    )
    
cuda_add_executable(multiply multiply_matrix.cu)
