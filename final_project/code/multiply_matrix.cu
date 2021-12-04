#include <cuda_runtime.h>
#include <pybind11.h>
#include <numpy.h>
#include <stl.h>
#include <chrono>
#include <iostream>

#define CUDA_HOST_ALLOC 1

void cpu_matmul(float* h_a, float* h_b, float* h_c, int M, int N, int K);

__global__ void gpu_global_matmul(float* a, float* b, float* c, int M, int N, int K);

__global__ void gpu_shared_matmul(float* a, float* b, float* c, int M, int N, int K);

int main(){
    srand(0);

    int M, N, K;

    M = 100;
    N = 100;
    K = 100;

    unsigned int sizeOfA = sizeof(float)*M*N;
    unsigned int sizeOfB = sizeof(float)*N*K;
    unsigned int sizeOfC = sizeof(float)*M*K;

    float *h_a, *h_b, *h_c, *h_c_true;
    if(CUDA_HOST_ALLOC){
        cudaHostAlloc((void**)&h_a, sizeOfA, 0);
        cudaHostAlloc((void**)&h_b, sizeOfB, 0);
        cudaHostAlloc((void**)&h_c, sizeOfC, 0);
        cudaHostAlloc((void**)&h_c_true, sizeOfC, 0);
    }
    else{
        h_a = (float*)malloc(sizeOfA);
        h_b = (float*)malloc(sizeOfB);
        h_c = (float*)malloc(sizeOfC);
        h_c_true = (float*)malloc(sizeOfC);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, sizeOfA);
    cudaMalloc((void **)&d_b, sizeOfB);
    cudaMalloc((void **)&d_c, sizeOfC);

    // For testing
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_a[i * N + j] = rand() % 1024;
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            h_b[i * K + j] = rand() % 1024;
        }
    }

    auto gpu_start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_a, h_a, sizeOfA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeOfB, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 16;
    unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dim_grid(grid_cols, grid_rows);
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);

    gpu_global_matmul<<<dim_grid, dim_block>>>(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(h_c, d_c, sizeOfC, cudaMemcpyDeviceToHost);

    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
    std::cout << "GPU execution time[us]: " << gpu_duration.count() << std::endl;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_matmul(h_a, h_b, h_c_true, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    std::cout << "CPU execution time[us]: " << cpu_duration.count() << std::endl;

    char match = 1;
    float treshold = 1e-12;
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < K; ++j){
            if(abs(h_c[i*K + j] - h_c_true[i*K + j]) > treshold){
                match = 0;
                std::cout << "Error on index [" << i << ", " << j <<"]" << std::endl;
                std::cout << "Expected " << h_c_true[i*K + j] << " but got " << h_c[i*K + j] << std::endl;
            }
        }
    }
    if(match){
        std::cout << "Correct results!! GPU was " << cpu_duration/gpu_duration << " times faster than CPU" << std::endl;
    }
    
    if(CUDA_HOST_ALLOC){
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        cudaFreeHost(h_c_true);
    }
    else{
        free(h_a);
        free(h_b);
        free(h_c);
        free(h_c_true);
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}

void cpu_matmul(float* h_a, float* h_b, float* h_c, int M, int N, int K){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            float sum = 0;
            for(int k = 0; k < N; k++){
                sum += h_a[i*N + k] * h_b[k*K + j];
            }
            h_c[i*K + j] = sum;
        }
    }
}

__global__ void gpu_global_matmul(float* a, float* b, float* c, int M, int N, int K){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx < K && idy < M) {
        float sum = 0;
        for(int i = 0; i < N; i++) {
            sum += a[idy * N + i] * b[i * K + idx];
        }
        c[idy * K + idx] = sum;
    }
}

__global__ void gpu_shared_matmul(float* a, float* b, float* c, int M, int N, int K){

}

void global_matmul(){
    
}

PYBIND11_MODULE(gpu_library, m)
{
  m.def("multiply_with_scalar", map_array<double>);
}