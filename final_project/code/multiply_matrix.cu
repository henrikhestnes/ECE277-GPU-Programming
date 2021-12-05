#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <chrono>
#include <iostream>

namespace py = pybind11;

void cpu_matmul(py::array_t<double> a, py::array_t<double> b, py::array_t<double> c, int M, int N, int K){  
    pybind11::buffer_info h_buff_a = a.request();
    pybind11::buffer_info h_buff_b = b.request();
    pybind11::buffer_info h_buff_c = c.request();

    double *h_a, *h_b, *h_c;
    h_a = reinterpret_cast<double*>(h_buff_a.ptr);
    h_b = reinterpret_cast<double*>(h_buff_b.ptr);
    h_c = reinterpret_cast<double*>(h_buff_c.ptr);

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

__global__ void gpu_global_matmul(double* a, double* b, double* c, int M, int N, int K){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx < K && idy < M) {
        double sum = 0;
        for(int i = 0; i < N; i++) {
            sum += a[idy * N + i] * b[i * K + idx];
        }
        c[idy * K + idx] = sum;
    }
}

__global__ void gpu_shared_matmul(float* a, float* b, float* c, int M, int N, int K){

}


void global_matmul(py::array_t<double> a, py::array_t<double> b, py::array_t<double> c, int M, int N, int K){
    unsigned int sizeOfA = sizeof(double)*M*N;
    unsigned int sizeOfB = sizeof(double)*N*K;
    unsigned int sizeOfC = sizeof(double)*M*K;
    
    pybind11::buffer_info h_buff_a = a.request();
    pybind11::buffer_info h_buff_b = b.request();
    pybind11::buffer_info h_buff_c = c.request();

    double *h_a, *h_b, *h_c;
    h_a = reinterpret_cast<double*>(h_buff_a.ptr);
    h_b = reinterpret_cast<double*>(h_buff_b.ptr);
    h_c = reinterpret_cast<double*>(h_buff_c.ptr);

    double *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, sizeOfA);
    cudaMalloc((void **)&d_b, sizeOfB);
    cudaMalloc((void **)&d_c, sizeOfC);

    cudaMemcpy(d_a, h_a, sizeOfA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeOfB, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 16;
    unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dim_grid(grid_cols, grid_rows);
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);

    gpu_global_matmul<<<dim_grid, dim_block>>>(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(h_c, d_c, sizeOfC, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

PYBIND11_MODULE(gpu_library, m){
    m.doc() = "Plugin for doing GPU accelerated matrix multiply in python";
    m.def("global_matrix_multiply", &global_matmul);
    m.def("cpu_matrix_multiply", &cpu_matmul);
}