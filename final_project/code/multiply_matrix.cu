#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <chrono>
#include <iostream>


#define BLOCK_SIZE 16


//*************************CUDA KERNEL CODE*************************


__global__ void gpu_global_matmul(const double* a, const double* b, double* c, int M, int N, int K){
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


__global__ void gpu_shared_matmul(const double* A, const double* B, double* C, int M, int N, int K){
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    
    double* C_block = &C[K * block_row * BLOCK_SIZE *  + block_col * BLOCK_SIZE]; //STRIDE?

    double C_val = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for(int i = 0; i < (double)N/(double)BLOCK_SIZE; i++){
        __shared__ double A_shared[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double B_shared[BLOCK_SIZE][BLOCK_SIZE];
        
        const double* A_block = &A[N * block_row * BLOCK_SIZE + i * BLOCK_SIZE];
        const double* B_block = &B[K * i * BLOCK_SIZE + block_col * BLOCK_SIZE];


        A_shared[row][col] = A_block[N * row + col];
        B_shared[row][col] = B_block[K * row + col];

        __syncthreads();

        for(int j = 0; j < BLOCK_SIZE; j++){
            C_val += A_shared[row][j] * B_shared[j][col];
        }

        __syncthreads();

        C_block[K * row + col] = C_val;
    }

}


//*************************BINDED C++ CODE*************************


namespace py = pybind11;


void cpu_matmul(const py::array_t<double> a, const py::array_t<double> b, py::array_t<double> c, int M, int N, int K){  
    const pybind11::buffer_info h_buff_a = a.request();
    const pybind11::buffer_info h_buff_b = b.request();
    pybind11::buffer_info h_buff_c = c.request();

    const double *h_a, *h_b;
    double *h_c;
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

enum class MEM_TYPE{
    SHARED,
    GLOBAL
};


void gpu_matmul(const py::array_t<const double> a, const py::array_t<const double> b, py::array_t<double> c, int M, int N, int K, MEM_TYPE memory){
    unsigned int sizeOfA = sizeof(double)*M*N;
    unsigned int sizeOfB = sizeof(double)*N*K;
    unsigned int sizeOfC = sizeof(double)*M*K;
    
    const pybind11::buffer_info h_buff_a = a.request();
    const pybind11::buffer_info h_buff_b = b.request();
    pybind11::buffer_info h_buff_c = c.request();

    const double *h_a, *h_b;
    double *h_c;
    h_a = reinterpret_cast<double*>(h_buff_a.ptr);
    h_b = reinterpret_cast<double*>(h_buff_b.ptr);
    h_c = reinterpret_cast<double*>(h_buff_c.ptr);

    double *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, sizeOfA);
    cudaMalloc((void **)&d_b, sizeOfB);
    cudaMalloc((void **)&d_c, sizeOfC);

    cudaMemcpy(d_a, h_a, sizeOfA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeOfB, cudaMemcpyHostToDevice);


    switch(memory){
        case MEM_TYPE::GLOBAL: {
            unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
            unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            dim3 dim_grid(grid_cols, grid_rows);
            dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
            gpu_global_matmul<<<dim_grid, dim_block>>>(d_a, d_b, d_c, M, N, K);
            break;
        }
        case MEM_TYPE::SHARED: {
            dim3 dim_grid(ceil((double)K / (double)BLOCK_SIZE), ceil((double)M / (double)BLOCK_SIZE));
            dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
            gpu_shared_matmul<<<dim_grid, dim_block>>>(d_a, d_b, d_c, M, N, K);
            break;
        }
    }


    cudaMemcpy(h_c, d_c, sizeOfC, cudaMemcpyDeviceToHost);

    if(memory == MEM_TYPE::SHARED){
        std::cout << "C[0]: " << h_c[0] << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void global_matmul(const py::array_t<const double> a, const py::array_t<const double> b, py::array_t<double> c, int M, int N, int K){
    gpu_matmul(a, b, c, M, N, K, MEM_TYPE::GLOBAL);
}

void shared_matmul(const py::array_t<const double> a, const py::array_t<const double> b, py::array_t<double> c, int M, int N, int K){
    gpu_matmul(a, b, c, M, N, K, MEM_TYPE::SHARED);
}


//*************************PYBIND11 BINDINGS*************************


PYBIND11_MODULE(gpu_library, m){
    m.doc() = "Plugin for doing GPU accelerated matrix multiply in python";
    m.def("cuda_global_matrix_multiply", &global_matmul);
    m.def("cuda_shared_matrix_multiply", &shared_matmul);
    m.def("cpu_matrix_multiply", &cpu_matmul);
}