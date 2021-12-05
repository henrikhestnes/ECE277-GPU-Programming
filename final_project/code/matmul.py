import numpy as np
import time

import sys
sys.path.append('build')
import gpu_library

def python_multiply(A, B):
    start = time.perf_counter()
    C = A @ B
    end = time.perf_counter()
    print(f"Python matrix multiplication took {end-start} seconds")
    return C


def main():
    # print("***********************MATRIX MULTIPLICATION***********************")
    # one_more_time = "y"
    # while(one_more_time == "y"):
    #     print("You have two matrices A(MxN) and B(NxK) that you want to multiply.")
    #     M, N, K = input("Enter their dimensions(M, N, K): ").split(",")
    #     M, N, K = int(M), int(N), int(K)

    #     A = np.random.rand(M, N)
    #     B = np.random.rand(N, K)

    #     own_elements = input("Do you want to specify your own elements?(y/n) ").lower()

    #     if own_elements == "y":
    #         for i in range(M):
    #             for j in range(N):
    #                 A[i, j] = float(input(f"Chose element ({i}, {j}) of matrix A: "))
    #         for i in range(N):
    #             for j in range(K):
    #                 B[i, j] = float(input(f"Chose element ({i}, {j}) of matrix B: "))

    #     python_multiply(A, B)



    #     one_more_time = input("Do you want to calculate anoter matrix multiply?(y/n) ").lower()
    M, N, K = 1000, 1000, 1000
    A = np.random.rand(M, N)
    B = np.random.rand(N, K)
    
    start = time.perf_counter()
    C_CPU_GLOBAL = np.zeros(M*K)
    gpu_library.cpu_matrix_multiply(A.reshape(M*N), B.reshape(N*K), C_CPU_GLOBAL, M, N, K)
    end = time.perf_counter()
    print("CPU in C++ time: " + str(end-start))

    start = time.perf_counter()
    C_GPU = np.zeros(M*K)
    gpu_library.global_matrix_multiply(A.reshape(M*N), B.reshape(N*K), C_GPU, M, N, K)
    end = time.perf_counter()
    print("GPU without shared memory time: " + str(end-start))

    C_PYTHON = python_multiply(A, B)

    print(f"\nCPU in C++ result correct: {np.allclose(C_CPU_GLOBAL.reshape(M, K),C_PYTHON)}")
    print(f"GPU without shared memory correct: {np.allclose(C_GPU.reshape(M, K),C_PYTHON)}")


main()