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
    A = np.linspace(1.0, 100.0, M*N)
    B = np.linspace(1.0, 100.0, N*K)
    C_GPU = np.zeros(M*K)

    print(f"A: {A}\nB: {B}")
    start = time.perf_counter()
    gpu_library.global_matrix_multiply(A, B, C_GPU, M, N, K)
    end = time.perf_counter()
    print("GPU time: " + str(end-start))
    C_CPU = python_multiply(A, B)
    print(f"C: {C_GPU}")
    print("results match: " + str(np.allclose(C_GPU,C_CPU)))



main()