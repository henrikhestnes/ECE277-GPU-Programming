import numpy as np
import time

import sys
sys.path.append('build')
import gpu_library

print("***********************MATRIX MULTIPLICATION***********************")
one_more_time = "y"
while(one_more_time == "y"):
    print("\nYou have two matrices A(MxN) and B(NxK) that you want to multiply.")
    while True:
        try:
            M, N, K = input("Enter their dimensions seperated by comma(M, N, K): ").split(",")
            M, N, K = abs(int(M)), abs(int(N)), abs(int(K))
            break
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            print("Wrong format on input, please try again")

    A = np.random.rand(M, N)
    B = np.random.rand(N, K)

    own_elements = input("Do you want to specify your own elements?(y/n) ").lower()
    while own_elements != 'y' and own_elements != 'n':
        print("Please answer by typing 'y' or 'n'")
        own_elements = input("Do you want to specify your own elements?(y/n) ").lower()


    if own_elements == "y":
        for i in range(M):
            for j in range(N):
                while True:
                    try:
                        A[i, j] = float(input(f"Chose element ({i}, {j}) of matrix A: "))
                        break
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except:
                        print("Wrong format on input, please try again")

        for i in range(N):
            for j in range(K):
                while True:
                    try:
                        B[i, j] = float(input(f"Chose element ({i}, {j}) of matrix B: "))
                        break
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except:
                        print("Wrong format on input, please try again")


    C_TRUE = A @ B

    compare_execution_time = input("Do you want to compare the execution times?(y/n) ").lower()
    while compare_execution_time != 'y' and compare_execution_time != 'n':
        print("Please answer by typing 'y' or 'n'")
        compare_execution_time = input("Do you want to compare the execution times?(y/n) ").lower()

    if compare_execution_time == 'y':
        start = time.perf_counter()
        C_CPU = np.zeros(M*K)
        gpu_library.cpu_matrix_multiply(A.reshape(M*N), B.reshape(N*K), C_CPU, M, N, K)
        end = time.perf_counter()
        print("\nCPU in C++ time: " + str(end-start))


        start = time.perf_counter()
        C_GPU_GLOBAL = np.zeros(M*K)
        gpu_library.cuda_global_matrix_multiply(A.reshape(M*N), B.reshape(N*K), C_GPU_GLOBAL, M, N, K)
        end = time.perf_counter()
        print("GPU with only global memory time: " + str(end-start))


        start = time.perf_counter()
        C_GPU_SHARED = np.zeros(M*K)
        gpu_library.cuda_shared_matrix_multiply(A.reshape(M*N), B.reshape(N*K), C_GPU_SHARED, M, N, K)
        end = time.perf_counter()
        print("GPU with shared memory time: " + str(end-start))


        start = time.perf_counter()
        C_PYTHON = A @ B
        end = time.perf_counter()
        print("Python time: " + str(end-start))


        print(f"\nCPU in C++ result correct: {np.allclose(C_CPU.reshape(M, K),C_PYTHON)}")
        print(f"GPU with only global memory correct: {np.allclose(C_GPU_GLOBAL.reshape(M, K),C_PYTHON)}")
        print(f"GPU with shared memory correct: {np.allclose(C_GPU_SHARED.reshape(M, K),C_PYTHON)}")
    
    else:
        C_GPU_SHARED = np.zeros(M*K)
        gpu_library.cuda_shared_matrix_multiply(A.reshape(M*N), B.reshape(N*K), C_GPU_SHARED, M, N, K)

    if np.allclose(C_GPU_SHARED.reshape(M, K),C_TRUE):
        see_results = input("Do you want to see the result?(y/n) ").lower()
        while see_results != 'y' and see_results != 'n':
            print("Please answer by typing 'y' or 'n'")
            see_results = input("Do you want to see the result?(y/n) ").lower()

        if see_results == 'y':
            print(f"\nResult: \n{C_GPU_SHARED.reshape(M, K)}")
    else:
        print(f"CUDA encountered a problem, so the resulting matrix is wrong")

    
    one_more_time = input("\nDo you want to calculate another matrix multiply?(y/n) ").lower()
    while one_more_time != 'y' and one_more_time != 'n':
        print("Please answer by typing 'y' or 'n'")
        one_more_time = input("Do you want to calculate another matrix multiply?(y/n) ").lower()