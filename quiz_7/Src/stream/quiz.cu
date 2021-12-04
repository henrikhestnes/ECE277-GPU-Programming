#include <stdio.h>
#include <helper_functions.h>
#include <helper_cuda.h>


// DO NOT change the kernel function
__global__ void vector_add(int *a, int *b, int *c)
{
// DO NOT change the kernel function
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	c[index] = a[index] + b[index];
}


#define N (2048*2048)
#define THREADS_PER_BLOCK 128
#define N_STREAMS 4

int main()
{
	int size = N * sizeof( int );

    int *a, *b, *c, *golden;
	cudaHostAlloc((void**)&a, size, 0);
	cudaHostAlloc((void**)&b, size, 0);
	cudaHostAlloc((void**)&c, size, 0);

	int *d_a, *d_b, *d_c;
	cudaMalloc( (void **) &d_a, size );
	cudaMalloc( (void **) &d_b, size );
	cudaMalloc( (void **) &d_c, size );

	golden = (int *)malloc(size);

	for( int i = 0; i < N; i++ )
	{
		a[i] = b[i] = i;
		golden[i] = a[i] + b[i];
		c[i] = 0;
	}

	cudaStream_t stream[N_STREAMS];

	for (int i = 0; i < N_STREAMS; ++i) {
		cudaStreamCreate(&stream[i]);
	}
	for (int i = 0; i < N_STREAMS; i++) {

	bool pass = true;
	for (int i = 0; i < N; i++) {
		if (golden[i] != c[i])
			pass = false;
	}
	
	if (pass)
		printf("PASS\n");
	else
		printf("FAIL\n");

	printf("Name: Henrik Albin Larsson Hestnes\nStudent ID: U09103474\n");

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
	free(golden);
	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_c );
	
	return 0;
} 