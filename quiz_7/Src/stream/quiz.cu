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
	}	int ndata = N;	int nsdata = ndata / N_STREAMS;	int iBytes = nsdata * sizeof(int);	int nblock = THREADS_PER_BLOCK;	int ngrid = (nsdata + nblock - 1) / nblock;	for (int i = 0; i < N_STREAMS; i++) {		int offset = i * nsdata;		cudaMemcpyAsync(&d_a[offset], &a[offset], iBytes, cudaMemcpyHostToDevice, stream[i]);		cudaMemcpyAsync(&d_b[offset], &b[offset], iBytes, cudaMemcpyHostToDevice, stream[i]);	}	for (int i = 0; i < N_STREAMS; i++) {		int offset = i * nsdata;		vector_add << < ngrid, nblock, 0, stream[i] >> > (&d_a[offset], &d_b[offset], &d_c[offset]);	}	for (int i = 0; i < N_STREAMS; i++) {		int offset = i * nsdata;		cudaMemcpyAsync(&c[offset], &d_c[offset], iBytes, cudaMemcpyDeviceToHost, stream[i]);	}	for (int i = 0; i < N_STREAMS; i++) {		cudaStreamSynchronize(stream[i]);	}
	for (int i = 0; i < N_STREAMS; i++) {		cudaStreamDestroy(stream[i]);	}

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
