#include <stdio.h>

// Size of Array
#define N 5120*100000

// CUDA Kernel
__global__ void add_vectors(int *a, int *b, int *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < N)
    {
        c[id] = a[id] + b[id];
    }
}

// Main Program
int main()
{
    // Number of bytes to allocate for N integers
    size_t size = N * sizeof(int);

    // Allocate memory for arrays a, b and c on Host (CPU)
    int *a = (int*)malloc(size);
    int *b = (int*)malloc(size);
    int *c = (int*)malloc(size);

    // Allocate memory for arrays d_a, d_b and d_c on Device (GPU)
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Initialize Arrays a and b
    for(int i=0; i<N; i++)
    {
        a[i] = 1;
        b[i] = 2;
    }

    // Cpoy data from CPU (Host) to GPU (Device)
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    // ThreadsPerBlock : Number of CUDA threads per Block
    // BlocksPerGrid : Number of blocks in grid
    int ThreadsPerBlock = 128;
    int BlocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;

    // Kernel Launch
    add_vectors<<< BlocksPerGrid,ThreadsPerBlock >>>(d_a, d_b, d_c);

    // Copy data from GPU (Device) to CPU (Host)
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify the results
    double tolerance = 1.0e-14;
    for(int i=0; i<N ; i++)
    {
        if( fabs(c[i] - 3) > tolerance)
        {
            printf("\nError: value of c[%d] - %d instead of 3\n", i, c[i]);
            exit(1);
        }
    }

    // Free CPU memory
    free(a);
    free(b);
    free(c);

    // Free GPU Memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("\n---------------------------\n");
        printf("__SUCCESS__\n");
        printf("---------------------------\n");
        printf("N                 = %d\n", N);
        printf("Threads Per Block = %d\n", ThreadsPerBlock);
        printf("Blocks In Grid    = %d\n", BlocksPerGrid);
        printf("---------------------------\n\n");

    return 0;
}
