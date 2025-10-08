#include <stdio.h>

// Kernel function to print "Hello, world!" from the GPU
__global__ void helloFromGPU()
{
    printf("Hello, world from GPU\n");
}

int main()
{
    // Print "Hello, World!" from the CPU
    printf("Hello, World from CPU\n");

    // Launch kernel to print "Hello, World!" from the GPU
    helloFromGPU<<<1, 10>>>();

    // Synchronize to ensure all printf statements from the GPU are executed
    cudaDeviceSynchronize();

    return 0;
}
