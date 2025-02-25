#include <stdio.h>
#include <cuda_runtime_api.h>

__global__ void helloGPU() {
    printf("Hello GPU\n");
}

int main() {
    helloGPU<<<1, 12>>>();

    cudaDeviceSynchronize();
    return 0;
}