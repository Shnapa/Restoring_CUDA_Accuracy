#include <cmath>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 16
__device__ double func(double x, double y) {
    double sums = 0.0;

    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++){
            sums += 1.0 / (5 * (i+2) + j + 3 + pow( x - 16*j , 6) + pow( y - 16*i , 6));
        }
    }
    return 1.0 / (0.002 + sums); 
}

__global__ void riemannSum(double x1, double x2,
                             double y1, double y2,
                             int max_iter,
                             double *result) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    double dx = (x2 - x1) / max_iter; 
    double dy = (y2 - y1) / max_iter;

    if (i < max_iter && j < max_iter) {
        double x = x1 + i * dx; 
        double y = y1 + j * dy;
        double f_val = func(x, y) * dx * dy;

        atomicAdd(result, f_val);
    }
}

double riemman_return(double x1, double x2, double y1, double y2, int max_iter){
    double h_result = 0.0, *d_result;
    cudaMalloc((void**)&d_result, sizeof(double));
    cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((max_iter + BLOCK_SIZE - 1) / BLOCK_SIZE, (max_iter + BLOCK_SIZE - 1) / BLOCK_SIZE);

    riemannSum<<<gridSize, blockSize>>>(x1, x2, y1, y2, max_iter, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    return h_result;
}

int main(){
	int n = 10000;
	double result = riemman_return(-50, 50, -50, 50, n);
	std::cout << "Result: " << result << std::endl;
}

