#include <cmath.h>
#include <vector>

__device__ func(double x, double y) {
    double sum1 = 0, sum2 = 0;

    for (int i = 1; i < 5; i++) {
        sum1 += i * cos((i + 1) * x + 1);
        sum2 += i * cos((i + 1) * y + 1);
    }

    return - sum1 * sum2;
}

__global__ double riemannSum(double x1, double x2,
                             double y1, double y2,
                             int max_interations,
//                             int steps_x, int steps_y,
//                             double abs_err, double rel_err,
                             double *result) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int dx, dy = abs(x1 - x2) / max_interations, abs(y1 - y2) / max_interations;

    if (i < n && j < n) {
        double x = x1 + (i + 0.5) * dx;
        double y = y1 + (j + 0.5) * dy;
        double f_val = func(x, y) * dx * dy;

        atomicAdd(result, f_val);
}

int main(){
    double h_result = 0.0, *d_result;
    cudaMalloc((void**)&d_result, sizeof(double));

    const int BLOCK_SIZE = 16;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice);

    riemannSum<<<gridSize, BLOCK_SIZE>>>(-100, 100, -100, 100, 30, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    printf("%lf\n", h_result);
}