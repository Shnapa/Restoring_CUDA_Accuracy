#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <regex>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

void parseDimensionsFromFilename(const std::string& filename, size_t& m, size_t& k, size_t& n) {
    std::regex pattern(".*_(\\d+)x(\\d+)x(\\d+)\\.txt");
    std::smatch match;
    if (std::regex_match(filename, match, pattern)) {
        m = std::stoi(match[1]);
        k = std::stoi(match[2]);
        n = std::stoi(match[3]);
    } else {
        throw std::invalid_argument("Filename does not match expected format: " + filename);
    }
}

int loadHalfMatricesFromFileArray(const std::string &filePath, __half* A, size_t A_elements, __half* B, size_t B_elements) {
    std::ifstream file(filePath);
    if (!file.is_open()) return -1;

    std::string line;
    size_t count = 0;

    // Read matrix A
    while (count < A_elements && std::getline(file, line)) {
        std::istringstream iss(line);
        float val;
        while (iss >> val && count < A_elements) {
            A[count++] = __float2half(val);
        }
    }

    count = 0;
    while (count < B_elements && std::getline(file, line)) {
        std::istringstream iss(line);
        float val;
        while (iss >> val && count < B_elements) {
            B[count++] = __float2half(val);
        }
    }

    return 0;
}

__global__ void matrixMultiplyWMMA(const __half *A, const __half *B, float *C, int m, int n, int k) {
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    if ((warpM + 1) * WMMA_M > m || (warpN + 1) * WMMA_N > n) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < k; i += WMMA_K) {
        if (i + WMMA_K <= k) {
            wmma::load_matrix_sync(a_frag, A + warpM * WMMA_M * k + i, k);
            wmma::load_matrix_sync(b_frag, B + i * n + warpN * WMMA_N, n);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    wmma::store_matrix_sync(C + warpM * WMMA_M * n + warpN * WMMA_N, c_frag, n, wmma::mem_row_major);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file_path>" << std::endl;
        return 1;
    }

    const std::string filePath = argv[1];
    size_t m, k, n;
    try {
        parseDimensionsFromFilename(filePath, m, k, n);
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    if (m % WMMA_M != 0 || k % WMMA_K != 0 || n % WMMA_N != 0) {
        std::cerr << "Error: WMMA requires matrix sizes to be multiples of 16." << std::endl;
        return 1;
    }

    const size_t sizeA = m * k * sizeof(__half);
    const size_t sizeB = k * n * sizeof(__half);
    const size_t sizeC = m * n * sizeof(float);

    auto *h_A = static_cast<__half*>(malloc(sizeA));
    auto *h_B = static_cast<__half*>(malloc(sizeB));
    auto *h_C = static_cast<float*>(malloc(sizeC));

    if (loadHalfMatricesFromFileArray(filePath, h_A, m * k, h_B, k * n) != 0) {
        std::cerr << "Failed to load matrices from file.\n";
        return 2;
    }

    __half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(2, 2);
    dim3 blocksPerGrid(n / WMMA_N / threadsPerBlock.x, m / WMMA_M / threadsPerBlock.y);

    matrixMultiplyWMMA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "Result matrix C (" << m << "x" << n << "):\n";
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << h_C[i * n + j] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
