#include "testing.h"
#include <cstdlib>
#include <ctime>
#include <map>


int main(int argc, char * argv[]) {
    std::string functionName = argv[1];

    //sample matrix generation i prikoly
    std::srand(std::time(0));
    int rowsA = std::atoi(argv[2]);
    int colsA = std::atoi(argv[3]);
    int rowsB = std::atoi(argv[4]);
    int colsB = std::atoi(argv[5]);
    std::vector<std::vector<int>> matrixA =  generateMatrix(rowsA, colsA);
    printMatrix(matrixA);
    std::vector<std::vector<int>> matrixB =  generateMatrix(rowsB, colsB);
    printMatrix(matrixB);

    //multiplication of A & B
    //standard (CPU bruteforce)
    std::vector<std::vector<int>> standard = multiplyMatrices(matrixA, matrixB);
    printMatrix(standard);

    //mult method that we want to check
    std::vector<std::vector<int>> result;
    std::map<std::string, std::vector<std::vector<int>>(*)(std::vector<std::vector<int>>, std::vector<std::vector<int>>)> functionMap;
    // Insert functions into the map
    functionMap["cpu_simd"] = &foo;
    functionMap["cpu_parallel"] = &bar;
    functionMap["cuda"] = &foo;
    functionMap["cuda_opt"] = &bar;
    functionMap["cublass"] = &foo;

    if (functionMap.find(functionName) != functionMap.end()) {
        // Call the function with matrixA and matrixB as arguments
        result = functionMap[functionName](matrixA, matrixB);
    } else {
        std::cout << "Function " << functionName << " not found!" << std::endl;
    }
    // compare matrices
    if (compareMatrices(standard, result)) {
        std::cout << "Matrices are identical!" << std::endl;
    } else {
        std::cout << "Matrices are different!" << std::endl;
    }

    return 0;
}