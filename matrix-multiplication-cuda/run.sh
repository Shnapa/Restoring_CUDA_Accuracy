#!/bin/bash

# Set the source and output file names
SRC_FILE="matrix_multiplication_cuda.cu"
OUTPUT_FILE="matrix_mul"

# Check if NVCC is installed
if ! command -v nvcc &> /dev/null
then
    echo "Error: nvcc (NVIDIA CUDA Compiler) not found. Make sure CUDA is installed and nvcc is in your PATH."
    exit 1
fi

# Compile the CUDA program
echo "Compiling $SRC_FILE..."
nvcc -O2 -arch=sm_50 -o $OUTPUT_FILE $SRC_FILE

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"

# Run the program
echo "Running the CUDA matrix multiplication..."
./$OUTPUT_FILE
