#!/bin/bash

# Set CUDA architecture (modify as needed)
ARCH=sm_60  # Change based on your GPU (use `nvcc --help` to check architectures)

# CUDA file name
CUDA_FILE="helloGPU.cu"
EXECUTABLE="helloGPU"

# Check if NVCC is installed
if ! command -v nvcc &> /dev/null
then
    echo "Error: NVCC (NVIDIA CUDA Compiler) is not installed or not in PATH."
    exit 1
fi

# Compile CUDA code
echo "Compiling $CUDA_FILE..."
nvcc -arch=$ARCH -o $EXECUTABLE $CUDA_FILE

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Run the CUDA program
echo "Running $EXECUTABLE..."
./$EXECUTABLE
