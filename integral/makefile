# Compiler
NVCC = nvcc

# Target executable name
TARGET = integral

# CUDA architecture (modify based on your GPU)
ARCH = -arch=sm_60  # Change if needed

# Source files
SRC = integral.cu

# Default rule: Compile CUDA code
all: $(TARGET)

# Compilation rule
$(TARGET): $(SRC)
	$(NVCC) $(ARCH) -o $(TARGET) $(SRC)

# Run the executable
run: $(TARGET)
	./$(TARGET)

# Clean build files
clean:
	rm -f $(TARGET)
