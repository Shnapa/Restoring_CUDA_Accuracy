import subprocess
import matplotlib.pyplot as plt
import numpy as np

def run_cpp_binary(flag, matrixA, matrixB):
    try:
        result = subprocess.run(
            ["./compare_accuracy", flag, matrixA, matrixB],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        for line in result.stdout.splitlines():
            if line.startswith("RESIDUAL="):
                return float(line.strip().split("=")[1])
    except subprocess.CalledProcessError as e:
        print(f"❌ Error for {flag}: {e.stderr}")
        return None

def main():
    matrixA = "matrixA.txt"
    matrixB = "matrixB.txt"

    implementations = {
        "--naive": "Naive",
        "--simd": "SIMD",
        "--cuda": "CUDA",
        "--cuda-opt": "CUDA Optimized",
        "--cublas": "cuBLAS",
        "--wmma": "WMMA",
        "--restore_wmma": "Restored WMMA"
    }

    residuals = {}
    for flag, name in implementations.items():
        residual = run_cpp_binary(flag, matrixA, matrixB)
        if residual is not None:
            residuals[name] = residual
            print(f"{name}: Residual = {residual:.3e}")
        else:
            print(f"⚠️ {name} skipped due to error.")

    # Visualization
    plt.figure(figsize=(8, 5))
    labels = list(residuals.keys())
    values = list(residuals.values())

    plt.bar(labels, values)
    plt.ylabel("Relative Residual")
    plt.xticks(rotation=30)
    plt.title("Accuracy of Different Matrix Multiplication Methods")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("residual_comparison.png")
    plt.show()
    print("✅ Plot saved as residual_comparison.png")

if __name__ == "__main__":
    main()
