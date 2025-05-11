import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os

def run_cpp_binary(flag, matrix_file):
    try:
        result = subprocess.run(
            ["../build/utils/matmul_compare", flag, matrix_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        for line in result.stdout.splitlines():
            if line.startswith("RESIDUAL="):
                return float(line.strip().split("=")[1])
    except subprocess.CalledProcessError as e:
        print(f"❌ Error for {flag}: {e.stderr.strip()}")
        return None


def main():
    # Change this if needed
    m, n, k = 16, 16, 16
    matrix_path = f"../data/matrix_{m}_{n}_{k}.txt"
    if not os.path.exists(matrix_path):
        print(f"❌ Matrix file not found: {matrix_path}")
        return

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
        print(f"🔁 Running {name}")
        residual = run_cpp_binary(flag, matrix_path)
        if residual is not None:
            residuals[name] = residual
            print(f"✅ {name}: Residual = {residual:.3e}")
        else:
            print(f"⚠️  {name} skipped due to error.")

    # Visualization
    if not residuals:
        print("❌ No results to plot.")
        return

    plt.figure(figsize=(9, 5))
    labels = list(residuals.keys())
    values = list(residuals.values())

    plt.bar(labels, values, color='mediumslateblue')
    plt.ylabel("Relative Residual (log scale)")
    plt.yscale("log")
    plt.xticks(rotation=30)
    plt.title(f"Accuracy Comparison: Matrix {m}×{n}×{k}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    filename = f"residual_comparison_{m}_{n}_{k}.png"
    plt.savefig(filename)
    plt.show()
    print(f"📊 Plot saved as {filename}")

if __name__ == "__main__":
    main()
