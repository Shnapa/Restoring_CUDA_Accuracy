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
        print(f"❌ Error for {flag} on {matrix_file}: {e.stderr.strip()}")
        return None

def main():
    m = n = 16
    ks = [2**i for i in range(6, 19)]  # 2^6 to 2^18

    # your implementations
    implementations = {
        "--naive": "Naive",
        "--simd": "SIMD",
        "--cuda": "CUDA",
        "--cuda-opt": "CUDA Optimized",
        "--cublas": "cuBLAS",
        "--wmma": "WMMA",
        "--restore_wmma": "Restored WMMA"
    }

    results = {name: [] for name in implementations.values()}

    for k in ks:
        matrix_path = f"../data/matrix_{m}_{n}_{k}.txt"
        if not os.path.exists(matrix_path):
            print(f"⚠️ Missing file: {matrix_path}")
            for res in results.values():
                res.append(None)
            continue

        print(f"\n📁 Matrix {m}×{n}×{k}")
        for flag, name in implementations.items():
            print(f"🔁 Running {name}...")
            residual = run_cpp_binary(flag, matrix_path)
            if residual is not None:
                results[name].append(residual)
                print(f"✅ {name}: Residual = {residual:.2e}")
            else:
                results[name].append(None)

    # Plotting
    plt.figure(figsize=(10, 6))
    for name, residuals in results.items():
        valid = [(k, r) for k, r in zip(ks, residuals) if r is not None]
        if valid:
            k_vals, res_vals = zip(*valid)
            plt.plot(k_vals, res_vals, marker="o", label=name)

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel(r"$k$ in $(16, 16, k)$", fontsize=12)
    plt.ylabel("Relative Residual", fontsize=12)
    plt.title("Relative Residual vs. k", fontsize=13)
    plt.grid(True, which='both', linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("residual_vs_k.png")
    plt.show()
    print("📊 Plot saved as residual_vs_k.png")

if __name__ == "__main__":
    main()
