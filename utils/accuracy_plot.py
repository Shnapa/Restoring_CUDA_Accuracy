import subprocess
import matplotlib.pyplot as plt
import numpy as np

methods = {
    "--simd":      ("FP32 SIMT", 'blue', 'o'),
    "--cuda":      ("TensorCore w/o ErrCor", 'red', 'v'),
    "--simd-opt":  ("Our method", 'orange', 'x'),
    "--wmma":      ("Markidis' method", 'green', '^'),
    "--cuda-opt":  ("Feng's method", 'purple', '+')
}

baseline_flag = "--naive"

k_values = [2**i for i in range(6, 19)]
m, n = 16, 16

def generate_matrix_file(filename, rows, cols):
    matrix = np.random.rand(rows, cols).astype(np.float32)
    np.savetxt(filename, matrix, fmt='%.6f')

errors_by_method = {key: [] for key in methods.keys()}

for k in k_values:
    generate_matrix_file("matrixA.txt", m, k)
    generate_matrix_file("matrixB.txt", k, n)

    result_naive = subprocess.run(
        ["../cmake-build-debug/matmul_compare", baseline_flag, "matrixA.txt", "matrixB.txt"],
        check=True, capture_output=True, text=True
    )

    # Check correctness of baseline
    if "error =" not in result_naive.stdout:
        raise RuntimeError("Baseline (naive) failed at k={}".format(k))

    for flag in methods:
        try:
            result = subprocess.run(
                ["../cmake-build-debug/matmul_compare", flag, "matrixA.txt", "matrixB.txt"],
                check=True,
                capture_output=True,
                text=True
            )
            for line in result.stdout.splitlines():
                if "error =" in line:
                    error_value = float(line.strip().split('=')[-1])
                    errors_by_method[flag].append(error_value)
                    break
        except subprocess.CalledProcessError as e:
            print(f"[{flag}] Failed at k={k}: {e.stderr}")
            errors_by_method[flag].append(None)

# Plot
plt.figure(figsize=(10, 6))
for flag, (label, color, marker) in methods.items():
    y_vals = errors_by_method[flag]
    if all(v is None for v in y_vals):
        continue
    plt.plot(k_values, y_vals, label=label, marker=marker, color=color)

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel(r"$k$ : matmul-(16, 16, $k$)")
plt.ylabel("Relative residual")
plt.title("Relative Residual vs Matrix Inner Dimension (k)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
