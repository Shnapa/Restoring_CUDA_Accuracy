import subprocess
import matplotlib.pyplot as plt
import numpy as np

methods = {
    "--simd":      ("SIMD", 'blue', 'o'),
    "--cuda":      ("Cuda", 'red', 'v'),
    "--simd-opt":  ("SIMD Opt", 'orange', 'x'),
    "--wmma":      ("WMMA' method", 'green', '^'),
    "--cuda-opt":  ("Cuda Opt", 'purple', '+')
}

baseline_flag = "--naive"

k_values = [2**i for i in range(6, 19)]
m, n = 16, 16

def generate_matrix_file(filename, rows, cols):
    matrix = np.random.rand(rows, cols).astype(np.float32)
    np.savetxt(filename, matrix, fmt='%.6f')

errors_by_method = {flag: [] for flag in methods}

for k in k_values:
    print(f"Running tests for k = {k}")
    generate_matrix_file("matrixA.txt", m, k)
    generate_matrix_file("matrixB.txt", k, n)

    result_naive = subprocess.run(
        ["../cmake-build-debug/matmul_compare", baseline_flag, "matrixA.txt", "matrixB.txt"],
        capture_output=True, text=True
    )
    if result_naive.returncode != 0 or "error =" not in result_naive.stdout:
        print(f"Naive failed at k={k}, skipping")
        for flag in methods:
            errors_by_method[flag].append(None)
        continue

    for flag in methods:
        try:
            result = subprocess.run(
                ["../cmake-build-debug/matmul_compare", flag, "matrixA.txt", "matrixB.txt"],
                capture_output=True, text=True
            )
            found = False
            for line in result.stdout.splitlines():
                if "error =" in line:
                    error_val = float(line.strip().split('=')[-1])
                    errors_by_method[flag].append(error_val)
                    found = True
                    break
            if not found:
                errors_by_method[flag].append(None)
                print(f"[{flag}] No error output at k={k}")
        except subprocess.CalledProcessError as e:
            print(f"[{flag}] Failed at k={k}: {e.stderr}")
            errors_by_method[flag].append(None)

plt.figure(figsize=(10, 6))
for flag, (label, color, marker) in methods.items():
    y_vals = errors_by_method[flag]
    if all(v is None for v in y_vals):
        continue
    y_vals_clean = [y if y is not None else np.nan for y in y_vals]
    plt.plot(k_values, y_vals_clean, label=label, marker=marker, color=color)

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel(r"$k$ : matmul-(16, 16, $k$)")
plt.ylabel("Relative residual")
plt.title("Relative Residual vs Matrix Inner Dimension (k)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
