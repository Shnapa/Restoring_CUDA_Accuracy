import os
import glob
import subprocess
import csv

def parse_filename(filename):
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    parts = name.split('_')
    if len(parts) < 2:
        raise ValueError("Filename format incorrect")
    matrix_type = parts[0]
    dims = parts[1]
    try:
        d1, d2 = dims.split('x')
        d1 = int(d1)
        d2 = int(d2)
    except Exception as e:
        raise ValueError(f"Error parsing dimensions in {filename}: {e}")
    return matrix_type, d1, d2

def main():
    dataset_dir = "data/matrix_dataset"
    executable = "cmake-build-debug/cudaMulOpt"
    csv_filename = "benchmark_results_opt.csv"
    results = []

    a_files = glob.glob(os.path.join(dataset_dir, "A_*.txt"))
    for a_file in a_files:
        try:
            matrix_type, n, k = parse_filename(a_file)
        except Exception as e:
            print(f"Skipping {a_file}: {e}")
            continue

        b_pattern = os.path.join(dataset_dir, f"B_{k}x*.txt")
        b_files = glob.glob(b_pattern)
        if not b_files:
            print(f"No matching B file for {a_file}")
            continue
        b_file = b_files[0]
        try:
            matrix_type_b, k_b, m = parse_filename(b_file)
            if k_b != k:
                print(f"Dimension mismatch for {a_file} and {b_file}")
                continue
        except Exception as e:
            print(f"Error parsing {b_file}: {e}")
            continue

        cmd = [executable, a_file, b_file]
        print("\nRunning:", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = proc.stdout.strip()
        print("Output:", output)
        match = output.split()[2]
        if match:
            time_sec = float(match)
        else:
            print(f"Failed to parse time from output: {output}")
            continue
        a_size = f"{n}x{k}"
        b_size = f"{k}x{m}"
        results.append({
            "A_matrix": a_size,
            "B_matrix": b_size,
            "n": n,
            "k": k,
            "m": m,
            "time_ms": f"{time_sec:.6f}"
        })

    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = ["A_matrix", "B_matrix", "n", "k", "m", "time_ms"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    main()
