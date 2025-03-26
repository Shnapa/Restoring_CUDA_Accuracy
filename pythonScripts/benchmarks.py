import subprocess
import csv

configs = [
    (100, 50, 100),
    (200, 100, 150),
    (500, 250, 300),
    (1000, 500, 800),
    (2000, 1000, 1500),
    (5000, 2500, 3000),
    (10000, 5000, 8000)
]

num_runs = 1

executables = [
    "cublasMul",
    "cudaMul",
    "cudaMulOpt",
    "simdMul",
    "simdMulOpt",
]

def run_benchmark(executable, matrix_file_A, matrix_file_B):
    times = []
    for _ in range(num_runs):
        result = subprocess.run(
            [f"build/{executable}", matrix_file_A, matrix_file_B],
            capture_output=True, text=True
        )
        output = result.stdout.strip()
        try:
            elapsed_time = float(output.split()[-2])
            times.append(elapsed_time)
        except ValueError:
            print(f"Error parsing time for {executable} with matrices {matrix_file_A} and {matrix_file_B}: {output}")

    avg_time = float(sum(times) / len(times))
    return avg_time

def run_benchmarks():
    for executable in executables:
        output_file = f"benchmark_results_{executable}.csv"
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Executable", "m", "n", "k", "average_time_ms"])

            for m, n, k in configs:
                matrix_A_filename = f"pythonScripts/matrix_A_{m}x{n}.txt"
                matrix_B_filename = f"pythonScripts/matrix_B_{n}x{k}.txt"
                print(f"Running benchmark for {executable} with matrices {matrix_A_filename} and {matrix_B_filename}")
                avg_time = run_benchmark(executable, matrix_A_filename, matrix_B_filename)
                writer.writerow([executable, m, n, k, avg_time])
                print(f"Benchmark completed for {executable} with matrices {m}x{n} and {n}x{k} with average time: {avg_time:.4f} ms")

        print(f"Benchmark results saved to {output_file}")


run_benchmarks()
