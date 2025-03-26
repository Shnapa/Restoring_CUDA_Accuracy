import numpy as np

matrix_sizes = [
    (100, 50, 100),
    (200, 100, 150),
    (500, 250, 300),
    (1000, 500, 800),
    (2000, 1000, 1500),
    (5000, 2500, 3000),
    (10000, 5000, 8000)
]

def generate_matrix(m, n):
    return np.random.uniform(1.0, 10000.0, (m, n)).astype(np.float32)

def save_matrix_to_file(matrix, filename):
    np.savetxt(filename, matrix, fmt='%0.6f')

def generate_and_save_matrices():
    for m, n, k in matrix_sizes:
        matrix_A = generate_matrix(m, n)
        matrix_B = generate_matrix(n, k)

        save_matrix_to_file(matrix_A, f"matrix_A_{m}x{n}.txt")
        save_matrix_to_file(matrix_B, f"matrix_B_{n}x{k}.txt")

        print(f"Matrix A: {m}x{n} saved to 'matrix_A_{m}x{n}.txt'")
        print(f"Matrix B: {n}x{k} saved to 'matrix_B_{n}x{k}.txt")

generate_and_save_matrices()