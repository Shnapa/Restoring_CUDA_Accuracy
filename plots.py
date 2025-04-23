#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    csv_files = [
        "results_cublasMul.csv",
        "results_cudaMul.csv",
        "results_cudaMulOpt.csv",
        "results_simdMul.csv",
        "results_simdMulOpt.csv"
    ]

    executables_list = ["cublasMul", "cudaMul", "cudaMulOpt", "simdMul", "simdMulOpt"]

    cmap = plt.cm.get_cmap("rainbow", len(executables_list))

    exe_to_color = {exe: cmap(i) for i, exe in enumerate(executables_list)}

    dfs = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df_temp = pd.read_csv(csv_file)
            dfs.append(df_temp)
        else:
            print(f"Warning: {csv_file} not found. Skipping.")

    if not dfs:
        print("No CSV files found; exiting.")
        return

    data = pd.concat(dfs, ignore_index=True)

    required_columns = {"Executable", "m", "n", "k", "average_time_ms"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Data is missing one or more required columns: {required_columns}")

    grouped = data.groupby(["m", "n", "k"])

    for (m_val, n_val, k_val), group_df in grouped:
        group_df = group_df.set_index("Executable").reindex(executables_list).dropna(subset=["m", "n", "k", "average_time_ms"])

        executables = group_df.index.values
        times = group_df["average_time_ms"].values

        colors = [exe_to_color[exe] for exe in executables]

        plt.figure(figsize=(8, 5))
        plt.bar(executables, times, color=colors)

        plt.title(f"Matrix A: {m_val}×{n_val}, Matrix B: {n_val}×{k_val} (Result: {m_val}×{k_val})")
        plt.xlabel("Executable")
        plt.ylabel("Average Time (ms)")

        for idx, val in enumerate(times):
            plt.text(idx, val + 0.01 * val, f"{val:.2f}",
                     ha="center", va="bottom", fontsize=8)

        plt.tight_layout()

        out_filename = f"matrixA_{m_val}x{n_val}_matrixB_{n_val}x{k_val}.png"
        plt.savefig(out_filename, dpi=150)
        plt.close()

        print(f"Saved plot: {out_filename}")

if __name__ == "__main__":
    main()
