import sys
import pandas as pd
import matplotlib.pyplot as plt

def main(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    df1["work"] = df1["n"] * df1["k"] * df1["m"]
    df2["work"] = df2["n"] * df2["k"] * df2["m"]

    df1["source"] = "File1"
    df2["source"] = "File2"

    merged = pd.concat([df1, df2], ignore_index=True)

    merged["key"] = merged["n"].astype(str) + "-" + merged["k"].astype(str) + "-" + merged["m"].astype(str)

    pivot_df = merged.pivot(index="key", columns="source", values="time_ms")

    key_to_work = merged.groupby("key")["work"].first()

    pivot_df = pivot_df.reindex(key_to_work.sort_values().index)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

    pivot_df.plot(
        kind="bar",
        ax=ax,
        logy=True,
        width=0.8,
        edgecolor="black"
    )

    x_labels = []
    for row_key in pivot_df.index:
        w = key_to_work[row_key]
        x_labels.append(f"{w:.1e}")

    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    ax.set_xlabel("Work (n * k * m)")
    ax.set_ylabel("Time (ms) [log scale]")
    ax.set_title("Comparison of Two CSV Benchmarks")

    ax.grid(True, which="both", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_name = "comparison_plot.png"
    plt.savefig(plot_name)
    plt.show()
    print(f"Plot saved as {plot_name}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <file1.csv> <file2.csv>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    main(file1, file2)
