from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["poly", "newton8", "newton12", "bisect40"]


def order_methods(df: pd.DataFrame) -> pd.DataFrame:
    existing = [m for m in METHOD_ORDER if m in set(df["method"])]
    extra = [m for m in df["method"].tolist() if m not in existing]
    order = existing + extra

    df = df.copy()
    df["method"] = pd.Categorical(df["method"], categories=order, ordered=True)
    return df.sort_values("method")


def save_bar(df, y_col: str, title: str, ylabel: str, output_path: Path, logy: bool = False):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.bar(df["method"].astype(str), df[y_col])

    if logy:
        plt.yscale("log")

    plt.xlabel("method")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y")
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Lp-Lasso Forward-Backward benchmark results."
    )

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="figures")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = order_methods(df)

    stem = input_path.stem

    save_bar(
        df,
        y_col="median_ms",
        title="Lp-Lasso Forward-Backward runtime",
        ylabel="Median runtime (ms)",
        output_path=output_dir / f"{stem}_runtime_ms.png",
    )

    save_bar(
        df,
        y_col="ms_per_iter",
        title="Lp-Lasso Forward-Backward runtime per iteration",
        ylabel="Median ms / iteration",
        output_path=output_dir / f"{stem}_ms_per_iter.png",
    )

    save_bar(
        df,
        y_col="final_objective",
        title="Final objective",
        ylabel="objective value",
        output_path=output_dir / f"{stem}_final_objective.png",
    )

    if "poly" in set(df["method"].astype(str)):
        poly_ms = float(df.loc[df["method"].astype(str) == "poly", "median_ms"].iloc[0])
        speedup_df = df.copy()
        speedup_df["runtime_ratio_vs_poly"] = speedup_df["median_ms"] / poly_ms

        save_bar(
            speedup_df,
            y_col="runtime_ratio_vs_poly",
            title="Runtime ratio relative to poly",
            ylabel="median_ms(method) / median_ms(poly)",
            output_path=output_dir / f"{stem}_runtime_ratio_vs_poly.png",
        )


if __name__ == "__main__":
    main()