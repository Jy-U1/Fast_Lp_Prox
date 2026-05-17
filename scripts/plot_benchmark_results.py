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


def save_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    logy: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.bar(df[x_col].astype(str), df[y_col])

    if logy:
        plt.yscale("log")

    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def plot_benchmark(input_csv: str, output_dir: str) -> None:
    input_path = Path(input_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = order_methods(df)

    if "median_ms" not in df.columns:
        raise ValueError("Input CSV must contain column: median_ms")

    if "million_elems_per_sec" not in df.columns:
        raise ValueError("Input CSV must contain column: million_elems_per_sec")

    stem = input_path.stem

    save_bar_chart(
        df=df,
        x_col="method",
        y_col="median_ms",
        title="Prox runtime by method",
        ylabel="Median runtime (ms)",
        output_path=output_path / f"{stem}_runtime_ms.png",
    )

    save_bar_chart(
        df=df,
        x_col="method",
        y_col="million_elems_per_sec",
        title="Prox throughput by method",
        ylabel="Million elements / second",
        output_path=output_path / f"{stem}_throughput.png",
    )

    if "poly" in set(df["method"].astype(str)):
        poly_ms = float(df.loc[df["method"].astype(str) == "poly", "median_ms"].iloc[0])

        speedup_df = df.copy()
        speedup_df["speedup_vs_poly"] = speedup_df["median_ms"] / poly_ms

        save_bar_chart(
            df=speedup_df,
            x_col="method",
            y_col="speedup_vs_poly",
            title="Runtime ratio relative to poly",
            ylabel="median_ms(method) / median_ms(poly)",
            output_path=output_path / f"{stem}_speedup_vs_poly.png",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from scripts/benchmark_prox.py CSV output."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input benchmark CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save figures.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    plot_benchmark(
        input_csv=args.input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()