from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["poly", "newton8", "newton12", "bisect40"]


def order_methods(df: pd.DataFrame) -> pd.DataFrame:
    if "method" not in df.columns:
        raise ValueError("Input CSV must contain column: method")

    existing = [m for m in METHOD_ORDER if m in set(df["method"])]
    extra = [m for m in df["method"].tolist() if m not in existing]
    order = existing + extra

    df = df.copy()
    df["method"] = pd.Categorical(df["method"], categories=order, ordered=True)
    return df.sort_values("method")


def save_bar_chart(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    logy: bool = True,
) -> None:
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


def plot_columns(
    input_csv: str,
    output_dir: str,
    columns: list[str],
    title_prefix: str,
) -> None:
    input_path = Path(input_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = order_methods(df)

    stem = input_path.stem

    for col in columns:
        if col not in df.columns:
            print(f"Skipping missing column: {col}")
            continue

        pretty = col.replace("_", " ")

        save_bar_chart(
            df=df,
            y_col=col,
            title=f"{title_prefix}: {pretty}",
            ylabel=pretty,
            output_path=output_path / f"{stem}_{col}.png",
            logy=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot accuracy or residual validation CSV results."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input validation CSV.",
    )

    parser.add_argument(
        "--kind",
        type=str,
        required=True,
        choices=["accuracy", "residual"],
        help="Type of validation CSV.",
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

    if args.kind == "accuracy":
        columns = [
            "max_abs_err",
            "mean_abs_err",
            "max_rel_err",
            "mean_rel_err",
        ]
        title_prefix = "Accuracy validation"

    elif args.kind == "residual":
        columns = [
            "max_abs_residual",
            "mean_abs_residual",
            "max_residual_bound",
            "mean_residual_bound",
            "max_abs_obj_gap_vs_ref",
            "mean_abs_obj_gap_vs_ref",
        ]
        title_prefix = "Residual validation"

    else:
        raise ValueError(f"Unknown kind: {args.kind}")

    plot_columns(
        input_csv=args.input,
        output_dir=args.output_dir,
        columns=columns,
        title_prefix=title_prefix,
    )


if __name__ == "__main__":
    main()