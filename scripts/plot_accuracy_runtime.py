#!/usr/bin/env python3
"""Plot accuracy-runtime Pareto curves from the Phase-1 benchmark CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    needed = {"family", "median_ms", "max_norm_error", "configuration"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing required CSV columns: {needed - set(df.columns)}")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, subset in df.groupby("family", sort=False):
        sub = subset.sort_values("median_ms")
        err = np.maximum(sub["max_norm_error"].astype(float).to_numpy(), 1e-17)
        time_ms = sub["median_ms"].astype(float).to_numpy()
        ax.plot(time_ms, err, "o-", label=method)
        for t, e, label in zip(time_ms, err, sub["configuration"]):
            suffix = label.split("_")[-1]
            ax.annotate(suffix, (t, e), fontsize=7, xytext=(3, 4), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("Median online runtime (ms)")
    ax.set_ylabel("Maximum normalized proximal error (measured)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
