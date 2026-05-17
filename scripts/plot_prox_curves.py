from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from fast_lp_prox import lp_constants, make_lp_prox


def parse_p(value: str) -> float:
    if "/" in value:
        num, den = value.split("/")
        return float(num) / float(den)
    return float(value)


def main():
    parser = argparse.ArgumentParser(description="Plot Lp prox curves.")
    parser.add_argument("--p", type=str, default="1/3")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--ymax", type=float, default=6.0)
    parser.add_argument("--num-points", type=int, default=2001)
    parser.add_argument("--output", type=str, default="figures/prox_curve_p_1_3.png")
    args = parser.parse_args()

    p = parse_p(args.p)
    lam = args.lam

    y = torch.linspace(-args.ymax, args.ymax, args.num_points)

    methods = ["poly", "newton8", "newton12", "bisect40"]
    outputs = {}

    for method in methods:
        prox = make_lp_prox(
            p=p,
            lam=lam,
            method=method,
        )
        outputs[method] = prox(y)

    const = lp_constants(p, lam)

    plt.figure(figsize=(7, 5))

    for method in methods:
        plt.plot(
            y.numpy(),
            outputs[method].numpy(),
            label=method,
            linewidth=1.5,
        )

    plt.axvline(const.tau, linestyle="--", linewidth=1.0, label="threshold")
    plt.axvline(-const.tau, linestyle="--", linewidth=1.0)
    plt.axhline(0.0, linewidth=0.8)

    plt.xlabel("y")
    plt.ylabel("prox(y)")
    plt.title(f"Lp prox curve, p={args.p}, lambda={lam}")
    plt.legend()
    plt.grid(True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()