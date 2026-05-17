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
    parser = argparse.ArgumentParser(description="Plot approximation error of polynomial Lp prox.")
    parser.add_argument("--p", type=str, default="1/3")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--ymax", type=float, default=6.0)
    parser.add_argument("--num-points", type=int, default=5001)
    parser.add_argument("--output", type=str, default="figures/prox_error_p_1_3.png")
    args = parser.parse_args()

    p = parse_p(args.p)
    lam = args.lam

    y = torch.linspace(-args.ymax, args.ymax, args.num_points)

    prox_poly = make_lp_prox(
        p=p,
        lam=lam,
        method="poly",
    )

    prox_ref = make_lp_prox(
        p=p,
        lam=lam,
        method="newton12",
    )

    x_poly = prox_poly(y)
    x_ref = prox_ref(y)

    abs_err = torch.abs(x_poly - x_ref)

    const = lp_constants(p, lam)

    plt.figure(figsize=(7, 5))

    plt.semilogy(y.numpy(), abs_err.numpy(), linewidth=1.5, label="|poly - newton12|")

    plt.axvline(const.tau, linestyle="--", linewidth=1.0, label="threshold")
    plt.axvline(-const.tau, linestyle="--", linewidth=1.0)

    plt.xlabel("y")
    plt.ylabel("absolute error")
    plt.title(f"Polynomial prox error, p={args.p}, lambda={lam}")
    plt.legend()
    plt.grid(True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    print(f"max abs error : {abs_err.max().item():.3e}")
    print(f"mean abs error: {abs_err.mean().item():.3e}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()