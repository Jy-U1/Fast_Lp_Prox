from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from fast_lp_prox import lp_constants, make_lp_prox


def main():
    lam = 1.0
    ymax = 6.0
    y = torch.linspace(-ymax, ymax, 3001)

    p_cases = [
        ("1/3", 1 / 3),
        ("2/5", 2 / 5),
        ("3/4", 3 / 4),
    ]

    plt.figure(figsize=(7, 5))

    for label, p in p_cases:
        prox = make_lp_prox(
            p=p,
            lam=lam,
            method="poly",
        )
        x = prox(y)
        const = lp_constants(p, lam)

        plt.plot(
            y.numpy(),
            x.numpy(),
            linewidth=1.5,
            label=f"p={label}, tau={const.tau:.3f}",
        )

    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("y")
    plt.ylabel("prox(y)")
    plt.title("Lp prox curves for different p")
    plt.legend()
    plt.grid(True)

    output_path = Path("figures/prox_curves_p_sweep.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()