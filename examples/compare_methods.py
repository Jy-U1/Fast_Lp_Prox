import torch

from fast_lp_prox import make_lp_prox


def main():
    p = 1 / 3
    lam = 1.0

    y = torch.linspace(-8.0, 8.0, 1001)

    methods = ["poly", "newton8", "newton12", "bisect40"]

    outputs = {}

    for method in methods:
        prox = make_lp_prox(
            p=p,
            lam=lam,
            method=method,
        )
        outputs[method] = prox(y)

    ref = outputs["newton12"]

    print(f"Comparison for p={p}, lambda={lam}")
    print("-" * 80)

    for method in methods:
        err = torch.max(torch.abs(outputs[method] - ref)).item()
        print(f"{method:10s} max |x - newton12| = {err:.3e}")


if __name__ == "__main__":
    main()