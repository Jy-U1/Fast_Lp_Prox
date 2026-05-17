import torch

from fast_lp_prox import lp_constants, make_lp_prox


def main():
    p = 1 / 3
    lam = 1.0

    const = lp_constants(p, lam)
    print("Lp constants:")
    print(const)

    y = torch.linspace(-5.0, 5.0, 21)

    prox = make_lp_prox(
        p=p,
        lam=lam,
        method="poly",
    )

    x = prox(y)

    print("\ny:")
    print(y)

    print("\nprox(y):")
    print(x)


if __name__ == "__main__":
    main()