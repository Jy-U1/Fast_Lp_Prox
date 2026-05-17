import torch

from fast_lp_prox import lp_constants, make_lp_prox


def test_threshold_returns_zero_newton8():
    p = 0.5
    lam = 1.0
    const = lp_constants(p, lam)

    prox = make_lp_prox(p=p, lam=lam, method="newton8")

    y = torch.tensor(
        [
            -const.tau,
            -0.5 * const.tau,
            0.0,
            0.5 * const.tau,
            const.tau,
        ],
        dtype=torch.float32,
    )

    x = prox(y)

    assert torch.allclose(x, torch.zeros_like(y))


def test_threshold_returns_zero_poly():
    p = 0.5
    lam = 1.0
    const = lp_constants(p, lam)

    prox = make_lp_prox(
        p=p,
        lam=lam,
        method="poly",
        target_error=1e-4,
        max_degree=24,
        n_fit=512,
        n_val=1001,
    )

    y = torch.tensor(
        [
            -const.tau,
            -0.5 * const.tau,
            0.0,
            0.5 * const.tau,
            const.tau,
        ],
        dtype=torch.float32,
    )

    x = prox(y)

    assert torch.allclose(x, torch.zeros_like(y))