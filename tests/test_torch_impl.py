import torch

from fast_lp_prox import make_lp_prox


def test_poly_matches_newton12_reasonably_on_cpu():
    p = 0.5
    lam = 1.0

    prox_poly = make_lp_prox(
        p=p,
        lam=lam,
        method="poly",
        target_error=1e-4,
        max_degree=24,
        n_fit=512,
        n_val=1001,
    )

    prox_newton = make_lp_prox(
        p=p,
        lam=lam,
        method="newton12",
    )

    y = torch.linspace(-8.0, 8.0, 2001, dtype=torch.float32)

    x_poly = prox_poly(y)
    x_newton = prox_newton(y)

    max_err = torch.max(torch.abs(x_poly - x_newton)).item()

    assert max_err <= 2e-3


def test_lp_prox_one_off_call():
    from fast_lp_prox import lp_prox

    y = torch.tensor([-3.0, 0.0, 3.0], dtype=torch.float32)

    x = lp_prox(
        y,
        p=0.5,
        lam=1.0,
        method="poly",
        target_error=1e-4,
        max_degree=24,
        n_fit=512,
        n_val=1001,
    )

    assert x.shape == y.shape
    assert torch.isfinite(x).all()