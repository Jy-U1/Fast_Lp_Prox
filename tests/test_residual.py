import torch

from fast_lp_prox import (
    active_mask,
    lp_constants,
    make_lp_prox,
    objective_gap_vs_reference,
    objective_value,
    prox_stationary_residual,
    residual_error_bound,
)


def test_objective_value_shape_and_finite():
    y = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float32)
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)

    obj = objective_value(y, x, p=0.5, lam=1.0)

    assert obj.shape == y.shape
    assert torch.isfinite(obj).all()


def test_active_mask_threshold():
    p = 0.5
    lam = 1.0
    const = lp_constants(p, lam)

    y = torch.tensor(
        [
            -2.0 * const.tau,
            -0.5 * const.tau,
            0.0,
            0.5 * const.tau,
            2.0 * const.tau,
        ],
        dtype=torch.float32,
    )

    mask = active_mask(y, p=p, lam=lam)

    expected = torch.tensor([True, False, False, False, True])

    assert torch.equal(mask.cpu(), expected)


def test_newton_residual_is_small_on_active_entries():
    p = 0.5
    lam = 1.0

    prox = make_lp_prox(p=p, lam=lam, method="newton12")

    y = torch.linspace(-8.0, 8.0, 2001, dtype=torch.float32)
    x = prox(y)

    mask = active_mask(y, p=p, lam=lam)
    residual = prox_stationary_residual(y, x, p=p, lam=lam)

    max_residual = torch.max(torch.abs(residual[mask])).item()

    assert max_residual < 1e-5


def test_residual_bound_is_nonnegative_and_finite():
    p = 0.5
    lam = 1.0

    prox = make_lp_prox(p=p, lam=lam, method="newton12")

    y = torch.linspace(-8.0, 8.0, 2001, dtype=torch.float32)
    x = prox(y)

    bound = residual_error_bound(y, x, p=p, lam=lam)

    assert bound.shape == y.shape
    assert torch.isfinite(bound).all()
    assert torch.all(bound >= 0)


def test_poly_objective_close_to_reference():
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
        use_table=False,
    )

    prox_ref = make_lp_prox(
        p=p,
        lam=lam,
        method="newton12",
    )

    y = torch.linspace(-8.0, 8.0, 2001, dtype=torch.float32)

    x_poly = prox_poly(y)
    x_ref = prox_ref(y)

    gap = objective_gap_vs_reference(y, x_poly, x_ref, p=p, lam=lam)

    assert torch.isfinite(gap).all()
    assert torch.max(torch.abs(gap)).item() < 1e-3