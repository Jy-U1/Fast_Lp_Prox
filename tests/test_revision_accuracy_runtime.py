"""Fast CPU checks of the additive revision benchmarking code."""
import numpy as np
import pytest
import torch
from numpy.polynomial.chebyshev import chebfit

from fast_lp_prox import lp_constants, make_lp_prox
from fast_lp_prox.benchmark_utils import make_test_vector
from fast_lp_prox.revision_bench import (
    fit_degree_family,
    make_preloaded_prox,
    measure_accuracy,
    normalized_branch_bisect,
    parse_positive_int_grid,
    prox_bisection_reference_numpy,
)


def test_parse_grid_validates_input():
    assert parse_positive_int_grid("8,4,12", "grid") == [4, 8, 12]
    for bad in ("", "4,4", "3,0", "4,hi", "-1,2"):
        with pytest.raises(ValueError):
            parse_positive_int_grid(bad, "grid")


def test_reproducible_inputs():
    kw = dict(n=1000, tau=2.0, active_ratio=0.7, seed=123,
              device="cpu", dtype=torch.float64)
    assert torch.equal(make_test_vector(**kw), make_test_vector(**kw))
    assert not torch.equal(make_test_vector(**kw), make_test_vector(**{**kw, "seed": 124}))


@pytest.mark.parametrize("p", [1 / 3, 0.75, 0.99])
def test_independent_reference_and_threshold(p):
    const = lp_constants(p, 1.0)
    points = np.array([0.0, -0.9 * const.tau, -0.999 * const.tau,
                       1.05 * const.tau, 2 * const.tau, -10 * const.tau])
    ref = prox_bisection_reference_numpy(points, p, 1.0)
    newton = make_lp_prox(p=p, lam=1.0, method="newton12", dtype=torch.float64,
                          newton_iters=50)
    compare = newton(torch.tensor(points, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(ref, compare, rtol=5e-13, atol=5e-13 * const.tau)
    assert np.all(ref[:3] == 0)
    assert np.all(np.sign(ref[3:]) == np.sign(points[3:]))
    active = np.abs(points) > const.tau
    resid = np.abs(np.abs(ref[active]) + p * np.abs(ref[active]) ** (p - 1) - np.abs(points[active]))
    assert np.max(resid) < 1e-11 * max(1, const.tau)


def test_chebyshev_discrete_transform_equals_least_squares():
    p = 1 / 3
    nfit = 128
    approx = fit_degree_family(p, [4, 8, 12], n_fit=nfit, n_validation=101)
    theta = np.pi * (np.arange(nfit) + 0.5) / nfit
    x = np.cos(theta)
    values = normalized_branch_bisect((x + 1) * 0.5, p)
    for d in [4, 8, 12]:
        expected = chebfit(x, values, d)
        np.testing.assert_allclose(approx.coefficients[:d + 1], expected, rtol=0, atol=2e-13)
    assert approx.sampled_errors[12] < approx.sampled_errors[8]


def test_preloaded_solvers_match_original_algorithms():
    p = 0.4
    lam = 1.0
    tau = lp_constants(p, lam).tau
    y = torch.tensor([0.0, -0.9*tau, 1.0001*tau, 2.0*tau,
                      -6.0*tau], dtype=torch.float64)
    for family, work, original, kw in [
        ("newton", 3, "newton8", {"newton_iters": 3}),
        ("newton", 7, "newton12", {"newton_iters": 7}),
        ("bisection", 8, "bisect40", {"bisect_iters": 8}),
    ]:
        fn = make_preloaded_prox(p=p, lam=lam, family=family,
                                 device=y.device, dtype=y.dtype, work=work)
        old_fn = make_lp_prox(p=p, lam=lam, method=original,
                             dtype=y.dtype, **kw)
        torch.testing.assert_close(fn(y), old_fn(y), rtol=0, atol=3e-15)


def test_preloaded_polynomial_is_degree_sensitive():
    approx = fit_degree_family(0.75, [4, 12], n_fit=256, n_validation=101)
    c = lp_constants(0.75, 1.0)
    y = torch.tensor([1.1*c.tau, 1.5*c.tau, 4*c.tau], dtype=torch.float64)
    fn4 = make_preloaded_prox(p=.75, lam=1., family="chebyshev",
                              device=y.device, dtype=y.dtype, work=4,
                              coefficients=approx.coefficients[:5])
    fn12 = make_preloaded_prox(p=.75, lam=1., family="chebyshev",
                               device=y.device, dtype=y.dtype, work=12,
                               coefficients=approx.coefficients[:13])
    ref = prox_bisection_reference_numpy(y.numpy(), .75, 1.)
    e4 = np.max(np.abs(fn4(y).numpy() - ref))
    e12 = np.max(np.abs(fn12(y).numpy() - ref))
    assert e12 < e4


def test_zero_safe_normalized_accuracy():
    p = 0.5
    y = np.array([0., -0.1, 0.1])
    x = np.zeros_like(y)
    m = measure_accuracy(y, x, x, p=p, lam=1.)
    assert m["max_norm_error"] == 0
    assert m["support_mismatch_count"] == 0
    assert m["all_finite"]

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_cuda_device_matching():
    device = torch.device("cuda")

    y = torch.tensor(
        [0.0, 2.0, 5.0],
        device=device,
        dtype=torch.float32,
    )

    fn = make_preloaded_prox(
        p=1 / 3,
        lam=1.0,
        family="newton",
        device=device,
        dtype=torch.float32,
        work=4,
    )

    out = fn(y)

    assert out.device == y.device
    assert torch.isfinite(out).all()