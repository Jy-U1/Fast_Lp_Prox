"""Regression tests for the Phase-5 float32 activation discrepancy.

These tests run against the actual fast_lp_prox.revision_bench module in the
project (not a mocked implementation). A CUDA device is tested if available.
"""
import numpy as np
import pytest
import torch

from fast_lp_prox.constants import lp_constants
from fast_lp_prox.revision_bench import (
    _float32_activation_cutoff,
    make_preloaded_prox,
    prox_bisection_reference_numpy,
)

P_VALUES = (1.0 / 3, 0.4, 0.75, 0.9, 0.99, 0.999)
METHODS = ("chebyshev", "newton", "bisection")


def _boundary_probes(tau: float) -> np.ndarray:
    """Inputs surrounding the *rounded* float32 threshold, both signs."""
    rounded = np.float32(tau)
    assert np.isfinite(rounded)
    down = np.nextafter(rounded, np.float32(-np.inf))
    up = np.nextafter(rounded, np.float32(np.inf))
    magnitudes = np.array([0.0, down, rounded, up, 1.05 * tau, 10.0 * tau],
                          dtype=np.float32)
    return np.concatenate((magnitudes, -magnitudes))


@pytest.mark.parametrize("tau", [0.0, 0.1, 0.9, 1.0, 1.0 + 1e-8, 1e-30, 2.0])
def test_cutoff_exact_for_finite_float32_grid_neighbors(tau):
    cutoff = _float32_activation_cutoff(tau)
    assert cutoff <= tau
    t = np.float32(cutoff)
    assert float(t) == cutoff
    next_up = np.nextafter(t, np.float32(np.inf))
    assert float(next_up) > tau
    probes = np.array([0.0, t, next_up], dtype=np.float32)
    assert np.array_equal(probes > np.float32(cutoff),
                          probes.astype(np.float64) > tau)


@pytest.mark.parametrize("p", (0.4, 0.9, 0.99, 0.999))
def test_regression_old_rounding_incorrectly_suppressed_active_value(p):
    tau = lp_constants(p, 1.0).tau
    up_rounded = torch.tensor(tau, dtype=torch.float32)
    assert float(up_rounded) > tau
    assert bool(up_rounded.double() > tau)
    # This is the specific legacy bug: direct tensor > Python float
    # rounds the right-hand side upward to the same float32 grid point.
    assert not bool(up_rounded > tau)
    assert bool(up_rounded > _float32_activation_cutoff(tau))


@pytest.mark.parametrize("p", P_VALUES)
def test_float32_threshold_matches_reference_grid(p):
    tau = lp_constants(p, 1.0).tau
    y = _boundary_probes(tau)
    cutoff = _float32_activation_cutoff(tau)
    actual = np.abs(y) > np.float32(cutoff)
    expected = np.abs(y.astype(np.float64)) > tau
    assert np.array_equal(actual, expected)
    assert np.array_equal(
        expected, prox_bisection_reference_numpy(y, p=p, lam=1.0) != 0.0
    )


@pytest.mark.parametrize("family", METHODS)
@pytest.mark.parametrize("p", P_VALUES)
def test_all_preloaded_float32_backends_agree_near_threshold(p, family):
    tau = lp_constants(p, 1.0).tau
    y = _boundary_probes(tau)
    # Constant polynomial z=1 tests the support rule without confusing it
    # with an independently chosen Chebyshev approximation degree.
    coeffs = np.array([1.0, 0.0]) if family == "chebyshev" else None
    backend = make_preloaded_prox(p=p, lam=1.0, family=family,
                                  device=torch.device("cpu"),
                                  dtype=torch.float32, work=1,
                                  coefficients=coeffs)
    result = backend(torch.from_numpy(y.copy())).detach().cpu().numpy()
    expected = np.abs(y.astype(np.float64)) > tau
    assert np.isfinite(result).all()
    assert np.array_equal(result != 0.0, expected)
    # The evaluator must also preserve the required odd symmetry.
    half = len(y) // 2
    np.testing.assert_allclose(result[half:], -result[:half], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("family", METHODS)
@pytest.mark.parametrize("p", P_VALUES)
def test_float64_legacy_branch_unchanged(p, family):
    tau = lp_constants(p, 1.0).tau
    threshold = torch.tensor(tau, dtype=torch.float64)
    negative_inf = torch.tensor(float("-inf"), dtype=torch.float64)
    positive_inf = torch.tensor(float("inf"), dtype=torch.float64)
    y = torch.stack((torch.nextafter(threshold, negative_inf),
                     threshold, torch.nextafter(threshold, positive_inf),
                     torch.tensor(3.0 * tau, dtype=torch.float64)))
    coeffs = np.array([1.0, 0.0]) if family == "chebyshev" else None
    backend = make_preloaded_prox(p=p, lam=1.0, family=family,
                                  device=torch.device("cpu"),
                                  dtype=torch.float64, work=1,
                                  coefficients=coeffs)
    actual = backend(y) != 0.0
    expected = y.abs() > tau
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("family", METHODS)
@pytest.mark.parametrize("p", (0.4, 0.9, 0.999))
def test_cuda_threshold_branch(p, family):
    tau = lp_constants(p, 1.0).tau
    y = torch.tensor(_boundary_probes(tau), device="cuda", dtype=torch.float32)
    coeffs = np.array([1.0, 0.0]) if family == "chebyshev" else None
    backend = make_preloaded_prox(p=p, lam=1.0, family=family,
                                  device=torch.device("cuda"),
                                  dtype=torch.float32, work=1,
                                  coefficients=coeffs)
    assert torch.equal(backend(y) != 0.0, y.double().abs() > tau)
