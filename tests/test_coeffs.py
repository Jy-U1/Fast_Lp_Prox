import math

import numpy as np

from fast_lp_prox import build_cheb_coeffs, eval_cheb_on_q, lp_constants, z_ref_numpy_vec


def test_z_ref_endpoints():
    p = 0.5
    const = lp_constants(p)

    z0 = z_ref_numpy_vec(np.array([0.0]), p)[0]
    z1 = z_ref_numpy_vec(np.array([1.0]), p)[0]

    assert math.isclose(z0, 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(z1, const.z_tau, rel_tol=1e-12, abs_tol=1e-12)


def test_z_ref_is_decreasing():
    p = 0.5
    q = np.linspace(0.0, 1.0, 101)
    z = z_ref_numpy_vec(q, p)

    diffs = np.diff(z)

    assert np.all(diffs <= 1e-12)


def test_build_cheb_coeffs_small_problem():
    approx = build_cheb_coeffs(
        p=0.5,
        target_error=1e-4,
        max_degree=24,
        n_fit=512,
        n_val=1001,
        verbose=False,
    )

    assert approx.degree <= 24
    assert approx.estimated_error <= 1e-4
    assert len(approx.coeffs) == approx.degree + 1


def test_eval_cheb_on_q_matches_reference_reasonably():
    p = 0.5

    approx = build_cheb_coeffs(
        p=p,
        target_error=1e-4,
        max_degree=24,
        n_fit=512,
        n_val=1001,
        verbose=False,
    )

    q = np.linspace(0.0, 1.0, 200)
    z_ref = z_ref_numpy_vec(q, p)
    z_hat = eval_cheb_on_q(q, approx.coeffs)

    max_err = np.max(np.abs(z_hat - z_ref))

    assert max_err <= 2e-4