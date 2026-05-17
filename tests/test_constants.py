import math

import pytest

from fast_lp_prox import LpConstants, lp_constants


def test_lp_constants_returns_dataclass():
    const = lp_constants(p=0.5, lam=1.0)

    assert isinstance(const, LpConstants)
    assert const.p == 0.5
    assert const.lam == 1.0


def test_lp_constants_half_values():
    const = lp_constants(p=0.5, lam=1.0)

    assert math.isclose(const.rho, 1.0, rel_tol=1e-12)
    assert math.isclose(const.tau, 1.5, rel_tol=1e-12)
    assert math.isclose(const.z_tau, 2.0 / 3.0, rel_tol=1e-12)

    expected_u_star = (1.0 - 0.5) ** (1.0 - 0.5) / (2.0 - 0.5) ** (2.0 - 0.5)
    expected_eps_tau = 0.5 * (2.0 ** 0.5) * expected_u_star

    assert math.isclose(const.u_star, expected_u_star, rel_tol=1e-12)
    assert math.isclose(const.eps_tau, expected_eps_tau, rel_tol=1e-12)


def test_eps_tau_independent_of_lambda():
    const1 = lp_constants(p=0.5, lam=1.0)
    const2 = lp_constants(p=0.5, lam=10.0)

    assert math.isclose(const1.eps_tau, const2.eps_tau, rel_tol=1e-12)


@pytest.mark.parametrize("bad_p", [-1.0, 0.0, 1.0, 2.0])
def test_invalid_p_raises(bad_p):
    with pytest.raises(ValueError):
        lp_constants(p=bad_p, lam=1.0)


@pytest.mark.parametrize("bad_lam", [-1.0, 0.0])
def test_invalid_lambda_raises(bad_lam):
    with pytest.raises(ValueError):
        lp_constants(p=0.5, lam=bad_lam)