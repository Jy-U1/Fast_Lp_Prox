from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LpConstants:
    """
    Constants for the scalar proximal operator of lambda * |x|^p.

    Parameters
    ----------
    p:
        Exponent in (0, 1).
    lam:
        Positive regularization parameter lambda.
    u_star:
        Radius-related constant for the active branch.
    eps_tau:
        Normalized epsilon value at the activation threshold.
        This is independent of lambda.
    rho:
        Lower endpoint of the larger active root interval.
    tau:
        Activation threshold. prox(y) = 0 when |y| <= tau.
    z_tau:
        z value at the activation threshold.
    """

    p: float
    lam: float
    u_star: float
    eps_tau: float
    rho: float
    tau: float
    z_tau: float


def lp_constants(p: float, lam: float = 1.0) -> LpConstants:
    """
    Compute constants for prox_{lambda |.|^p}.

    The scalar proximal operator has a threshold structure:

        prox_{lambda |.|^p}(y) = 0, if |y| <= tau.

    For |y| > tau, the active root lies on the larger branch.
    """

    p = float(p)
    lam = float(lam)

    if not (0.0 < p < 1.0):
        raise ValueError(f"p must satisfy 0 < p < 1, got p={p}.")

    if lam <= 0.0:
        raise ValueError(f"lam must be positive, got lam={lam}.")

    u_star = (1.0 - p) ** (1.0 - p) / (2.0 - p) ** (2.0 - p)

    # eps_tau = lambda * p * tau^(p - 2)
    # This simplifies to p * 2^(1-p) * u_star, independent of lambda.
    eps_tau = p * (2.0 ** (1.0 - p)) * u_star

    rho = (2.0 * lam * (1.0 - p)) ** (1.0 / (2.0 - p))
    tau = ((2.0 - p) / (2.0 * (1.0 - p))) * rho

    z_tau = 2.0 * (1.0 - p) / (2.0 - p)

    return LpConstants(
        p=p,
        lam=lam,
        u_star=u_star,
        eps_tau=eps_tau,
        rho=rho,
        tau=tau,
        z_tau=z_tau,
    )