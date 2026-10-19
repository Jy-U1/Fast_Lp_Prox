"""Phase-1, accuracy-matched benchmark helpers for the scalar ell_p prox.

This module is *additive*: original paper scripts and torch_impl.py remain unchanged.
Offline fitting/reference calculations use NumPy float64; online candidate kernels
use PyTorch in the requested dtype/device. The Newton and bisection steps copy
the mathematical iteration of torch_impl, while preloading reusable scalars.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .constants import lp_constants
from .torch_impl import cheb_eval_torch, pow_p_minus_1_torch, pow_p_minus_2_torch


@dataclass(frozen=True)
class OfflineChebyshev:
    p: float
    coefficients: np.ndarray  # float64, degrees 0,...,Dmax
    n_fit: int
    n_validation: int
    max_degree: int
    sampled_errors: dict[int, float]


def parse_positive_int_grid(spec: str, name: str) -> list[int]:
    """Parse a comma-separated increasing list, rejecting empty/invalid grids."""
    try:
        nums = [int(part.strip()) for part in spec.split(",")]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a comma-separated list of positive integers") from exc
    if not nums or any(n < 1 for n in nums) or len(set(nums)) != len(nums):
        raise ValueError(f"{name} must contain distinct positive integers")
    return sorted(nums)


def normalized_branch_bisect(q: np.ndarray, p: float, *, iters: int = 75) -> np.ndarray:
    """Independently compute g_p(q) using bracketed bisection on [z_tau,1]."""
    q = np.asarray(q, dtype=np.float64)
    if not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
        raise ValueError("q must be finite and in [0,1]")
    const = lp_constants(p, lam=1.0)
    lo = np.full_like(q, const.z_tau, dtype=np.float64)
    hi = np.ones_like(q, dtype=np.float64)
    eps = const.eps_tau * q
    for _ in range(iters):
        mid = lo + 0.5 * (hi - lo)
        phi = (1.0 - mid) * np.power(mid, 1.0 - p)
        # phi decreases on the active branch: phi(mid)>eps -> root to the right
        move_right = phi > eps
        lo = np.where(move_right, mid, lo)
        hi = np.where(move_right, hi, mid)
    z = lo + 0.5 * (hi - lo)
    return np.where(q <= 0.0, 1.0, np.where(q >= 1.0, const.z_tau, z))


def fit_degree_family(
    p: float,
    degrees: list[int],
    *,
    n_fit: int = 8192,
    n_validation: int = 4001,
) -> OfflineChebyshev:
    """Fit all degrees using one Chebyshev-Gauss transform, then validate.

    At x_j=cos((j+1/2)pi/N), the discrete LS coefficients are
      c_0 = (1/N) sum_j f(x_j),
      c_k = (2/N) sum_j f(x_j) cos(k (j+1/2)pi/N), k>=1.
    This is equivalent to independent chebfit on each degree D<N in exact
    arithmetic, but avoids repeated dense least-squares solves.
    """
    if n_fit <= max(degrees) or n_validation < 3:
        raise ValueError("Require n_fit > max(degrees) and n_validation >= 3")
    deg_max = max(degrees)
    theta = (np.arange(n_fit, dtype=np.float64) + 0.5) * (np.pi / n_fit)
    q_nodes = 0.5 * (1.0 + np.cos(theta))
    values = normalized_branch_bisect(q_nodes, p)
    k = np.arange(deg_max + 1, dtype=np.float64)
    coeffs = (2.0 / n_fit) * (np.cos(k[:, None] * theta[None, :]) @ values)
    coeffs[0] *= 0.5
    qval = np.linspace(0.0, 1.0, n_validation, dtype=np.float64)
    zval = normalized_branch_bisect(qval, p)
    xval = 2.0 * qval - 1.0
    errors: dict[int, float] = {}
    for d in degrees:
        approximation = np.polynomial.chebyshev.chebval(xval, coeffs[:d + 1])
        errors[d] = float(np.max(np.abs(approximation - zval)))
    return OfflineChebyshev(float(p), coeffs, n_fit, n_validation, deg_max, errors)


def prox_bisection_reference_numpy(
    y: np.ndarray, p: float, lam: float, *, iters: int = 75
) -> np.ndarray:
    """Float64 reference: scalar upper-root bisection in original x-coordinates.

    This solver does not call the repo's Newton implementation and is independent
    of the Chebyshev fitting routine. At |y|=tau, it chooses the zero minimizer.
    """
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise ValueError("All reference inputs must be finite")
    const = lp_constants(p, lam)
    a = np.abs(y)
    result = np.zeros_like(y)
    mask = a > const.tau
    if not np.any(mask):
        return result
    a_act = a[mask]
    lo = np.full_like(a_act, const.rho)
    hi = a_act.copy()
    for _ in range(iters):
        mid = lo + 0.5 * (hi - lo)
        hmid = mid + lam * p * np.power(mid, p - 1.0) - a_act
        right_of_root = hmid > 0.0
        hi = np.where(right_of_root, mid, hi)
        lo = np.where(right_of_root, lo, mid)
    result[mask] = np.sign(y[mask]) * (lo + 0.5 * (hi - lo))
    return result


def make_preloaded_prox(
    *,
    p: float,
    lam: float,
    family: str,
    device: torch.device,
    dtype: torch.dtype,
    work: int,
    coefficients: np.ndarray | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build an online-only candidate with constants/tables loaded beforehand.

    Existing projected Newton and bisection update rules are unchanged except
    that rho is materialized once, not afresh on every call.
    """
    const = lp_constants(p, lam)
    tau, rho, eps_tau = float(const.tau), float(const.rho), float(const.eps_tau)
    rho_tensor = torch.tensor(rho, device=device, dtype=dtype)
    poly_coeffs = None
    if family == "chebyshev":
        if coefficients is None or len(coefficients) != work + 1:
            raise ValueError("Chebyshev coefficients must have length D+1")
        poly_coeffs = torch.as_tensor(np.array(coefficients, copy=True), device=device, dtype=dtype)
    elif family not in {"newton", "bisection"}:
        raise ValueError(f"Unknown family: {family}")
    if work <= 0:
        raise ValueError("Degree / iteration count must be positive")

    def evaluate(y: torch.Tensor) -> torch.Tensor:
        if y.device != device or y.dtype != dtype:
            raise ValueError("Input dtype and device must match the preloaded evaluator")
        a = torch.abs(y)
        active = a > tau
        if family == "chebyshev":
            a_safe = torch.clamp(a, min=tau)
            eps = lam * p * pow_p_minus_2_torch(a_safe, p)
            q = torch.clamp(eps / eps_tau, 0.0, 1.0)
            z = cheb_eval_torch(2.0 * q - 1.0, poly_coeffs)
            x = a * z
            return torch.where(active, torch.sign(y) * x, torch.zeros_like(y))
        if family == "newton":
            a_safe = torch.maximum(a, rho_tensor)
            x = a_safe.clone()  # exact initial iterate used by torch_impl.py
            for _ in range(work):
                xp1 = pow_p_minus_1_torch(x, p)
                h = x - a + lam * p * xp1
                dh = 1.0 - lam * p * (1.0 - p) * (xp1 / x)
                proposal = x - h / dh
                x = torch.minimum(torch.maximum(proposal, rho_tensor), a_safe)
            return torch.where(active, torch.sign(y) * x, torch.zeros_like(y))
        hi = torch.maximum(a, rho_tensor)
        lo = torch.full_like(a, rho)
        for _ in range(work):
            mid = 0.5 * (lo + hi)
            hmid = mid - a + lam * p * pow_p_minus_1_torch(mid, p)
            hi = torch.where(hmid > 0.0, mid, hi)
            lo = torch.where(hmid > 0.0, lo, mid)
        x = 0.5 * (lo + hi)
        return torch.where(active, torch.sign(y) * x, torch.zeros_like(y))
    return evaluate


def measure_accuracy(
    y: np.ndarray, out: np.ndarray, reference: np.ndarray, *, p: float, lam: float
) -> dict[str, float | int | bool]:
    """Full-input metrics with normalization stable at zero outputs."""
    y = np.asarray(y, dtype=np.float64)
    out = np.asarray(out, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if out.shape != y.shape or reference.shape != y.shape:
        raise ValueError("All metrics inputs must have equal shape")
    const = lp_constants(p, lam)
    finite = bool(np.all(np.isfinite(out)))
    abs_err = np.abs(out - reference)
    norm_err = abs_err / np.maximum(np.abs(y), const.tau)
    active = np.abs(y) > const.tau
    if np.any(active):
        xact = np.abs(out[active])
        # If an approximate candidate outputs zero/NaN on an active entry,
        # residual is infinite, not artificially treated as satisfactory.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            residual = np.abs(xact + lam * p * np.power(xact, p - 1.0) - np.abs(y[active]))
        residual = np.where(np.isfinite(residual), residual, np.inf)
        rmax, rmean = float(np.max(residual)), float(np.mean(residual))
    else:
        rmax, rmean = 0.0, 0.0
    obj = 0.5 * (out - y) ** 2 + lam * np.abs(out) ** p
    obj_ref = 0.5 * (reference - y) ** 2 + lam * np.abs(reference) ** p
    return {
        "all_finite": finite,
        "max_abs_error": float(np.max(abs_err)),
        "mean_abs_error": float(np.mean(abs_err)),
        "max_norm_error": float(np.max(norm_err)),
        "mean_norm_error": float(np.mean(norm_err)),
        "max_active_stationarity_residual": rmax,
        "mean_active_stationarity_residual": rmean,
        "mean_prox_objective_gap": float(np.mean(obj - obj_ref)),
        "support_mismatch_count": int(np.count_nonzero((out != 0.0) != (reference != 0.0))),
        "evaluated_active_count": int(np.count_nonzero(active)),
    }
