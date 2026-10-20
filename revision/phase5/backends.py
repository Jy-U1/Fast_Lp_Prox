"""Additional scalar lp proximity backends for a Reviewer-2 comparison.

Three intentionally separate concepts:
  * fixed_term_series: literal N-term Lagrange--Buermann expansion, no fallback;
  * residual_series_bisect: posterior residual checked series plus bisection;
  * select_against_zero: a scalar OBJECTIVE guard, not a convergence certificate.

Algorithm 5.7 / Eq. (28) of arXiv:2606.31951v2 motivate the residual
criterion. The hybrid implemented HERE is NOT that work's MB hybrid and is
NOT an interval-certified floating-point evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import time
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class ProxConstants:
    p: float
    lam: float
    rho: float
    tau: float
    ustar: float
    qp: float


def proximal_constants(p: float, lam: float) -> ProxConstants:
    p = float(p)
    lam = float(lam)
    if not (math.isfinite(p) and 0.0 < p < 1.0):
        raise ValueError("p must be finite and strictly between 0 and 1")
    if not (math.isfinite(lam) and lam > 0.0):
        raise ValueError("lam must be finite and positive")
    log_rho = math.log(2.0 * lam * (1.0 - p)) / (2.0 - p)
    rho = math.exp(log_rho)
    tau = rho * (2.0 - p) / (2.0 * (1.0 - p))
    log_ustar = (1.0 - p) * math.log1p(-p) - (2.0 - p) * math.log(2.0 - p)
    ustar = math.exp(log_ustar)
    qp = p * 2.0 ** (1.0 - p)
    if not (math.isfinite(rho) and 0 < rho < tau and math.isfinite(tau) and tau > 0.0):
        raise ValueError("proximal thresholds overflowed/underflowed; restrict parameter scale")
    return ProxConstants(p=p, lam=lam, rho=rho, tau=tau, ustar=ustar, qp=qp)


@lru_cache(maxsize=64)
def scaled_series_coefficients(p: float, terms: int) -> tuple[float, ...]:
    """b_n := Gamma((2-p)n-1)/(n! Gamma((1-p)n)) * ustar**n.

    Scaling by ustar makes the coefficients stable even when unscaled
    a_n grows exponentially and the input epsilon approaches ustar.
    """
    if not (0.0 < p < 1.0 and math.isfinite(p)):
        raise ValueError("invalid exponent")
    if not 1 <= terms <= 16384:
        raise ValueError("series term budget must belong to [1,16384]")
    log_u = (1.0 - p) * math.log1p(-p) - (2.0 - p) * math.log(2.0 - p)
    out = []
    for n in range(1, terms + 1):
        lg = (math.lgamma((2.0 - p) * n - 1.0)
              - math.lgamma((1.0 - p) * n) - math.lgamma(n + 1.0)
              + n * log_u)
        c = math.exp(lg)
        if not math.isfinite(c) or c < 0.0:
            raise ArithmeticError(f"nonfinite series coefficient at n={n}")
        out.append(c)
    return tuple(out)


def _validate_tensor(y: torch.Tensor) -> None:
    if not isinstance(y, torch.Tensor) or y.dtype not in (torch.float32, torch.float64):
        raise TypeError("y must be a float32 or float64 torch.Tensor")
    if not bool(torch.isfinite(y).all()):
        raise ValueError("input contains nonfinite entries")


def _t_normalized(yabs: torch.Tensor, c: ProxConstants) -> torch.Tensor:
    # t = epsilon/u_star = q_p * (tau/|y|)^(2-p), active only.
    # The clamp avoids division by zero and positive-power overflow on the
    # inactive branch while guaranteeing t <= q_p < 1 on active entries.
    return c.qp * (c.tau / torch.clamp(yabs, min=c.tau)).pow(2.0 - c.p)


def _eval_series_z(t: torch.Tensor, coeff: tuple[float, ...]) -> torch.Tensor:
    # Horner: sum_{n=1}^N b_n t^n = t*(b_1+t*(b_2+...)).
    poly = torch.full_like(t, coeff[-1])
    for v in reversed(coeff[:-1]):
        poly = poly * t + v
    return 1.0 - t * poly


def fixed_term_series(y: torch.Tensor, *, p: float, lam: float,
                      terms: int, coefficients: Optional[tuple[float, ...]] = None) -> torch.Tensor:
    """N-term Lagrange--Buermann positive branch; exact zero selection.

    This is a genuine fixed N budget; it can have large error near activation
    when p is close to 1. Neither an objective guard nor fallback is applied.
    """
    _validate_tensor(y)
    c = proximal_constants(p, lam)
    coeff = coefficients if coefficients is not None else scaled_series_coefficients(float(p), int(terms))
    if len(coeff) != terms:
        raise ValueError("preloaded series coefficients have the wrong length")
    yabs = y.abs()
    t = _t_normalized(yabs, c)
    z = _eval_series_z(t, coeff)
    candidate = torch.sign(y) * yabs * z
    return torch.where(yabs > c.tau, candidate, torch.zeros_like(y))


class FixedSeries:
    """Prepared series backend, with coefficient generation excluded from online timing."""
    def __init__(self, *, p: float, lam: float, terms: int):
        self.p = float(p)
        self.lam = float(lam)
        self.terms = int(terms)
        self.c = proximal_constants(self.p, self.lam)
        self.coefficients = scaled_series_coefficients(self.p, self.terms)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        return fixed_term_series(y, p=self.p, lam=self.lam, terms=self.terms,
                                 coefficients=self.coefficients)


def normalized_stationarity(yabs: torch.Tensor, w: torch.Tensor,
                             c: ProxConstants) -> torch.Tensor:
    # R(w)=w-1+epsilon*w^(p-1), epsilon = ustar*t.
    # When y=0, R is not defined, so this helper is called on safe active
    # proxy magnitudes, and the caller masks inactive results.
    eps = c.ustar * _t_normalized(yabs, c)
    return w - 1.0 + eps * torch.clamp(w, min=torch.finfo(w.dtype).tiny).pow(c.p - 1.0)


def reference_bisection(y: torch.Tensor, *, p: float, lam: float,
                        iterations: int = 75) -> torch.Tensor:
    """Independent float64 root bracket on [rho/y,1]; zero at threshold.

    Numerical reference; not a directed-rounding interval certificate.
    """
    _validate_tensor(y)
    if not (1 <= int(iterations) <= 512):
        raise ValueError("invalid bisection budget")
    c = proximal_constants(p, lam)
    ya = y.to(torch.float64).abs()
    active = ya > c.tau
    safe_y = torch.clamp(ya, min=c.tau)
    lo = c.rho / safe_y
    hi = torch.ones_like(lo)
    eps = c.ustar * _t_normalized(safe_y, c)
    for _ in range(int(iterations)):
        mid = lo + (hi - lo) * 0.5
        resid = mid - 1.0 + eps * mid.pow(p - 1.0)
        lo = torch.where(resid <= 0.0, mid, lo)
        hi = torch.where(resid > 0.0, mid, hi)
    w = lo + 0.5 * (hi - lo)
    return torch.where(active, torch.sign(y) * ya * w,
                       torch.zeros_like(ya))


@dataclass
class HybridStatistics:
    total_active: int
    series_accepted: int
    fallback_count: int
    max_indicator_accepted: float


class ResidualSeriesBisection:
    """Series estimates checked by a posterior residual, bisection fallback.

    The denominator (1-p/2) of the indicator is valid only when the
    candidate belongs to the positive-branch bracket [rho/y,1].
    In floating point this is a diagnostic bound, not a formal certificate.
    """
    def __init__(self, *, p: float, lam: float, orders: tuple[int, ...] = (8, 32, 128),
                 tol: float = 1e-6, bisect_steps: int = 56):
        self.c = proximal_constants(p, lam)
        if not orders or any(a < 1 or a > 16384 for a in orders):
            raise ValueError("orders must be nonempty and all in [1,16384]")
        if list(orders) != sorted(set(orders)):
            raise ValueError("orders must be strictly increasing")
        if not (math.isfinite(tol) and 0 < tol < 1):
            raise ValueError("invalid normalized tolerance")
        if not 1 <= bisect_steps <= 512:
            raise ValueError("invalid bisection budget")
        self.orders = tuple(int(o) for o in orders)
        self.tol = float(tol)
        self.bisect_steps = int(bisect_steps)
        self.coeffs = {o: scaled_series_coefficients(self.c.p, o) for o in self.orders}
        self.last_stats: Optional[HybridStatistics] = None

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        _validate_tensor(y)
        c = self.c
        # Residual checks and fallback in double even for float32 throughput.
        ya = y.to(torch.float64).abs()
        active = ya > c.tau
        ys = torch.clamp(ya, min=c.tau)
        lo = c.rho / ys
        t = _t_normalized(ys, c)
        best = torch.zeros_like(ya)
        unresolved = active.clone()
        best_indicator = 0.0
        accepted = 0
        for order in self.orders:
            # Evaluate only with preloaded coefficients; every input is the
            # same tensor and no per-sample Python loops are used.
            w = _eval_series_z(t, self.coeffs[order])
            in_bracket = (w >= lo) & (w <= 1.0) & torch.isfinite(w)
            resid = normalized_stationarity(ys, w, c).abs()
            indicator = resid / (1.0 - c.p / 2.0)
            good = unresolved & in_bracket & (indicator <= self.tol)
            best = torch.where(good, w, best)
            if bool(good.any()):
                accepted += int(good.sum().item())
                best_indicator = max(best_indicator, float(indicator[good].max().item()))
            unresolved = unresolved & ~good
            if not bool(unresolved.any()):
                break
        count_fallback = int(unresolved.sum().item())
        if count_fallback:
            eps = c.ustar * t
            hi = torch.ones_like(lo)
            low = lo.clone()
            for _ in range(self.bisect_steps):
                mid = low + 0.5 * (hi - low)
                r = mid - 1.0 + eps * mid.pow(c.p - 1.0)
                low = torch.where(unresolved & (r <= 0.0), mid, low)
                hi = torch.where(unresolved & (r > 0.0), mid, hi)
            best = torch.where(unresolved, low + 0.5 * (hi - low), best)
        self.last_stats = HybridStatistics(total_active=int(active.sum().item()),
                                         series_accepted=accepted,
                                         fallback_count=count_fallback,
                                         max_indicator_accepted=best_indicator)
        out = torch.where(active, torch.sign(y).to(torch.float64) * ya * best,
                          torch.zeros_like(ya))
        return out.to(y.dtype)


def select_against_zero(y: torch.Tensor, candidate: torch.Tensor, *,
                        p: float, lam: float, safety_factor: float = 16.0) -> torch.Tensor:
    """Reject approximate active candidate unless its proximal objective beats zero.

    Scalar objective difference is evaluated with float64 arithmetic.
    This alone does NOT imply an inexact-prox model descent or convergence.
    """
    _validate_tensor(y)
    if not isinstance(candidate, torch.Tensor) or candidate.dtype != y.dtype or candidate.device != y.device or candidate.shape != y.shape:
        raise ValueError("candidate must match input in dtype, device and shape")
    _ = proximal_constants(p, lam)
    if not (math.isfinite(safety_factor) and safety_factor >= 0):
        raise ValueError("invalid safety factor")
    yy = y.to(torch.float64)
    xx = candidate.to(torch.float64)
    q = 0.5 * xx.square()
    linear = yy * xx
    penalty = lam * xx.abs().pow(p)
    diff = q - linear + penalty
    scale = q + linear.abs() + penalty
    margin = safety_factor * torch.finfo(torch.float64).eps * scale
    accepted = torch.isfinite(diff) & torch.isfinite(scale) & (yy * xx >= 0.0) & (diff < -margin)
    return torch.where(accepted, candidate, torch.zeros_like(candidate))
