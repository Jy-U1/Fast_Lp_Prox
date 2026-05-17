from __future__ import annotations

from typing import Dict

import torch

from .constants import lp_constants


def objective_value(
    y: torch.Tensor,
    x: torch.Tensor,
    p: float,
    lam: float = 1.0,
) -> torch.Tensor:
    """
    Coordinatewise objective value:

        0.5 * (x - y)^2 + lam * |x|^p

    Returns a tensor with the same shape as y and x.
    """

    p = float(p)
    lam = float(lam)

    return 0.5 * (x - y) ** 2 + lam * torch.pow(torch.abs(x), p)


def active_mask(
    y: torch.Tensor,
    p: float,
    lam: float = 1.0,
) -> torch.Tensor:
    """
    Return mask for entries above the activation threshold.

    The scalar Lp prox returns zero when |y| <= tau.
    """

    const = lp_constants(p, lam)
    return torch.abs(y) > const.tau


def prox_stationary_residual(
    y: torch.Tensor,
    x: torch.Tensor,
    p: float,
    lam: float = 1.0,
    inactive_value: float = 0.0,
) -> torch.Tensor:
    """
    Compute the active-branch stationary residual.

    For active entries, the positive magnitude s = |x| should satisfy:

        s - |y| + lam * p * s^(p-1) = 0.

    For inactive entries, this function returns inactive_value.
    """

    p = float(p)
    lam = float(lam)

    a = torch.abs(y)
    s = torch.abs(x)

    mask = active_mask(y, p=p, lam=lam) & (s > 0)

    residual = torch.full_like(y, float(inactive_value))

    if mask.any():
        s_active = s[mask]
        a_active = a[mask]

        residual_active = s_active - a_active + lam * p * torch.pow(
            s_active,
            p - 1.0,
        )

        residual[mask] = residual_active

    return residual


def residual_error_bound(
    y: torch.Tensor,
    x: torch.Tensor,
    p: float,
    lam: float = 1.0,
    inactive_value: float = 0.0,
) -> torch.Tensor:
    """
    A posteriori active-root error bound.

    On the larger active branch:

        |x_hat - x_+| <= |R(x_hat; y)| / (1 - p / 2)

    where

        R(s; |y|) = s - |y| + lam * p * s^(p-1).

    For inactive entries, this function returns inactive_value.
    """

    p = float(p)

    residual = prox_stationary_residual(
        y=y,
        x=x,
        p=p,
        lam=lam,
        inactive_value=inactive_value,
    )

    mask = active_mask(y, p=p, lam=lam) & (torch.abs(x) > 0)

    bound = torch.full_like(y, float(inactive_value))

    if mask.any():
        bound[mask] = torch.abs(residual[mask]) / (1.0 - p / 2.0)

    return bound


def objective_gap_vs_reference(
    y: torch.Tensor,
    x: torch.Tensor,
    x_ref: torch.Tensor,
    p: float,
    lam: float = 1.0,
) -> torch.Tensor:
    """
    Coordinatewise objective gap relative to a reference output.
    """

    obj = objective_value(y, x, p=p, lam=lam)
    obj_ref = objective_value(y, x_ref, p=p, lam=lam)

    return obj - obj_ref


def summarize_tensor(
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Summarize a tensor, optionally restricted to a mask.

    Returns max, mean, median, and finite check.
    """

    if mask is not None:
        values = values[mask]

    if values.numel() == 0:
        return {
            f"{prefix}numel": 0.0,
            f"{prefix}all_finite": 1.0,
            f"{prefix}max": 0.0,
            f"{prefix}mean": 0.0,
            f"{prefix}median": 0.0,
        }

    abs_values = torch.abs(values)

    return {
        f"{prefix}numel": float(values.numel()),
        f"{prefix}all_finite": float(torch.isfinite(values).all().item()),
        f"{prefix}max": float(abs_values.max().item()),
        f"{prefix}mean": float(abs_values.mean().item()),
        f"{prefix}median": float(abs_values.median().item()),
    }