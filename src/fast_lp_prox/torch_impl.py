from __future__ import annotations

from typing import Callable, Literal

import torch

from .coeffs import dtype_to_name, get_or_build_cheb_approx
from .constants import lp_constants


MethodName = Literal["poly", "newton8", "newton12", "bisect40"]


def _as_torch_dtype(dtype: torch.dtype | None, fallback: torch.dtype) -> torch.dtype:
    if dtype is None:
        return fallback
    return dtype


def pow_p_minus_1_torch(x: torch.Tensor, p: float) -> torch.Tensor:
    """
    Compute x^(p-1).

    The p=1/2 case is specialized because it is common and faster.
    """

    if abs(float(p) - 0.5) < 1e-14:
        return torch.rsqrt(x)

    return torch.pow(x, float(p) - 1.0)


def pow_p_minus_2_torch(x: torch.Tensor, p: float) -> torch.Tensor:
    """
    Compute x^(p-2).

    The p=1/2 case is specialized because it is common and faster.
    """

    if abs(float(p) - 0.5) < 1e-14:
        return torch.rsqrt(x) / x

    return torch.pow(x, float(p) - 2.0)


def cheb_eval_torch(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate

        sum_k coeffs[k] T_k(x)

    using Clenshaw recurrence.

    Parameters
    ----------
    x:
        Tensor with values usually in [-1, 1].
    coeffs:
        1D tensor of Chebyshev coefficients.
    """

    if coeffs.ndim != 1:
        raise ValueError("coeffs must be a 1D tensor.")

    b1 = torch.zeros_like(x)
    b2 = torch.zeros_like(x)
    two_x = 2.0 * x

    for k in range(coeffs.shape[0] - 1, 0, -1):
        b0 = two_x * b1 - b2 + coeffs[k]
        b2 = b1
        b1 = b0

    return x * b1 - b2 + coeffs[0]


def make_lp_prox(
    p: float,
    lam: float = 1.0,
    method: MethodName = "poly",
    dtype: torch.dtype | None = None,
    target_error: float = 2e-7,
    max_degree: int = 96,
    n_fit: int = 8192,
    n_val: int = 80001,
    use_table: bool = True,
    newton_iters: int | None = None,
    bisect_iters: int | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a PyTorch proximal operator for lambda * |x|^p.

    Parameters
    ----------
    p:
        Exponent in (0, 1).
    lam:
        Positive regularization parameter.
    method:
        One of "poly", "newton8", "newton12", "bisect40".
    dtype:
        Optional coefficient dtype. If None, the input tensor dtype is used.
    target_error:
        Target Chebyshev z-error for method="poly".
    max_degree:
        Maximum Chebyshev degree for method="poly".
    n_fit:
        Number of fitting nodes for coefficient generation.
    n_val:
        Number of validation nodes for coefficient generation.
    newton_iters:
        Optional override for Newton methods.
    bisect_iters:
        Optional override for bisection methods.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        Function mapping y to prox_{lambda |.|^p}(y).
    """

    p = float(p)
    lam = float(lam)

    const = lp_constants(p, lam)
    rho_f = float(const.rho)
    tau_f = float(const.tau)
    eps_tau_f = float(const.eps_tau)

    if method == "poly":
        dtype_name = dtype_to_name(dtype if dtype is not None else "float32")

        approx = get_or_build_cheb_approx(
            p=p,
            dtype=dtype_name,
            use_table=use_table,
            target_error=target_error,
            max_degree=max_degree,
            n_fit=n_fit,
            n_val=n_val,
            verbose=False,
        )
        coeff_np = approx.coeffs
    else:
        coeff_np = None

    def prox_poly(y: torch.Tensor) -> torch.Tensor:
        local_dtype = _as_torch_dtype(dtype, y.dtype)
        coeffs = torch.as_tensor(coeff_np, device=y.device, dtype=local_dtype)

        a = torch.abs(y)

        # Avoid negative powers at zero.
        # Inactive entries are discarded later.
        a_safe = torch.clamp(a, min=tau_f)

        eps = lam * p * pow_p_minus_2_torch(a_safe, p)
        q = torch.clamp(eps / eps_tau_f, 0.0, 1.0)

        x_cheb = 2.0 * q - 1.0
        z = cheb_eval_torch(x_cheb, coeffs)

        x = a * z
        return torch.where(a > tau_f, torch.sign(y) * x, torch.zeros_like(y))

    def prox_newton(y: torch.Tensor, iters: int) -> torch.Tensor:
        a = torch.abs(y)
        active = a > tau_f

        rho_t = torch.tensor(rho_f, device=y.device, dtype=y.dtype)
        a_safe = torch.maximum(a, rho_t)

        x = a_safe.clone()

        for _ in range(iters):
            xp1 = pow_p_minus_1_torch(x, p)
            residual = x - a + lam * p * xp1
            derivative = 1.0 - lam * p * (1.0 - p) * (xp1 / x)

            x_new = x - residual / derivative

            # Keep Newton on the larger-root interval [rho, |y|].
            x_new = torch.maximum(x_new, rho_t)
            x_new = torch.minimum(x_new, a_safe)
            x = x_new

        return torch.where(active, torch.sign(y) * x, torch.zeros_like(y))

    def prox_bisect(y: torch.Tensor, iters: int) -> torch.Tensor:
        a = torch.abs(y)
        active = a > tau_f

        rho_t = torch.tensor(rho_f, device=y.device, dtype=y.dtype)

        hi = torch.maximum(a, rho_t)
        lo = torch.full_like(a, rho_f)

        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            residual = mid - a + lam * p * pow_p_minus_1_torch(mid, p)

            # On [rho, infinity), residual is increasing.
            hi = torch.where(residual > 0.0, mid, hi)
            lo = torch.where(residual > 0.0, lo, mid)

        x = 0.5 * (lo + hi)
        return torch.where(active, torch.sign(y) * x, torch.zeros_like(y))

    if method == "poly":
        return prox_poly

    if method == "newton8":
        iters = 8 if newton_iters is None else int(newton_iters)
        return lambda y: prox_newton(y, iters=iters)

    if method == "newton12":
        iters = 12 if newton_iters is None else int(newton_iters)
        return lambda y: prox_newton(y, iters=iters)

    if method == "bisect40":
        iters = 40 if bisect_iters is None else int(bisect_iters)
        return lambda y: prox_bisect(y, iters=iters)

    raise ValueError(
        f"Unknown method={method!r}. "
        "Expected one of: 'poly', 'newton8', 'newton12', 'bisect40'."
    )


def lp_prox(
    y: torch.Tensor,
    p: float,
    lam: float = 1.0,
    method: MethodName = "poly",
    **kwargs,
) -> torch.Tensor:
    """
    Convenience wrapper for one-off proximal evaluation.

    For repeated calls, prefer make_lp_prox(...), because method='poly'
    builds Chebyshev coefficients.
    """

    prox = make_lp_prox(p=p, lam=lam, method=method, dtype=y.dtype, **kwargs)
    return prox(y)