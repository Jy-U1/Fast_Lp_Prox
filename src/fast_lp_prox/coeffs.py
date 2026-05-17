from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval

from .constants import lp_constants


@dataclass(frozen=True)
class ChebyshevApprox:
    """
    Chebyshev approximation for the active branch z(eps_tau * q).

    The approximation has the form:

        z(q) approx sum_k coeffs[k] T_k(2q - 1),

    where q is in [0, 1].
    """

    p: float
    coeffs: np.ndarray
    degree: int
    estimated_error: float
    target_error: float
    n_fit: int
    n_val: int


def z_ref_numpy_vec(q, p: float, iters: int = 80) -> np.ndarray:
    """
    High-accuracy vectorized Newton solver for

        (1 - z) z^(1-p) = eps_tau q,

    on the larger active branch z in [z_tau, 1].

    Parameters
    ----------
    q:
        Scalar or array-like values in [0, 1].
    p:
        Exponent in (0, 1).
    iters:
        Number of Newton iterations.

    Returns
    -------
    np.ndarray
        Reference values z(q).
    """

    p = float(p)
    q_arr = np.asarray(q, dtype=np.float64)

    const = lp_constants(p, lam=1.0)
    eps = const.eps_tau * q_arr
    z_tau = const.z_tau

    # Linear initial guess from z(0)=1 to z(1)=z_tau.
    z = 1.0 - (1.0 - z_tau) * q_arr
    z = np.clip(z, z_tau, 1.0)

    for _ in range(iters):
        z_1mp = np.power(z, 1.0 - p)
        f = (1.0 - z) * z_1mp - eps
        fp = -z_1mp + (1.0 - z) * (1.0 - p) * np.power(z, -p)

        z_new = z - f / fp
        z = np.clip(z_new, z_tau, 1.0)

    z = np.where(q_arr <= 0.0, 1.0, z)
    z = np.where(q_arr >= 1.0, z_tau, z)

    return z


def build_cheb_coeffs(
    p: float,
    target_error: float = 2e-7,
    max_degree: int = 96,
    n_fit: int = 8192,
    n_val: int = 80001,
    verbose: bool = False,
) -> ChebyshevApprox:
    """
    Build a Chebyshev approximation for

        g(q) = z(eps_tau q), q in [0, 1].

    The polynomial is fitted in x = 2q - 1, x in [-1, 1].

    Returns
    -------
    ChebyshevApprox
        Dataclass containing coefficients, degree, and validation error.
    """

    p = float(p)
    target_error = float(target_error)

    if target_error <= 0.0:
        raise ValueError("target_error must be positive.")

    if max_degree < 1:
        raise ValueError("max_degree must be at least 1.")

    if n_fit < max_degree + 1:
        raise ValueError("n_fit must be larger than max_degree.")

    if n_val < 10:
        raise ValueError("n_val must be at least 10.")

    # Chebyshev nodes in [-1, 1].
    x_nodes = np.cos(np.pi * (np.arange(n_fit) + 0.5) / n_fit)
    q_nodes = 0.5 * (x_nodes + 1.0)
    z_nodes = z_ref_numpy_vec(q_nodes, p)

    # Validation grid.
    x_val = np.linspace(-1.0, 1.0, n_val)
    q_val = 0.5 * (x_val + 1.0)
    z_val = z_ref_numpy_vec(q_val, p)

    best_coeffs = None
    best_degree = None
    best_error = np.inf

    for degree in range(4, max_degree + 1, 4):
        coeffs = chebfit(x_nodes, z_nodes, degree)
        z_hat = chebval(x_val, coeffs)
        err = float(np.max(np.abs(z_hat - z_val)))

        if verbose:
            print(f"p={p:.8g}, degree={degree:3d}, max error={err:.3e}")

        if err < best_error:
            best_coeffs = coeffs
            best_degree = degree
            best_error = err

        if err <= target_error:
            return ChebyshevApprox(
                p=p,
                coeffs=coeffs,
                degree=degree,
                estimated_error=err,
                target_error=target_error,
                n_fit=n_fit,
                n_val=n_val,
            )

    warnings.warn(
        f"Target error {target_error:.3e} was not reached for p={p}. "
        f"Using degree {best_degree} with estimated error {best_error:.3e}."
    )

    return ChebyshevApprox(
        p=p,
        coeffs=best_coeffs,
        degree=best_degree,
        estimated_error=float(best_error),
        target_error=target_error,
        n_fit=n_fit,
        n_val=n_val,
    )


def eval_cheb_on_q(q, coeffs) -> np.ndarray:
    """
    Evaluate the Chebyshev approximation on q in [0, 1].

    Parameters
    ----------
    q:
        Scalar or array-like values in [0, 1].
    coeffs:
        Chebyshev coefficients for T_k(2q - 1).
    """

    q_arr = np.asarray(q, dtype=np.float64)
    x = 2.0 * q_arr - 1.0
    return chebval(x, coeffs)

import json
from pathlib import Path
from typing import Any


def cheb_approx_to_dict(approx: ChebyshevApprox, dtype: str = "float32") -> dict[str, Any]:
    """
    Convert a ChebyshevApprox object to a JSON-serializable dictionary.
    """

    const = lp_constants(approx.p, lam=1.0)

    return {
        "p": float(approx.p),
        "dtype": dtype,
        "degree": int(approx.degree),
        "target_error": float(approx.target_error),
        "estimated_error": float(approx.estimated_error),
        "n_fit": int(approx.n_fit),
        "n_val": int(approx.n_val),
        "eps_tau": float(const.eps_tau),
        "z_tau": float(const.z_tau),
        "coeff_basis": "chebyshev_T_on_[-1,1]",
        "q_domain": [0.0, 1.0],
        "x_transform": "x = 2*q - 1",
        "coefficients": [float(c) for c in approx.coeffs],
    }


def save_cheb_approx(
    approx: ChebyshevApprox,
    path: str | Path,
    dtype: str = "float32",
) -> None:
    """
    Save Chebyshev approximation metadata and coefficients to JSON.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = cheb_approx_to_dict(approx, dtype=dtype)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_cheb_approx(path: str | Path) -> ChebyshevApprox:
    """
    Load Chebyshev approximation from JSON.
    """

    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    coeffs = np.asarray(data["coefficients"], dtype=np.float64)

    return ChebyshevApprox(
        p=float(data["p"]),
        coeffs=coeffs,
        degree=int(data["degree"]),
        estimated_error=float(data["estimated_error"]),
        target_error=float(data["target_error"]),
        n_fit=int(data["n_fit"]),
        n_val=int(data["n_val"]),
    )

from fractions import Fraction
from importlib import resources


def dtype_to_name(dtype) -> str:
    """
    Convert torch/numpy/string dtype-like object to a coefficient table dtype name.
    """

    text = str(dtype).lower()

    if "float32" in text or "fp32" in text:
        return "float32"

    if "float64" in text or "double" in text or "fp64" in text:
        return "float64"

    # Default for the current project.
    return "float32"


def p_to_rational_token(p: float, max_denominator: int = 1000) -> str:
    """
    Convert p to a rational filename token.

    Examples:
        1/3 -> p_1_3
        2/5 -> p_2_5
        3/4 -> p_3_4
        0.5 -> p_1_2
    """

    frac = Fraction(float(p)).limit_denominator(max_denominator)
    return f"p_{frac.numerator}_{frac.denominator}"


def candidate_coeff_table_names(p: float, dtype: str = "float32") -> list[str]:
    """
    Candidate built-in coefficient table filenames for a given p and dtype.
    """

    dtype = dtype_to_name(dtype)

    rational_token = p_to_rational_token(p)

    # Also support decimal-style names generated during early development,
    # e.g. p_0_5_float32.json.
    decimal_token = f"p_{str(float(p)).replace('.', '_')}".replace("-", "minus_")

    names = [
        f"{rational_token}_{dtype}.json",
        f"{decimal_token}_{dtype}.json",
    ]

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(names))


def load_builtin_cheb_approx(
    p: float,
    dtype: str = "float32",
) -> ChebyshevApprox:
    """
    Load a coefficient table bundled inside fast_lp_prox/coeff_tables.

    Raises FileNotFoundError if no matching table exists.
    """

    dtype = dtype_to_name(dtype)
    names = candidate_coeff_table_names(p, dtype=dtype)

    table_dir = resources.files("fast_lp_prox").joinpath("coeff_tables")

    tried = []

    for name in names:
        path = table_dir.joinpath(name)
        tried.append(str(path))

        if path.is_file():
            return load_cheb_approx(path)

    raise FileNotFoundError(
        "No built-in coefficient table found for "
        f"p={p}, dtype={dtype}. Tried: {tried}"
    )


def get_or_build_cheb_approx(
    p: float,
    dtype: str = "float32",
    use_table: bool = True,
    target_error: float = 2e-7,
    max_degree: int = 96,
    n_fit: int = 8192,
    n_val: int = 80001,
    verbose: bool = False,
) -> ChebyshevApprox:
    """
    Load a built-in Chebyshev table if available; otherwise build one.

    This is the preferred entry point for production code.
    """

    dtype = dtype_to_name(dtype)

    if use_table:
        try:
            approx = load_builtin_cheb_approx(p=p, dtype=dtype)

            if approx.estimated_error <= target_error:
                return approx

            if verbose:
                print(
                    "Built-in coefficient table found, but estimated_error "
                    f"{approx.estimated_error:.3e} exceeds target_error "
                    f"{target_error:.3e}. Rebuilding."
                )

        except FileNotFoundError:
            if verbose:
                print(f"No built-in coefficient table found for p={p}, dtype={dtype}. Rebuilding.")

    return build_cheb_coeffs(
        p=p,
        target_error=target_error,
        max_degree=max_degree,
        n_fit=n_fit,
        n_val=n_val,
        verbose=verbose,
    )