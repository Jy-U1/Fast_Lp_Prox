#!/usr/bin/env python3
"""Phase 2A: numerical robustness at the nonconvex ell_p prox threshold.

Public experiment only; uses the existing repository's *unmodified* phase-1
online evaluators, and an independent arbitrary-precision scalar oracle.
No timing comparison is made in this script.

From repository root, after ``pip install -e '.[dev]' mpmath``:
    python revision/phase2/scripts/run_threshold_robustness.py \
      --device cpu --p-grid 1/3,0.9 --lambda-grid 1,1e6 \
      --output-dir results/phase2/threshold_smoke
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from collections import OrderedDict
from typing import Any

import mpmath as mp
import numpy as np
import torch

from fast_lp_prox.revision_bench import fit_degree_family, make_preloaded_prox
from fast_lp_prox.constants import lp_constants


# Configurations selected from the *tested discrete grids* of Phase 1.
# The second Chebyshev degree is deliberately more conservative; it is NOT
# a certified whole-domain error bound.
CONFIGS = [
    (1.0 / 3.0, "1/3", 6, 8, 3),
    (2.0 / 5.0, "2/5", 6, 8, 3),
    (3.0 / 4.0, "3/4", 12, 16, 4),
    (0.9, "0.9", 24, 28, 4),
    (0.99, "0.99", 54, 60, 5),
    (0.999, "0.999", 116, 128, 4),
]


@dataclass(frozen=True)
class Threshold:
    rho: mp.mpf
    tau: mp.mpf
    p: mp.mpf
    lam: mp.mpf


def threshold_high_precision(p: float, lam: float, dps: int = 80) -> Threshold:
    """Threshold for the *exact binary64* p and lambda passed by Python.

    Keeping those inputs exact matters within a float64 ULP. In particular,
    interpreting a Python float as its intended decimal spelling can change
    the comparison with nearest representable inputs.
    """
    if not (0.0 < p < 1.0 and lam > 0 and math.isfinite(p) and math.isfinite(lam)):
        raise ValueError("Require 0 < p < 1 and finite lambda > 0")
    with mp.workdps(dps):
        pm, lm = mp.mpf(p), mp.mpf(lam)
        rho = mp.power(2 * lm * (1 - pm), 1 / (2 - pm))
        tau = rho * (2 - pm) / (2 * (1 - pm))
        return Threshold(+rho, +tau, +pm, +lm)


def high_precision_positive_prox(a: float, t: Threshold, *, iters: int = 155) -> tuple[str, mp.mpf]:
    """Independent high-precision reference at a representable |y|>=0.

    Tie convention: return zero at |y|=tau, where the true proximal map is
    set-valued {0, rho}. Uses bracketed bisection, NOT candidate Newton/Cheb.
    """
    if not math.isfinite(a) or a < 0:
        raise ValueError("Reference inputs must be finite and nonnegative")
    am = mp.mpf(float(a))
    if am < t.tau:
        return "below", mp.mpf(0)
    if am == t.tau:
        return "tie", mp.mpf(0)
    lo, hi = t.rho, am
    if hi <= lo:
        raise AssertionError("Above-threshold y must exceed rho")
    for _ in range(iters):
        mid = (lo + hi) / 2
        h = mid + t.lam * t.p * mp.power(mid, t.p - 1) - am
        if h > 0:
            hi = mid
        else:
            lo = mid
    root = (hi + lo) / 2
    return "above", +root


def _parsed_p_grid(spec: str) -> list[tuple[float, str, int, int, int]]:
    tokens = [x.strip() for x in spec.split(",") if x.strip()]
    result = []
    for token in tokens:
        value = float(Fraction(token)) if "/" in token else float(token)
        found = next((c for c in CONFIGS if abs(c[0] - value) < 1e-15), None)
        if found is None:
            raise ValueError(f"Unsupported p={token}; supported: {', '.join(c[1] for c in CONFIGS)}")
        if found not in result:
            result.append(found)
    if not result:
        raise ValueError("p-grid cannot be empty")
    return result


def _parsed_lam_grid(spec: str) -> list[float]:
    lams = [float(s.strip()) for s in spec.split(",") if s.strip()]
    if not lams or len(set(lams)) != len(lams) or any(x <= 0 or not math.isfinite(x) for x in lams):
        raise ValueError("lambda-grid needs distinct positive finite values")
    return lams


def _dtype(label: str) -> np.dtype:
    return np.dtype(label)


def _as_typed(value: mp.mpf, dtype: np.dtype) -> np.generic:
    # The decimal mp string is parsed directly to the target float dtype.
    return dtype.type(mp.nstr(value, 85))


def _step_ulps(x: np.generic, count: int, dtype: np.dtype) -> np.generic:
    if count == 0:
        return x
    dst = dtype.type(np.inf if count > 0 else -np.inf)
    for _ in range(abs(count)):
        x = np.nextafter(x, dst, dtype=dtype)
    return x


def generate_threshold_points(
    t: Threshold,
    dtype: np.dtype,
    ulps: tuple[int, ...] = (1, 2, 4, 8, 16, 64),
    rel_shifts: tuple[str, ...] = ("1e-4", "1e-6", "1e-8", "1e-10", "1e-12", "1e-14"),
    far_factors: tuple[str, ...] = ("0", "0.5", "0.99", "1.01", "2", "10", "100"),
) -> list[dict[str, Any]]:
    """Unique representable signed inputs, recording all generation labels."""
    center = _as_typed(t.tau, dtype)
    values: OrderedDict[bytes, dict[str, Any]] = OrderedDict()

    def insert(v: np.generic, origin: str) -> None:
        if not np.isfinite(v):
            raise OverflowError(f"A generated input overflowed {dtype}")
        v = dtype.type(v)
        key = np.asarray([v], dtype=dtype).tobytes()
        if key in values:
            values[key]["origins"].append(origin)
        else:
            values[key] = {"y": float(v), "origins": [origin]}

    insert(center, "center+signed")
    insert(dtype.type(-center), "center-signed")
    for n in ulps:
        for d in (-n, n):
            a = _step_ulps(center, d, dtype)
            insert(a, f"ulp{d:+d}:+")
            insert(dtype.type(-a), f"ulp{d:+d}:-")
    for shift in rel_shifts:
        for sign in (-1, 1):
            with mp.workdps(mp.mp.dps):
                a = _as_typed(t.tau * (1 + sign * mp.mpf(shift)), dtype)
            insert(a, f"rel{sign:+d}{shift}:+")
            insert(dtype.type(-a), f"rel{sign:+d}{shift}:-")
    for factor in far_factors:
        with mp.workdps(mp.mp.dps):
            a = _as_typed(t.tau * mp.mpf(factor), dtype)
        insert(a, f"factor{factor}:+")
        if a != 0:
            insert(dtype.type(-a), f"factor{factor}:-")
    return [{**v, "origins": ";".join(v["origins"])} for v in values.values()]


def _mpf_from_output(y: float) -> mp.mpf:
    return mp.mpf(float(y))


def _objective(x: mp.mpf, a: mp.mpf, t: Threshold) -> mp.mpf:
    return (x - a) ** 2 / 2 + t.lam * mp.power(abs(x), t.p)


def _metric_record(y: float, xhat: float, side: str, xref: mp.mpf, t: Threshold,
                   dtype: np.dtype, center: float, repo_tau: float) -> dict[str, Any]:
    a = _mpf_from_output(abs(y))
    xmag = _mpf_from_output(abs(xhat)) if math.isfinite(xhat) else mp.nan
    sign = (-1 if y < 0 else 1)
    ref = sign * xref
    is_ref_active = side == "above"
    is_out_active = xhat != 0 and math.isfinite(xhat)
    # Diagnosis only: predict the branch selected if the library threshold is
    # rounded once into the input tensor dtype. Actual output is still tested.
    device_tau = float(dtype.type(repo_tau))
    quantized_branch = abs(y) > device_tau
    gap = (a - t.tau) / t.tau
    # Report signed threshold distances in units of one spacing at center.
    center_t = dtype.type(center)
    next_t = np.nextafter(center_t, dtype.type(np.inf), dtype=dtype)
    spacing = float(next_t) - float(center_t)
    ref_norm = max(a, t.tau)
    result: dict[str, Any] = {
        "y": y,
        "abs_y": float(a),
        "threshold_gap_relative": float(gap),
        "threshold_gap_center_ulp": float((a - t.tau) / mp.mpf(spacing)),
        "oracle_side": side,
        "oracle_support": is_ref_active,
        "oracle_x": float(ref),
        "x_output": xhat,
        "output_support": is_out_active,
        "support_mismatch": bool(is_ref_active != is_out_active),
        "quantized_tau": device_tau,
        "quantized_threshold_support": bool(quantized_branch),
        "threshold_quantization_disagrees_with_oracle": bool(quantized_branch != is_ref_active),
        "output_agrees_with_quantized_threshold": bool(is_out_active == quantized_branch),
        "all_finite": math.isfinite(xhat),
        "abs_output_error": float(abs(_mpf_from_output(xhat) - ref)) if math.isfinite(xhat) else math.inf,
        "norm_output_error": float(abs(_mpf_from_output(xhat) - ref) / ref_norm) if math.isfinite(xhat) else math.inf,
    }
    if not math.isfinite(xhat):
        result.update({"stationarity_residual": math.nan,
                       "norm_stationarity_residual": math.nan,
                       "objective_regret": math.nan, "norm_objective_regret": math.nan})
        return result
    if is_out_active:
        r = abs(xmag + t.lam * t.p * mp.power(xmag, t.p - 1) - a)
        result["stationarity_residual"] = float(r)
        result["norm_stationarity_residual"] = float(r / ref_norm)
    else:
        # Stationarity of the positive branch is UNDEFINED at zero.
        result["stationarity_residual"] = math.nan
        result["norm_stationarity_residual"] = math.nan
    objective_opt = _objective(xref, a, t)
    objective_out = _objective(xmag, a, t)
    regret = objective_out - objective_opt
    # Arbitrary precision oracle has finite residual due to root bisection;
    # a slightly negative value of order 1e-40 is roundoff, NOT an improvement.
    result["objective_regret"] = float(regret)
    result["norm_objective_regret"] = float(regret / (ref_norm ** 2))
    return result


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True, timeout=3).strip()
    except Exception:
        return "unavailable"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase 2A: near-threshold precision and scale robustness")
    ap.add_argument("--p-grid", default="1/3,2/5,3/4,0.9,0.99,0.999")
    ap.add_argument("--lambda-grid", default="1e-6,1,1e6")
    ap.add_argument("--dtypes", default="float32,float64")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    ap.add_argument("--fit-n", type=int, default=8192)
    ap.add_argument("--validation-n", type=int, default=80001)
    ap.add_argument("--mp-dps", type=int, default=80)
    ap.add_argument("--ref-iterations", type=int, default=155)
    ap.add_argument("--output-dir", default="results/phase2/threshold")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def run(args: argparse.Namespace) -> dict[str, Any]:
    grid = _parsed_p_grid(args.p_grid)
    lams = _parsed_lam_grid(args.lambda_grid)
    dtypes = [s.strip() for s in args.dtypes.split(",") if s.strip()]
    if len(set(dtypes)) != len(dtypes) or not dtypes or not set(dtypes).issubset({"float32", "float64"}):
        raise ValueError("dtypes must be distinct elements of float32,float64")
    if args.mp_dps < 55 or args.ref_iterations < 120:
        raise ValueError("Need mp-dps>=55 and ref-iterations>=120 for near-ULP checks")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    outdir = Path(args.output_dir)
    detail_path = outdir / "threshold_detail.csv"
    if detail_path.exists() and not args.overwrite:
        raise FileExistsError(f"{detail_path} exists; use a new path or --overwrite")
    outdir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "task": "Phase 2A near-threshold and scale numerical stability",
        "not_a_timing_benchmark": True,
        "git_commit": _get_git_commit(), "device_requested": args.device,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "p_grid": [c[1] for c in grid], "lambda_grid": lams,
        "dtypes": dtypes, "fit_n": args.fit_n,
        "validation_n": args.validation_n, "mp_dps": args.mp_dps,
        "ref_iterations": args.ref_iterations,
        "python": platform.python_version(),
        "numpy": np.__version__, "torch": torch.__version__,
        "mpmath": mp.__version__, "tie_convention": "zero at exact mathematical threshold",
        "input_parameter_interpretation": "Python binary64 p and lambda interpreted exactly by mp oracle",
        "float32_threshold_comparison": "compare actual candidate outputs to high-precision oracle; no correction applied",
    }
    rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    with mp.workdps(args.mp_dps):
        for p, plabel, cheb_fast, cheb_safe, newton_steps in grid:
            degree_grid = [cheb_fast, cheb_safe]
            fit = fit_degree_family(p, degree_grid, n_fit=args.fit_n, n_validation=args.validation_n)
            print(f"Fitted p={plabel} D={degree_grid}, float64 sampled errors: " +
                  ", ".join(f"D{d}={fit.sampled_errors[d]:.3e}" for d in degree_grid), flush=True)
            for lam in lams:
                t = threshold_high_precision(p, lam, dps=args.mp_dps)
                legacy_tau = lp_constants(p, lam).tau
                for dtype_name in dtypes:
                    npdtype = _dtype(dtype_name)
                    torchdtype = torch.float32 if dtype_name == "float32" else torch.float64
                    pts = generate_threshold_points(t, npdtype)
                    y = np.array([a["y"] for a in pts], dtype=npdtype)
                    center = float(_as_typed(t.tau, npdtype))
                    # Exact float-to-mp conversion is used in oracle comparisons.
                    mag_ref: dict[float, tuple[str, mp.mpf]] = {}
                    for a in np.unique(np.abs(y).astype(np.float64)):
                        mag_ref[float(a)] = high_precision_positive_prox(float(a), t, iters=args.ref_iterations)
                    threshold_rows.append({
                        "p": p, "p_label": plabel, "lambda": lam,
                        "dtype": dtype_name, "tau_hp": mp.nstr(t.tau, 45),
                        "rho_hp": mp.nstr(t.rho, 45),
                        "tau_repo_float64": legacy_tau,
                        "tau_repo_rounded_dtype": float(npdtype.type(legacy_tau)),
                        "tau_repo_relative_error": float((mp.mpf(float(legacy_tau)) - t.tau) / t.tau),
                        "tau_nearest_dtype": center,
                        "center_side": mag_ref[center][0],
                        "input_count": len(pts),
                        "machine_epsilon": float(np.finfo(npdtype).eps),
                    })
                    case_id = f"p{plabel.replace('/', '_')}_lam{lam:g}_{dtype_name}"
                    configurations = [
                        ("chebyshev", cheb_fast, f"chebyshev_D{cheb_fast}", fit.coefficients[:cheb_fast+1]),
                        ("chebyshev", cheb_safe, f"chebyshev_D{cheb_safe}", fit.coefficients[:cheb_safe+1]),
                        ("newton", newton_steps, f"newton_K{newton_steps}", None),
                        ("bisection", 20, "bisection_K20", None),
                    ]
                    y_torch = torch.as_tensor(y.copy(), dtype=torchdtype, device=device)
                    for family, work, label, coefficients in configurations:
                        fn = make_preloaded_prox(p=p, lam=lam, family=family,
                                                 device=device, dtype=torchdtype,
                                                 work=work, coefficients=coefficients)
                        with torch.inference_mode():
                            xout = fn(y_torch).detach().cpu().numpy().astype(np.float64)
                        for src, yval, outval in zip(pts, y.astype(np.float64), xout):
                            a = abs(float(yval))
                            side, xref = mag_ref[a]
                            row = {
                                "case_id": case_id, "p": p, "p_label": plabel,
                                "lambda": lam, "dtype": dtype_name, "device": str(device),
                                "family": family, "configuration": label,
                                "work": work,
                                "origin_tags": src["origins"],
                                "is_near_threshold": bool(a > 0 and abs(float((mp.mpf(a)-t.tau)/t.tau)) <= 1e-3),
                            }
                            row.update(_metric_record(float(yval), float(outval), side, xref,
                                                      t, npdtype, center, legacy_tau))
                            rows.append(row)
                    print(f" {case_id}: {len(pts)} inputs, {len(pts)*len(configurations)} records", flush=True)
    if not rows:
        raise RuntimeError("No records generated")
    # Write candidate results after all numerical steps complete.
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (outdir / "threshold_parameters.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)
    metadata["detail_rows"] = len(rows)
    metadata["parameter_rows"] = len(threshold_rows)
    metadata["support_mismatch_count"] = int(sum(x["support_mismatch"] for x in rows))
    metadata["detail_sha256"] = hashlib.sha256(detail_path.read_bytes()).hexdigest()
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} candidate records to {detail_path}", flush=True)
    print(f"Support mismatches (not suppressed): {metadata['support_mismatch_count']}", flush=True)
    return metadata


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
