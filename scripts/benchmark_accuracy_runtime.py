#!/usr/bin/env python3
"""Phase-1: accuracy-vs-runtime sweeps using identical input tensors.

Run from the installed/editable fast-lp-prox repository root, for example:
    python scripts/benchmark_accuracy_runtime.py --device cpu --dtype float64 \
        --p 0.3333333333333333 --n 10000 --repeats 5 \
        --output results/revision_phase1_cpu.csv
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from fast_lp_prox import lp_constants, make_lp_prox
from fast_lp_prox.benchmark_utils import get_device, make_test_vector, time_function
from fast_lp_prox.revision_bench import (
    fit_degree_family,
    make_preloaded_prox,
    measure_accuracy,
    parse_positive_int_grid,
    prox_bisection_reference_numpy,
)


def _p_value(raw: str) -> float:
    try:
        return float(Fraction(raw))
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError("p must be a decimal or fraction, e.g. 1/3") from exc


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
            text=True, timeout=3,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Accuracy-matched ell_p proximal benchmark (revision Phase 1)")
    parser.add_argument("--p", type=_p_value, default=1.0 / 3.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--n", type=int, default=10000, help="Elements used for timing")
    parser.add_argument("--accuracy-n", type=int, default=0,
                        help="0 checks all timed elements; positive value checks a common subset")
    parser.add_argument("--active-ratio", type=float, default=0.9)
    parser.add_argument("--ymax-factor", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cheb-degrees", default="4,8,12,16")
    parser.add_argument("--newton-iters-grid", default="2,4,6,8")
    parser.add_argument("--bisect-iters-grid", default="8,16,24")
    parser.add_argument("--n-fit", type=int, default=8192)
    parser.add_argument("--n-val", type=int, default=4001)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ref-bisect-iters", type=int, default=75)
    parser.add_argument("--ref-newton-iters", type=int, default=50)
    parser.add_argument("--ref-check-n", type=int, default=4096)
    parser.add_argument("--ref-check-tol", type=float, default=2e-12)
    parser.add_argument("--target-norm-error", type=float, default=1e-6)
    parser.add_argument("--output", type=str, default="results/revision_phase1.csv")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output CSV")
    return parser


def run_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    if args.n < 1 or args.ref_check_n < 1 or args.ref_bisect_iters < 2:
        raise ValueError("n, ref-check-n must be positive; ref-bisect-iters >=2")
    if not 0.0 <= args.active_ratio <= 1.0 or args.ymax_factor <= 1.0:
        raise ValueError("Require active ratio in [0,1] and ymax-factor>1")
    if args.accuracy_n < 0 or args.accuracy_n > args.n:
        raise ValueError("accuracy-n must be in [0,n]")
    if args.ref_check_tol <= 0 or args.target_norm_error <= 0:
        raise ValueError("reference check and accuracy target tolerances must be positive")
    degrees = parse_positive_int_grid(args.cheb_degrees, "cheb-degrees")
    newton_grid = parse_positive_int_grid(args.newton_iters_grid, "newton-iters-grid")
    bisect_grid = parse_positive_int_grid(args.bisect_iters_grid, "bisect-iters-grid")
    if args.n_fit <= max(degrees):
        raise ValueError("n-fit must exceed the largest Chebyshev degree")
    if args.n_val < 3:
        raise ValueError("n-val must be at least 3")
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats>0 and warmup>=0 required")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; choose a different name or --overwrite")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = get_device(args.device)
    const = lp_constants(args.p, args.lam)
    print(f"Benchmark: p={args.p:g}, lam={args.lam:g}, {device}/{dtype}, n={args.n}")
    y = make_test_vector(
        n=args.n, tau=const.tau, active_ratio=args.active_ratio,
        ymax_factor=args.ymax_factor, dtype=dtype, device=device, seed=args.seed,
    )
    # Evaluate exactly the same inputs across all candidate methods. When
    # accuracy-n=0 these are also exactly all inputs used in the timings.
    accuracy_n = args.n if args.accuracy_n == 0 else args.accuracy_n
    if accuracy_n == args.n:
        y_acc = y
    else:
        idx = torch.linspace(0, args.n - 1, accuracy_n, device=device).long()
        y_acc = y[idx]
    y_cpu = y_acc.detach().cpu().numpy().astype(np.float64, copy=False)
    t_ref = time.perf_counter()
    x_ref = prox_bisection_reference_numpy(y_cpu, args.p, args.lam,
                                           iters=args.ref_bisect_iters)
    reference_seconds = time.perf_counter() - t_ref

    # Independent reference check against existing float64 Newton on a shared
    # subset; this check is not part of timed candidate evaluations.
    n_check = min(args.ref_check_n, accuracy_n)
    check_indices = np.linspace(0, accuracy_n - 1, n_check, dtype=np.int64)
    check_input = torch.from_numpy(np.array(y_cpu[check_indices], copy=True))
    ref_newton = make_lp_prox(
        p=args.p, lam=args.lam, method="newton12", dtype=torch.float64,
        newton_iters=args.ref_newton_iters,
    )
    with torch.inference_mode():
        x_ncheck = ref_newton(check_input).detach().cpu().numpy()
    discrepancy = np.abs(x_ncheck - x_ref[check_indices]) / np.maximum(
        np.abs(y_cpu[check_indices]), const.tau,
    )
    max_reference_disagreement = float(np.max(discrepancy))
    if not np.isfinite(max_reference_disagreement) or max_reference_disagreement > args.ref_check_tol:
        raise RuntimeError(
            "Independent reference check FAILED: normalized Newton-vs-bisection "
            f"difference={max_reference_disagreement:.3e} > {args.ref_check_tol:.3e}"
        )
    print("Independent reference check passed; max normalized discrepancy = "
          f"{max_reference_disagreement:.3e} ({n_check} elements)")

    t_fit = time.perf_counter()
    approx = fit_degree_family(args.p, degrees, n_fit=args.n_fit, n_validation=args.n_val)
    offline_seconds = time.perf_counter() - t_fit
    print(f"Offline Chebyshev coefficient family built in {offline_seconds:.3f} s")
    print("Fitted degree errors (sampled float64): " + ", ".join(
        f"D={d}: {approx.sampled_errors[d]:.2e}" for d in degrees
    ))

    runtime_candidates = []
    for d in degrees:
        runtime_candidates.append(("chebyshev", d, approx.coefficients[:d + 1]))
    for k in newton_grid:
        runtime_candidates.append(("newton", k, None))
    for k in bisect_grid:
        runtime_candidates.append(("bisection", k, None))
    rows = []
    try:
        device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    except Exception:
        device_name = str(device)
    for family, work, coeff in runtime_candidates:
        # Prepare all candidate-specific constants and buffers before timing.
        fn = make_preloaded_prox(
            p=args.p, lam=args.lam, family=family, device=device,
            dtype=dtype, work=work, coefficients=coeff,
        )
        with torch.inference_mode():
            stats, _ = time_function(fn, y, warmup=args.warmup, repeats=args.repeats)
            candidate = fn(y_acc)
        out = candidate.detach().cpu().numpy().astype(np.float64, copy=False)
        metrics = measure_accuracy(y_cpu, out, x_ref, p=args.p, lam=args.lam)
        row = {
            "family": family,
            "configuration": f"{family}_D{work}" if family == "chebyshev" else f"{family}_K{work}",
            "degree": work if family == "chebyshev" else "",
            "iterations": work if family != "chebyshev" else "",
            "p": args.p,
            "lambda": args.lam,
            "dtype": args.dtype,
            "device": str(device),
            "device_name": device_name,
            "n_timing": args.n,
            "n_accuracy": accuracy_n,
            "accuracy_same_inputs_as_timing": accuracy_n == args.n,
            "active_ratio_requested": args.active_ratio,
            "active_ratio_realized": float(np.mean(np.abs(y_cpu) > const.tau)),
            "ymax_factor": args.ymax_factor,
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "median_ms": stats["median_ms"],
            "mean_ms": stats["mean_ms"],
            "std_ms": stats["std_ms"],
            "throughput_million_per_second": args.n / (1000.0 * stats["median_ms"]),
            "fit_n": args.n_fit if family == "chebyshev" else "",
            "validation_n": args.n_val if family == "chebyshev" else "",
            "sampled_float64_branch_error": approx.sampled_errors[work] if family == "chebyshev" else "",
            "offline_family_seconds": offline_seconds if family == "chebyshev" else "",
            "reference_seconds": reference_seconds,
            "reference_method": "float64_bisection_75_steps".replace("75", str(args.ref_bisect_iters)),
            "reference_newton_crosscheck_n": n_check,
            "max_reference_crosscheck_norm_error": max_reference_disagreement,
            "accuracy_target": args.target_norm_error,
            "meets_accuracy_target": bool(metrics["all_finite"] and
                                          metrics["max_norm_error"] <= args.target_norm_error),
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
        }
        row.update(metrics)
        rows.append(row)
        print(f"{row['configuration']:>15}: {stats['median_ms']:.4f} ms "
              f"norm_err_max={metrics['max_norm_error']:.3e} "
              f"pass={row['meets_accuracy_target']}")
    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} configurations to {output}")
    return df


def main() -> None:
    run_benchmark(build_parser().parse_args())


if __name__ == "__main__":
    main()
