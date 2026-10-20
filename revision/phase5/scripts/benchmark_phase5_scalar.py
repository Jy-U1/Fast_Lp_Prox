#!/usr/bin/env python3
"""Run a new same-input accuracy/runtime study (not archived Phase-1 timings).

Two added backends are evaluated alongside the EXISTING Phase-3 Chebyshev,
projected Newton and bisection implementations when --existing-methods all.
The series backends are additions; no historical input distribution or
historical timing is silently reused. Raw results and provenance are new.

Example:
  python -m revision.phase5.scripts.benchmark_phase5_scalar \
    --device cpu --dtype float64 --p 0.3333333333333333 \
    --n 2000 --series-terms 8,16,32 --existing-methods none \
    --output-dir results/phase5_cpu_smoke
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from revision.phase5.backends import (
    FixedSeries, ResidualSeriesBisection, proximal_constants,
    reference_bisection, select_against_zero,
)


def git_identity() -> dict:
    def run(args):
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                           text=True).strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
    return {"branch": run(["branch", "--show-current"]),
            "commit": run(["rev-parse", "HEAD"]),
            "dirty_files": run(["status", "--porcelain"])}


def generate_shared_inputs(*, n: int, p: float, lam: float, active_fraction: float,
                           ymax_factor: float, seed: int,
                           dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Deterministic pooled active/inactive, threshold and wide-range samples.

    Not identical to the original Phase-1 generator. Each method called by
    THIS driver uses precisely this ONE common device tensor. The fractions,
    seed and generation rule are recorded in metadata.
    """
    if n < 32 or not 0 <= active_fraction <= 1 or ymax_factor <= 1:
        raise ValueError("invalid generator arguments")
    c = proximal_constants(p, lam)
    rng = np.random.default_rng(seed)
    num_active = int(round(n * active_fraction))
    num_inactive = n - num_active
    # Inactive samples include a small number immediately below threshold.
    inactive = c.tau * rng.uniform(0.0, 0.999, num_inactive)
    if num_inactive:
        k = min(num_inactive, max(1, num_inactive // 20))
        delta = np.logspace(-8, -3, k)
        inactive[:k] = c.tau * (1.0 - delta)
    if num_active:
        k = min(num_active, max(1, num_active // 20))
        magnitudes = c.tau * np.exp(rng.uniform(np.log1p(1e-3),
                                                math.log(ymax_factor), num_active))
        magnitudes[:k] = c.tau * (1.0 + np.logspace(-8, -3, k))
        active = magnitudes
    else:
        active = np.empty(0, dtype=np.float64)
    vals = np.concatenate((inactive, active))
    vals *= rng.choice([-1.0, 1.0], size=n)
    rng.shuffle(vals)
    return torch.tensor(vals, dtype=dtype, device=device)


def metrics(y: torch.Tensor, x: torch.Tensor, ref: torch.Tensor, *, p: float,
            lam: float) -> dict:
    if x.shape != y.shape or x.device != y.device or x.dtype != y.dtype:
        raise ValueError("backend must preserve tensor shape, dtype and device")
    c = proximal_constants(p, lam)
    yd = y.to(torch.float64)
    xd = x.to(torch.float64)
    rd = ref.to(torch.float64)
    diff = (xd - rd).abs()
    norm_err = diff / torch.clamp(yd.abs(), min=c.tau)
    active = xd != 0
    safe_x = xd.abs().clamp(min=torch.finfo(torch.float64).tiny)
    stat = torch.abs(safe_x - yd.abs() + lam * p * safe_x.pow(p - 1))
    stat = torch.where(active, stat, torch.zeros_like(stat))
    # Objective gap in cancellation-reduced form:
    # Phi(x;y)-Phi(0;y) = 1/2*x*x - y*x + lam*|x|^p.
    dx = 0.5 * xd.square() - yd * xd + lam * xd.abs().pow(p)
    dr = 0.5 * rd.square() - yd * rd + lam * rd.abs().pow(p)
    signed_gap = dx - dr
    finite = bool(torch.isfinite(xd).all())
    return {
        "all_finite": finite,
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "max_norm_error": float(norm_err.max().item()),
        "mean_norm_error": float(norm_err.mean().item()),
        "max_active_stationarity_residual": float(stat.max().item()),
        "mean_prox_objective_gap": float(signed_gap.clamp(min=0).mean().item()),
        "min_signed_objective_gap": float(signed_gap.min().item()),
        "max_prox_objective_gap": float(signed_gap.max().item()),
        "support_mismatch_count": int(((xd != 0) != (rd != 0)).sum().item()),
        "evaluated_active_count": int(active.sum().item()),
    }


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_backend(func, y: torch.Tensor, *, warmup: int, repeats: int) -> dict:
    if warmup < 0 or repeats < 1:
        raise ValueError("invalid warmup/repeats")
    with torch.no_grad():
        for _ in range(warmup):
            func(y)
        cuda_sync(y.device)
        timings = []
        for _ in range(repeats):
            cuda_sync(y.device)
            t0 = time.perf_counter()
            _ = func(y)
            cuda_sync(y.device)
            timings.append(1e3 * (time.perf_counter() - t0))
    return {"median_ms": float(np.median(timings)),
            "mean_ms": float(np.mean(timings)),
            "std_ms": float(np.std(timings)),
            "timing_repeats_raw_ms": timings}


def _parse_positive_grid(text: str) -> tuple[int, ...]:
    vals = tuple(int(k.strip()) for k in text.split(",") if k.strip())
    if not vals or min(vals) < 1 or len(vals) != len(set(vals)):
        raise ValueError("invalid positive integer grid")
    return vals


def existing_backends(p: float, lam: float, *, device: torch.device,
                      dtype: torch.dtype, cheb: tuple[int, ...],
                      newton: tuple[int, ...], bisect: tuple[int, ...],
                      n_fit: int, n_validation: int) -> list[tuple[str, str, object, float]]:
    """Reuse exact Phase-3 implementation without changing it or silently swapping a baseline."""
    try:
        from fast_lp_prox.revision_bench import fit_degree_family, make_preloaded_prox
    except ImportError as exc:
        raise RuntimeError("Existing Phase-3 package is not installed. From repo root run "
                           "pip install -e '.[dev]', or use --existing-methods none") from exc
    t0 = time.perf_counter()
    fitted = fit_degree_family(p, sorted(set(cheb)), n_fit=n_fit,
                               n_validation=n_validation) if cheb else None
    fit_seconds = time.perf_counter() - t0
    rows = []
    for degree in cheb:
        func = make_preloaded_prox(p=p, lam=lam, family="chebyshev", work=degree,
                                   coefficients=fitted.coefficients[:degree + 1],
                                   dtype=dtype, device=device)
        rows.append(("chebyshev", f"chebyshev_D{degree}", func, fit_seconds))
    for count in newton:
        func = make_preloaded_prox(p=p, lam=lam, family="newton", work=count,
                                   coefficients=None, dtype=dtype, device=device)
        rows.append(("newton", f"newton_K{count}", func, 0.0))
    for count in bisect:
        func = make_preloaded_prox(p=p, lam=lam, family="bisection", work=count,
                                   coefficients=None, dtype=dtype, device=device)
        rows.append(("bisection", f"bisection_K{count}", func, 0.0))
    return rows


def build_backends(args: argparse.Namespace, device: torch.device,
                   dtype: torch.dtype):
    funcs = []
    for nterms in _parse_positive_grid(args.series_terms):
        t0 = time.perf_counter()
        func = FixedSeries(p=args.p, lam=args.lam, terms=nterms)
        offline = time.perf_counter() - t0
        funcs.append(("fixed_series", f"series_N{nterms}", func, offline))
    orders = tuple(sorted(_parse_positive_grid(args.hybrid_orders)))
    t0 = time.perf_counter()
    hybrid = ResidualSeriesBisection(p=args.p, lam=args.lam, orders=orders,
                                     tol=args.hybrid_tol,
                                     bisect_steps=args.hybrid_bisect_steps)
    offline = time.perf_counter() - t0
    funcs.append(("residual_series_bisect", "residual_series_bisect", hybrid, offline))
    if args.existing_methods == "all":
        funcs.extend(existing_backends(args.p, args.lam, device=device, dtype=dtype,
                                       cheb=_parse_positive_grid(args.cheb_degrees),
                                       newton=_parse_positive_grid(args.newton_iters),
                                       bisect=_parse_positive_grid(args.bisect_iters),
                                       n_fit=args.fit_n, n_validation=args.validation_n))
    if args.guard_chebyshev:
        for family, name, backend, offline in funcs[:]:
            if family == "chebyshev":
                def guarded(y, inner=backend):
                    return select_against_zero(y, inner(y), p=args.p, lam=args.lam)
                funcs.append(("chebyshev_guarded", name + "_objectiveguard", guarded, offline))
    return funcs


def run_experiment(args: argparse.Namespace) -> Path:
    if not (0 < args.p < 1 and args.lam > 0):
        raise ValueError("invalid p/lam")
    if args.guard_chebyshev and args.existing_methods != "all":
        raise ValueError("--guard-chebyshev requires --existing-methods all")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but unavailable")
    dev = torch.device(args.device)
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    dest = args.output_dir.resolve()
    if dest.exists():
        raise FileExistsError("Output directory exists; choose a NEW --output-dir: " + str(dest))
    y = generate_shared_inputs(n=args.n, p=args.p, lam=args.lam,
                               active_fraction=args.active_fraction,
                               ymax_factor=args.ymax_factor, seed=args.seed,
                               dtype=dtype, device=dev)
    ref = reference_bisection(y, p=args.p, lam=args.lam,
                              iterations=args.ref_steps)
    input_sha256 = hashlib.sha256(y.detach().cpu().numpy().tobytes()).hexdigest()
    funcs = build_backends(args, dev, dtype)
    result = []
    for family, name, backend, offline in funcs:
        with torch.no_grad():
            x = backend(y)
            met = metrics(y, x, ref, p=args.p, lam=args.lam)
            extra = {}
            if isinstance(backend, ResidualSeriesBisection):
                st = backend.last_stats
                extra = {"active_count": st.total_active,
                         "series_accepted_count": st.series_accepted,
                         "fallback_count": st.fallback_count,
                         "max_accepted_normalized_indicator": st.max_indicator_accepted}
            else:
                extra = {"active_count": met["evaluated_active_count"],
                         "series_accepted_count": "", "fallback_count": "",
                         "max_accepted_normalized_indicator": ""}
            timing = time_backend(backend, y, warmup=args.warmup,
                                  repeats=args.repeats)
        raw = timing.pop("timing_repeats_raw_ms")
        result.append({"p": args.p, "lam": args.lam, "family": family,
                       "configuration": name, "device": args.device,
                       "dtype": args.dtype, "n_accuracy": args.n,
                       "n_timing": args.n, "seed": args.seed,
                       "active_fraction_requested": args.active_fraction,
                       "ymax_factor": args.ymax_factor,
                       "accuracy_target": args.accuracy_target,
                       "meets_accuracy_target": bool(met["all_finite"] and met["max_norm_error"] <= args.accuracy_target),
                       "warmup": args.warmup, "repeats": args.repeats,
                       "offline_family_seconds": offline,
                       **met, **extra, **timing,
                       "throughput_million_per_second": args.n/timing["median_ms"]/1000,
                       "timing_repeats_raw_ms": json.dumps(raw)})
        print(f"{name:30} E_norm={met['max_norm_error']:.3g} "
              f"res={met['max_active_stationarity_residual']:.3g} "
              f"time={timing['median_ms']:.3f} ms", flush=True)
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_identity(),
            "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if dev.type == 'cuda' else 'cpu',
            "args": vars(args).copy(),
            "reference": f"monotone float64 bisection {args.ref_steps} steps, not a directed rounding certificate",
            "input_sha256": input_sha256,
            "fairness": "within this run: same exact input tensor for all methods; prefit/precompute excluded from timed calls",
            "input_generator": "new Phase-5 generator, log-uniform active + threshold stress, distinct from historical Phase-1",
            "hybrid_semantics": "posterior-residual-checked series with monotone bisection fallback; not the MB hybrid of arXiv:2606.31951 and not interval-certified",
            "guard_semantics": "scalar compare candidate vs zero only; cannot imply exact criticality or model descent",
            "accuracy_semantics": "sampled maximum over finite input tensor; all diagnostics empirical float64 reference",
            }
    meta['args']['output_dir'] = str(args.output_dir)
    dest.mkdir(parents=True, exist_ok=False)
    with (dest / "phase5_scalar_results.csv").open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    (dest / "phase5_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + '\n')
    print("Results saved:", dest)
    return dest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    p.add_argument('--p', type=float, default=1.0/3)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--active-fraction', type=float, default=0.9)
    p.add_argument('--ymax-factor', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--accuracy-target', type=float, default=1e-6)
    p.add_argument('--ref-steps', type=int, default=75)
    p.add_argument('--series-terms', type=str, default='8,16,32,64')
    p.add_argument('--hybrid-orders', type=str, default='8,32,128')
    p.add_argument('--hybrid-tol', type=float, default=1e-6)
    p.add_argument('--hybrid-bisect-steps', type=int, default=64)
    p.add_argument('--existing-methods', choices=('none', 'all'), default='none')
    p.add_argument('--cheb-degrees', type=str, default='6')
    p.add_argument('--newton-iters', type=str, default='3,4')
    p.add_argument('--bisect-iters', type=str, default='20')
    p.add_argument('--fit-n', type=int, default=8192)
    p.add_argument('--validation-n', type=int, default=80001)
    p.add_argument('--guard-chebyshev', action='store_true')
    p.add_argument('--output-dir', type=Path, default=Path('results/reviewer2_phase5_scalar_smoke'))
    args = p.parse_args(argv)
    run_experiment(args)


if __name__ == '__main__':
    main()
