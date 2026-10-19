#!/usr/bin/env python3
"""Phase 2B: repeated, randomized-block accuracy-matched eager PyTorch timings.

Run from the editable package repository root. Does not modify Phase 1 files.
CUDA event and synchronized host-wall time are both recorded PER INVOCATION.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from fast_lp_prox import lp_constants
from fast_lp_prox.benchmark_utils import make_test_vector
from fast_lp_prox.revision_bench import (
    fit_degree_family, make_preloaded_prox, measure_accuracy,
    prox_bisection_reference_numpy,
)

# Freeze Phase-1 selected competitor configurations. The p=3/4 extra D14
# is a precision-margin candidate, not the Phase-1 fastest D12.
CANDIDATE_GRID = {
    "1/3": (("chebyshev", 6), ("newton", 3)),
    "2/5": (("chebyshev", 6), ("newton", 3)),
    "3/4": (("chebyshev", 12), ("chebyshev", 14), ("newton", 4)),
}


def parse_positive_sizes(raw: str) -> list[int]:
    try:
        numbers = [int(x.strip()) for x in raw.split(",")]
    except (ValueError, AttributeError) as exc:
        raise ValueError("sizes must be comma-separated positive integers") from exc
    if not numbers or any(n <= 0 for n in numbers) or len(set(numbers)) != len(numbers):
        raise ValueError("sizes must be nonempty, positive and distinct")
    return sorted(numbers)


def parse_p_labels(raw: str) -> list[str]:
    labels = [s.strip() for s in raw.split(",")]
    if not labels or len(set(labels)) != len(labels) or any(s not in CANDIDATE_GRID for s in labels):
        raise ValueError("supported p labels: " + ", ".join(CANDIDATE_GRID))
    return labels


def configuration(family: str, work: int) -> str:
    return f"{family}_{'D' if family == 'chebyshev' else 'K'}{work}"


def randomized_blocks(labels: list[str], n_blocks: int, rng: random.Random) -> list[list[str]]:
    """Balanced, randomized blocks; each candidate occupies every position once/cycle.

    For 30 blocks, two candidates run 15 times each in positions 0 and 1;
    three candidates run 10 times each in positions 0,1,2.
    """
    if not labels or n_blocks < 1 or len(set(labels)) != len(labels):
        raise ValueError("nonempty distinct candidates and positive block count required")
    order = []
    count = len(labels)
    while len(order) < n_blocks:
        base = list(labels)
        rng.shuffle(base)
        shifts = list(range(count))
        rng.shuffle(shifts)
        for offset in shifts:
            if len(order) >= n_blocks:
                break
            order.append(base[offset:] + base[:offset])
    return order


def measure_invocation(fn, y: torch.Tensor, events=None) -> dict[str, float]:
    """Synchronous wall time and CUDA event time for one online invocation.

    CUDA events time the stream interval; the wall timer includes Python work,
    event recording and synchronization. Neither isolates pure arithmetic cost.
    """
    is_cuda = y.device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(y.device)
        start, end = events if events is not None else (
            torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        )
        t0 = time.perf_counter_ns()
        start.record()
        result = fn(y)
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter_ns() - t0) * 1e-6
        event_ms = float(start.elapsed_time(end))
    else:
        t0 = time.perf_counter_ns()
        result = fn(y)
        wall_ms = (time.perf_counter_ns() - t0) * 1e-6
        event_ms = float("nan")
    if result.shape != y.shape:
        raise ValueError("Online evaluator returned the wrong output shape")
    return {"wall_ms": wall_ms, "cuda_event_ms": event_ms}


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL, timeout=4,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable"


def _gpu_details(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    dev = torch.cuda.get_device_properties(device)
    return {"name": dev.name, "total_memory": int(dev.total_memory),
            "device": str(device), "cuda_runtime": str(torch.version.cuda)}


def run(args: argparse.Namespace) -> dict:
    labels = parse_p_labels(args.p_grid)
    sizes = parse_positive_sizes(args.sizes)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU smoke: --device cpu")
    if args.dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    if args.sessions < 1 or args.blocks < 1 or args.warmup < 0:
        raise ValueError("sessions and blocks must be positive; warmup nonnegative")
    if args.accuracy_n < 1 or args.fit_n <= 14 or args.validation_n < 3:
        raise ValueError("accuracy-n positive; fit-n > 14; validation-n >= 3")
    if args.target_norm_error <= 0 or not 0 <= args.active_ratio <= 1 or args.ymax_factor <= 1:
        raise ValueError("invalid accuracy target / input distribution parameters")

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} is not empty; choose a new output directory or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")

    created = datetime.now(timezone.utc).isoformat()
    fits = {}
    for label in labels:
        p = float(Fraction(label))
        degs = [work for family, work in CANDIDATE_GRID[label] if family == "chebyshev"]
        tic = time.perf_counter()
        fits[label] = fit_degree_family(p, degs, n_fit=args.fit_n, n_validation=args.validation_n)
        print(f"Preloaded p={label} coefficients, D={degs}, offline={time.perf_counter()-tic:.3f}s", flush=True)

    timing_rows: list[dict] = []
    accuracy_rows: list[dict] = []
    for session in range(args.sessions):
        session_seed = args.seed + 1009 * session
        rng = random.Random(session_seed)
        group_order = [(label, n) for label in labels for n in sizes]
        rng.shuffle(group_order)  # mitigates time/temperature trend across n and p
        for group_pos, (label, n) in enumerate(group_order):
            p = float(Fraction(label))
            tau = float(lp_constants(p, args.lam).tau)
            input_seed = session_seed + 1000003 * group_pos
            y = make_test_vector(
                n=n, tau=tau, active_ratio=args.active_ratio,
                ymax_factor=args.ymax_factor, dtype=dtype, device=device, seed=input_seed,
            )
            if tuple(y.shape) != (n,):
                raise RuntimeError("Expected a 1D input from make_test_vector")
            funcs = {}
            for family, work in CANDIDATE_GRID[label]:
                cfg = configuration(family, work)
                coeff = fits[label].coefficients[:work + 1] if family == "chebyshev" else None
                funcs[cfg] = make_preloaded_prox(
                    p=p, lam=args.lam, family=family, device=device,
                    dtype=dtype, work=work, coefficients=coeff,
                )

            # Shared *representative subset of the same input tensor*, outside timing.
            if n <= args.accuracy_n:
                ya = y
            else:
                ii = torch.linspace(0, n-1, args.accuracy_n, device=device).long()
                ya = y[ii]
            ya_cpu = ya.detach().cpu().numpy().astype(np.float64, copy=False)
            ref = prox_bisection_reference_numpy(ya_cpu, p, args.lam, iters=75)
            with torch.inference_mode():
                for cfg, fn in funcs.items():
                    candidate = fn(ya).detach().cpu().numpy().astype(np.float64, copy=False)
                    metrics = measure_accuracy(ya_cpu, candidate, ref, p=p, lam=args.lam)
                    accuracy_rows.append({
                        "p_label": label, "p": p, "n": n, "session": session,
                        "input_seed": input_seed, "configuration": cfg,
                        "n_accuracy": len(ya_cpu), "target_norm_error": args.target_norm_error,
                        "accuracy_pass": bool(metrics["all_finite"] and
                                              metrics["max_norm_error"] <= args.target_norm_error and
                                              metrics["support_mismatch_count"] == 0),
                        **metrics,
                    })
                # Candidate warmup is outside all timed measurements.
                for fn in funcs.values():
                    for _ in range(args.warmup):
                        fn(y)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                # Pre-allocate CUDA events outside timing; synchronization after
                # each invocation makes reuse on the same stream safe.
                events = ({cfg: (torch.cuda.Event(enable_timing=True),
                                 torch.cuda.Event(enable_timing=True))
                           for cfg in funcs} if device.type == "cuda" else {})
                if events:
                    for pair in events.values():
                        pair[0].record()
                        pair[1].record()
                    torch.cuda.synchronize(device)

                blocks = randomized_blocks(list(funcs), args.blocks, rng)
                for block_i, order in enumerate(blocks):
                    for order_i, cfg in enumerate(order):
                        sample = measure_invocation(funcs[cfg], y, events=events.get(cfg))
                        timing_rows.append({
                            "p_label": label, "p": p, "lambda": args.lam,
                            "n": n, "dtype": args.dtype, "device": str(device),
                            "session": session, "group_order": group_pos,
                            "block": block_i, "order_in_block": order_i,
                            "configuration": cfg, "input_seed": input_seed,
                            **sample,
                        })
            print(f"session {session+1}/{args.sessions} p={label} n={n}: "
                  f"{args.blocks} randomized blocks x {len(funcs)} candidates", flush=True)
            del y, ya, funcs, events

    timing = pd.DataFrame(timing_rows)
    accuracy = pd.DataFrame(accuracy_rows)
    timing.to_csv(out / "timings_raw.csv", index=False)
    accuracy.to_csv(out / "accuracy_checks.csv", index=False)
    metadata = {
        "created_utc": created,
        "experiment": "Phase 2B interleaved eager-PyTorch benchmark reliability",
        "git_commit": _git_head(),
        "p_labels": labels, "sizes": sizes, "sessions": args.sessions,
        "blocks_per_session": args.blocks, "warmup_per_candidate": args.warmup,
        "dtype": args.dtype, "device": str(device), "device_info": _gpu_details(device),
        "pytorch": str(torch.__version__), "numpy": str(np.__version__),
        "python": platform.python_version(), "os": platform.platform(),
        "lam": args.lam, "seed": args.seed, "active_ratio": args.active_ratio,
        "ymax_factor": args.ymax_factor, "accuracy_subset_max_n": args.accuracy_n,
        "accuracy_target_max_normalized_error": args.target_norm_error,
        "cheb_fit_n": args.fit_n, "cheb_validation_n": args.validation_n,
        "n_raw_timing_rows": len(timing), "n_accuracy_rows": len(accuracy),
        "all_accuracy_checks_pass": bool(accuracy["accuracy_pass"].all()),
        "fit_sampled_errors": {label: {str(k): v for k, v in fits[label].sampled_errors.items()}
                               for label in labels},
        "timing_note": "cuda_event_ms is GPU stream event interval; wall_ms includes Python dispatch + synchronization. "
                       "CUDA event pairs are preallocated and reused within a session/size group. "
                       "Each invocation is synchronized before its next pairing. "
                       "No CUDA Graph, kernel fusion, or device frequency control; not pure arithmetic time.",
        "independence_note": "Sessions inside one runtime are repeated measurement sessions, not independently restarted Colab runtimes.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(timing)} individual timings, {len(accuracy)} accuracy checks -> {out}")
    if not metadata["all_accuracy_checks_pass"]:
        print("ATTENTION: some configurations failed accuracy checks; do not claim unconditional accuracy-matched speedups")
    return metadata


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--p-grid", default="1/3,2/5,3/4")
    ap.add_argument("--sizes", default="10000,100000,1000000,10000000")
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--blocks", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--accuracy-n", type=int, default=8192,
                    help="Representative subset of each timed input for spot-check; not a uniform error certificate")
    ap.add_argument("--target-norm-error", type=float, default=1e-6)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--active-ratio", type=float, default=0.9)
    ap.add_argument("--ymax-factor", type=float, default=10.0)
    ap.add_argument("--fit-n", type=int, default=8192)
    ap.add_argument("--validation-n", type=int, default=80001)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--output-dir", default="results/phase2/benchmark_reliability/formal")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
