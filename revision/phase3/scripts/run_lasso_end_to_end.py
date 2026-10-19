#!/usr/bin/env python3
"""End-to-end ell_p-Lasso forward-backward comparison with an independent stop check.

Two separate tracks: (i) diagnostic trajectories (no runtime claims), and
(ii) timed replays / verified-stop loops. No existing proximal code is modified.

The verified residual is ||x-T_ref(x-alpha*grad f(x))||/max(1,||x||), where
T_ref uses independent float64 scalar bisection after float64 gradient evaluation.
This is a normalized *one-step displacement*, not the gradient mapping norm;
the latter is 1/alpha times this quantity.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from fast_lp_prox.revision_bench import (
    fit_degree_family, make_preloaded_prox, prox_bisection_reference_numpy,
)


def fraction_float(raw: str) -> float:
    try:
        x = float(Fraction(raw))
    except (ValueError, ZeroDivisionError, TypeError) as exc:
        raise argparse.ArgumentTypeError('Expected p as 1/3 or 0.75') from exc
    if not (0 < x < 1):
        raise argparse.ArgumentTypeError('p must belong to (0,1)')
    return x


def parse_seeds(spec: str) -> list[int]:
    values = [int(s.strip()) for s in spec.split(',') if s.strip()]
    if not values or any(v < 0 for v in values) or len(set(values)) != len(values):
        raise ValueError('Seeds must be distinct nonnegative integers')
    return values


@dataclass(frozen=True)
class Candidate:
    family: str
    work: int

    @property
    def label(self) -> str:
        return f'{self.family}_{"D" if self.family == "chebyshev" else "K"}{self.work}'


def parse_candidates(raw: str) -> list[Candidate]:
    entries = []
    for part in raw.split(','):
        k, sep, v = part.strip().partition(':')
        if not sep or k not in ('chebyshev', 'newton', 'bisection'):
            raise ValueError('Candidates must use family:work, e.g. chebyshev:6,newton:3')
        n = int(v)
        if n < 1:
            raise ValueError('Degree/iterations must be positive')
        entries.append(Candidate(k, n))
    if not entries or len({x.label for x in entries}) != len(entries):
        raise ValueError('Candidate configurations must be nonempty and unique')
    return entries


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True,
            stderr=subprocess.DEVNULL, timeout=2,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return 'unavailable'


def torch_device(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False')
    return torch.device(name)


@dataclass
class Problem:
    A: torch.Tensor
    b: torch.Tensor
    Ad: torch.Tensor
    bd: torch.Tensor
    xtrue_support: np.ndarray
    alpha: float
    lambda_reg: float
    p: float
    seed: int
    L_est: float

    @property
    def n(self) -> int:
        return int(self.A.shape[1])


def estimate_lipschitz(A: torch.Tensor, *, seed: int, steps: int = 100) -> float:
    n = A.shape[1]
    rng = np.random.default_rng(seed + 110003)
    v = torch.tensor(rng.standard_normal(n), dtype=A.dtype, device=A.device)
    v = v / torch.linalg.vector_norm(v)
    with torch.no_grad():
        for _ in range(steps):
            w = A.T @ (A @ v)
            norm = torch.linalg.vector_norm(w)
            v = w / norm
        L = float(torch.dot(v, A.T @ (A @ v)).item())
    if L <= 0 or not math.isfinite(L):
        raise RuntimeError('Lipschitz estimate is not positive and finite')
    return L


def make_problem(*, m: int, n: int, support: int, sigma: float,
                 lam: float, p: float, seed: int, device: torch.device,
                 dtype: torch.dtype, power_iterations: int) -> Problem:
    if not (m >= 1 and n >= 2 and 1 <= support <= n):
        raise ValueError('Invalid problem dimensions/support size')
    if not (lam > 0 and sigma >= 0):
        raise ValueError('lambda must be positive and sigma nonnegative')
    rng = np.random.default_rng(seed)
    A_np = rng.standard_normal((m, n)).astype(np.float32) / np.float32(np.sqrt(m))
    xtrue = np.zeros(n, dtype=np.float64)
    supp = np.sort(rng.choice(n, size=support, replace=False))
    xtrue[supp] = rng.standard_normal(support)
    # Observation generation in float64 from the same quantized matrix.
    b_np = A_np.astype(np.float64) @ xtrue + sigma * rng.standard_normal(m)
    A = torch.as_tensor(A_np, dtype=dtype, device=device)
    b = torch.as_tensor(b_np, dtype=dtype, device=device)
    # Both the candidate and reference use exactly the SAME stored rounded data.
    Ad = A.to(dtype=torch.float64)
    bd = b.to(dtype=torch.float64)
    L_est = estimate_lipschitz(A, seed=seed, steps=power_iterations)
    alpha = 0.95 / L_est
    return Problem(A, b, Ad, bd, supp, alpha, lam, p, seed, L_est)


def forward_backward_step(x: torch.Tensor, problem: Problem, prox) -> torch.Tensor:
    r = x - problem.alpha * (problem.A.T @ (problem.A @ x - problem.b))
    return prox(r)


def reference_update(x: torch.Tensor, problem: Problem,
                     *, bisect_iters: int = 80) -> np.ndarray:
    """Independent high-accuracy FB update: float64 gradient and prox bisection."""
    with torch.no_grad():
        xd = x.to(dtype=torch.float64)
        z = xd - problem.alpha * (problem.Ad.T @ (problem.Ad @ xd - problem.bd))
    z_np = z.detach().cpu().numpy()
    return prox_bisection_reference_numpy(
        z_np, p=problem.p, lam=problem.alpha * problem.lambda_reg,
        iters=bisect_iters,
    )


def relative_verified_displacement(x: torch.Tensor, problem: Problem,
                                   *, bisect_iters: int = 80) -> float:
    """Reference fixed-point residual; independent of candidate prox backend."""
    xnp = x.detach().to(dtype=torch.float64).cpu().numpy()
    ref = reference_update(x, problem, bisect_iters=bisect_iters)
    return float(np.linalg.norm(xnp - ref) / max(1.0, np.linalg.norm(xnp)))


def objective_tensor(x: torch.Tensor, problem: Problem) -> torch.Tensor:
    # Diagnostic objective in float64; it is not part of timed FB step.
    xd = x.to(dtype=torch.float64)
    return (0.5 * torch.sum((problem.Ad @ xd - problem.bd) ** 2)
            + problem.lambda_reg * torch.sum(torch.abs(xd) ** problem.p))


def support_stats(x: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    a = set(np.flatnonzero(x != 0.0).tolist())
    b = set(int(i) for i in target.tolist())
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return {'size': len(a), 'intersection': intersection, 'union': union,
            'jaccard': intersection / union if union else 1.0,
            'exact_support': int(a == b)}


def trace_method(problem: Problem, label: str, prox, *, max_iter: int,
                 verify_every: int, bisect_iters: int, collect_timing: bool = True,
                 float64_iterates: bool = False) -> tuple[list[dict], torch.Tensor]:
    """Untimed diagnostics: objective each iteration, reference check every q.

    Online event intervals exclude objective evaluation and reference validation.
    They are diagnostic timings (not the more reliable replay times).
    """
    x = torch.zeros(problem.n, dtype=(torch.float64 if float64_iterates else problem.A.dtype),
                    device=problem.A.device)
    init_res = relative_verified_displacement(x, problem, bisect_iters=bisect_iters)
    objectives = [objective_tensor(x, problem)]
    residuals = [init_res]
    start_events, end_events, cpu_durations = [], [], []
    with torch.no_grad():
        for k in range(1, max_iter + 1):
            if problem.A.device.type == 'cuda' and collect_timing:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
            else:
                start = time.perf_counter()
            if float64_iterates:
                r = x - problem.alpha * (problem.Ad.T @ (problem.Ad @ x - problem.bd))
                x_new = torch.as_tensor(prox_bisection_reference_numpy(
                    r.detach().cpu().numpy(), problem.p,
                    problem.alpha * problem.lambda_reg, iters=bisect_iters),
                    dtype=torch.float64, device=x.device)
            else:
                x_new = forward_backward_step(x, problem, prox)
            if problem.A.device.type == 'cuda' and collect_timing:
                e.record()
                start_events.append(s)
                end_events.append(e)
            else:
                cpu_durations.append((time.perf_counter() - start) * 1000)
            x = x_new
            objectives.append(objective_tensor(x, problem))
            if k % verify_every == 0 or k == max_iter:
                residuals.append(relative_verified_displacement(
                    x, problem, bisect_iters=bisect_iters))
            else:
                residuals.append(float('nan'))
    if problem.A.device.type == 'cuda':
        torch.cuda.synchronize(problem.A.device)
    obj_values = torch.stack(objectives).detach().cpu().numpy().tolist()
    if problem.A.device.type == 'cuda' and collect_timing:
        times = [float(s.elapsed_time(e)) for s, e in zip(start_events, end_events)]
    else:
        times = cpu_durations
    if len(times) != max_iter:
        raise RuntimeError('Trace timing count mismatch')
    rows = []
    cumulative = 0.0
    for k in range(max_iter + 1):
        if k:
            cumulative += times[k - 1]
        rows.append({
            'seed': problem.seed, 'p': problem.p, 'method': label, 'iteration': k,
            'objective': float(obj_values[k]),
            'verified_displacement': float(residuals[k]),
            'cumulative_diagnostic_step_ms': cumulative,
        })
    return rows, x


def first_verified_hit(trace: list[dict], target: float) -> int | None:
    for row in trace:
        r = float(row['verified_displacement'])
        if math.isfinite(r) and r <= target:
            return int(row['iteration'])
    return None


def timed_replay(problem: Problem, prox, steps: int, *, repeats: int,
                 warmup: int) -> list[dict]:
    if steps < 0:
        raise ValueError('steps must be nonnegative')
    rows = []
    with torch.no_grad():
        for i in range(warmup + repeats):
            x = torch.zeros(problem.n, device=problem.A.device, dtype=problem.A.dtype)
            if problem.A.device.type == 'cuda':
                torch.cuda.synchronize(problem.A.device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            t0 = time.perf_counter()
            for _ in range(steps):
                x = forward_backward_step(x, problem, prox)
            if problem.A.device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize(problem.A.device)
                device_ms = float(start_event.elapsed_time(end_event))
            else:
                device_ms = float('nan')
            wall_ms = 1000.0 * (time.perf_counter() - t0)
            if i >= warmup:
                rows.append({'repeat': i - warmup, 'steps': steps,
                             'replay_cuda_event_ms': device_ms,
                             'replay_wall_ms': wall_ms})
    return rows


def timed_checked_stop(problem: Problem, prox, *, max_iter: int,
                       verify_every: int, tol: float, bisect_iters: int,
                       repeats: int) -> list[dict]:
    """Real stop timing with reference checks INCLUDED in elapsed wall clock."""
    rows = []
    with torch.no_grad():
        for rep in range(repeats):
            x = torch.zeros(problem.n, dtype=problem.A.dtype, device=problem.A.device)
            if problem.A.device.type == 'cuda':
                torch.cuda.synchronize(problem.A.device)
            t0 = time.perf_counter()
            r = relative_verified_displacement(x, problem, bisect_iters=bisect_iters)
            k_hit = 0 if r <= tol else None
            for k in range(1, max_iter + 1):
                if k_hit is not None:
                    break
                x = forward_backward_step(x, problem, prox)
                if k % verify_every == 0 or k == max_iter:
                    r = relative_verified_displacement(x, problem, bisect_iters=bisect_iters)
                    if r <= tol:
                        k_hit = k
                        break
            if problem.A.device.type == 'cuda':
                torch.cuda.synchronize(problem.A.device)
            rows.append({'repeat': rep, 'checked_stop_hit': k_hit is not None,
                         'checked_stop_iteration': k_hit if k_hit is not None else max_iter,
                         'checked_stop_wall_ms': 1000.0*(time.perf_counter()-t0),
                         'final_verified_displacement': r})
    return rows


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Revision: end-to-end ell_p-Lasso FB comparison')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    ap.add_argument('--p', type=fraction_float, default=1/3)
    ap.add_argument('--p-label', default='1/3')
    ap.add_argument('--lambda-reg', type=float, default=0.05)
    ap.add_argument('--m', type=int, default=2000)
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--support', type=int, default=200)
    ap.add_argument('--sigma', type=float, default=0.01)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--max-iter', type=int, default=250)
    ap.add_argument('--verify-every', type=int, default=5)
    ap.add_argument('--tol', type=float, default=1e-5)
    ap.add_argument('--power-iterations', type=int, default=100)
    ap.add_argument('--ref-bisect-iters', type=int, default=80)
    ap.add_argument('--candidates', default='chebyshev:6,newton:3,bisection:20')
    ap.add_argument('--n-fit', type=int, default=8192)
    ap.add_argument('--n-val', type=int, default=80001)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--checked-stop-repeats', type=int, default=2)
    ap.add_argument('--output-dir', default='results/revision_phase3_lasso')
    ap.add_argument('--overwrite', action='store_true')
    return ap


def run(args: argparse.Namespace) -> None:
    if args.max_iter < 1 or args.verify_every < 1 or args.repeats < 1 or args.warmup < 0:
        raise ValueError('Invalid max-iter, verify-every, repeats or warmup')
    if args.checked_stop_repeats < 0 or args.tol <= 0 or args.ref_bisect_iters < 3:
        raise ValueError('Invalid checked-stop-repeats, tol or bisection iteration count')
    if args.power_iterations < 1:
        raise ValueError('power-iterations must be positive')
    candidates = parse_candidates(args.candidates)
    seeds = parse_seeds(args.seeds)
    device = torch_device(args.device)
    dtype = torch.float32 if args.dtype == 'float32' else torch.float64
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{output} exists and is nonempty; choose a new --output-dir')
    output.mkdir(parents=True, exist_ok=True)
    # One shared p-dependent offline table; can be reused at every lambda.
    degrees = sorted({c.work for c in candidates if c.family == 'chebyshev'})
    fit = None
    t_fit = time.perf_counter()
    if degrees:
        fit = fit_degree_family(args.p, degrees, n_fit=args.n_fit,
                                n_validation=args.n_val)
    offline_sec = time.perf_counter() - t_fit
    traces, ref_traces, summaries, timing_rows, support_rows = [], [], [], [], []
    parameters = []
    for seed in seeds:
        prob = make_problem(m=args.m, n=args.n, support=args.support,
                            sigma=args.sigma, lam=args.lambda_reg, p=args.p,
                            seed=seed, device=device, dtype=dtype,
                            power_iterations=args.power_iterations)
        print(f'seed={seed}, p={args.p:.8g}, alpha={prob.alpha:.6g}, '
              f'mu={prob.alpha*args.lambda_reg:.6g}, device={device}', flush=True)
        parameters.append({'seed': seed, 'alpha': prob.alpha,
                           'L_est': prob.L_est, 'mu': prob.alpha * args.lambda_reg})
        # All methods use exactly the same synthetic instance and x0=0.
        reference_rows, x_reference = trace_method(
            prob, 'reference_FB_float64', None, max_iter=args.max_iter,
            verify_every=args.verify_every,
            bisect_iters=args.ref_bisect_iters, float64_iterates=True,
            collect_timing=False)
        ref_traces.extend(reference_rows)
        target_support = np.flatnonzero(x_reference.detach().cpu().numpy() != 0)
        ref_obj = reference_rows[-1]['objective']
        for candidate in candidates:
            coeff = (fit.coefficients[:candidate.work + 1] if
                     candidate.family == 'chebyshev' else None)
            prox = make_preloaded_prox(
                p=args.p, lam=prob.alpha * args.lambda_reg,
                family=candidate.family, work=candidate.work,
                coefficients=coeff, dtype=dtype, device=device)
            trace, x_final = trace_method(
                prob, candidate.label, prox, max_iter=args.max_iter,
                verify_every=args.verify_every, bisect_iters=args.ref_bisect_iters)
            traces.extend(trace)
            hit = first_verified_hit(trace, args.tol)
            budget = hit if hit is not None else args.max_iter
            for row in timed_replay(prob, prox, budget,
                                    repeats=args.repeats, warmup=args.warmup):
                timing_rows.append({'seed': seed, 'method': candidate.label,
                                    'timing_type': 'replay_no_validation', **row})
            checked_times = []
            if args.checked_stop_repeats:
                checked_times = timed_checked_stop(
                    prob, prox, max_iter=args.max_iter,
                    verify_every=args.verify_every, tol=args.tol,
                    bisect_iters=args.ref_bisect_iters,
                    repeats=args.checked_stop_repeats)
                for row in checked_times:
                    timing_rows.append({'seed': seed, 'method': candidate.label,
                                        'timing_type': 'checked_stop_with_reference', **row})
            final_np = x_final.detach().cpu().numpy()
            st = support_stats(final_np, target_support)
            true_st = support_stats(final_np, prob.xtrue_support)
            replay_rows = [r for r in timing_rows if r['seed'] == seed and
                           r['method'] == candidate.label and
                           r['timing_type'] == 'replay_no_validation']
            metric = 'replay_cuda_event_ms' if device.type == 'cuda' else 'replay_wall_ms'
            replay_med = float(np.median([r[metric] for r in replay_rows]))
            checked_med = (float(np.median([r['checked_stop_wall_ms']
                           for r in checked_times])) if checked_times else float('nan'))
            summaries.append({
                'seed': seed, 'p': args.p, 'method': candidate.label,
                'family': candidate.family, 'work': candidate.work,
                'L_est': prob.L_est, 'alpha': prob.alpha,
                'mu': prob.alpha * args.lambda_reg,
                'verified_stop_tolerance': args.tol,
                'verified_stop_hit': int(hit is not None),
                'first_verified_stop_iteration': hit if hit is not None else -1,
                'time_budget_iterations': budget,
                'replay_budget_median_ms': replay_med,
                'checked_stop_median_wall_ms': checked_med,
                'final_objective': float(trace[-1]['objective']),
                'reference_FB_objective_at_max_iter': ref_obj,
                'objective_minus_reference_FB': float(trace[-1]['objective'] - ref_obj),
                'final_verified_displacement': float(trace[-1]['verified_displacement']),
                'final_nonzero_count': st['size'],
                'support_jaccard_vs_reference_FB': st['jaccard'],
                'support_exact_vs_reference_FB': st['exact_support'],
                'support_jaccard_vs_truth': true_st['jaccard'],
            })
            support_rows.append({'seed': seed, 'method': candidate.label,
                                 'size': st['size'],
                                 'reference_size': len(target_support),
                                 'intersection': st['intersection'],
                                 'union': st['union'],
                                 'jaccard': st['jaccard'],
                                 'exact_support': st['exact_support'],
                                 'true_support_jaccard': true_st['jaccard']})
            print(f'  {candidate.label}: hit={hit} (max={args.max_iter}), '
                  f'F={trace[-1]["objective"]:.7g}, '
                  f'Rref={trace[-1]["verified_displacement"]:.4g}, '
                  f'Jaccard={st["jaccard"]:.3f}, replay_ms={replay_med:.4g}',
                  flush=True)
    pd.DataFrame(traces).to_csv(output / 'trajectories.csv', index=False)
    pd.DataFrame(ref_traces).to_csv(output / 'reference_FB_trajectories.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'run_summary.csv', index=False)
    pd.DataFrame(timing_rows).to_csv(output / 'timings_raw.csv', index=False)
    pd.DataFrame(support_rows).to_csv(output / 'support_comparison.csv', index=False)
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'experiment': 'end-to-end nonconvex ell_p-Lasso FB accuracy-matched comparison',
        'code_commit': git_commit(), 'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'torch': torch.__version__, 'numpy': np.__version__,
        'python': sys.version, 'platform': platform.platform(),
        'm': args.m, 'n': args.n, 's': args.support, 'sigma': args.sigma,
        'p': args.p, 'p_label': args.p_label,
        'lambda_reg': args.lambda_reg, 'seeds': seeds,
        'max_iterations': args.max_iter, 'verify_every': args.verify_every,
        'reference_bisection_iterations': args.ref_bisect_iters,
        'stopping_tolerance': args.tol, 'candidates': [c.label for c in candidates],
        'offline_fit_seconds': offline_sec,
        'fit_sampled_errors': (fit.sampled_errors if fit else {}),
        'parameters_per_seed': parameters,
        'time_semantics': {
            'cumulative_diagnostic_step_ms': 'per-step CUDA events, excluding objective and reference checks; diagnostic, not benchmark timing',
            'replay_budget_median_ms': 'median of timed full-loop replays up to first independently verified checkpoint; excludes reference checks',
            'checked_stop_median_wall_ms': 'median actual wall-clock time until reference residual meets tolerance, INCLUDING all high-precision checks',
            'uncertainty': 'all measurements in one runtime; repeated starts separate, no across-GPU guarantees',
        },
        'reference_semantics': 'float64 gradient from rounded A,b, independent float64 numpy bisection prox; zero on mathematical threshold tie',
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Wrote {output} (traces, summary, timings, support, metadata)', flush=True)


if __name__ == '__main__':
    run(parser().parse_args())
