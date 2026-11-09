#!/usr/bin/env python3
"""Recompute Stage-3 proximal objective gaps without re-running timing.

Inputs, p, and reference method exactly match the Stage-3 fixed-input study.
The NumPy evaluation of the legacy Chebyshev/Newton/Bisection formulas is
checked against the saved sampled-error statistic for every configuration.
MB follows the paper-sign contour formula in revision.phase6.mb_port; the
real part is used as the scalar proximal candidate, with the complex-output
error and imaginary component checked separately by Stage 3.

Reconstruction is not a certified floating-point bound, and is not a timed
comparison. This file contains no manuscript or reviewer-response content.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from revision.phase6.mb_port import MBPort, constants, positive_root_bisect, series_normalized
from revision.phase6.scripts.benchmark_mb_cpu import inputs_phase5


def normalized_branch(q, p, iters=75):
    c = constants(p, 1.0)
    q = np.asarray(q, dtype=np.float64)
    lo = np.full_like(q, c.rho / c.tau)
    hi = np.ones_like(q)
    eps = p * c.tau ** (p - 2.0) * q
    for _ in range(iters):
        mid = lo + 0.5 * (hi - lo)
        move_right = (1.0 - mid) * mid ** (1.0 - p) > eps
        lo = np.where(move_right, mid, lo)
        hi = np.where(move_right, hi, mid)
    return np.where(q <= 0.0, 1.0, np.where(q >= 1.0, c.rho / c.tau, lo + 0.5 * (hi - lo)))


def cheb_coefficients(p, degree, nfit=8192):
    theta = (np.arange(nfit, dtype=np.float64) + 0.5) * (np.pi / nfit)
    values = normalized_branch(0.5 * (1.0 + np.cos(theta)), p)
    k = np.arange(degree + 1, dtype=np.float64)
    coeffs = (2.0 / nfit) * (np.cos(k[:, None] * theta[None, :]) @ values)
    coeffs[0] *= 0.5
    return coeffs


def evaluate_numpy(family, configuration, p, y, cheb_cache):
    c = constants(p, 1.0)
    a = np.abs(y)
    active = a > c.tau
    work = int(configuration[1:])
    if family == "chebyshev":
        if work not in cheb_cache:
            cheb_cache[work] = cheb_coefficients(p, work)
        eps = p * np.maximum(a, c.tau) ** (p - 2.0)
        eps_tau = p * c.tau ** (p - 2.0)
        q = np.clip(eps / eps_tau, 0.0, 1.0)
        z = np.polynomial.chebyshev.chebval(2.0 * q - 1.0, cheb_cache[work])
        x = np.where(active, np.sign(y) * a * z, 0.0)
        return x, x
    if family == "newton":
        a_safe = np.maximum(a, c.rho)
        x = a_safe.copy()
        for _ in range(work):
            xp1 = x ** (p - 1.0)
            h = x - a + p * xp1
            dh = 1.0 - p * (1.0 - p) * (xp1 / x)
            x = np.minimum(np.maximum(x - h / dh, c.rho), a_safe)
        x = np.where(active, np.sign(y) * x, 0.0)
        return x, x
    if family == "bisection":
        hi = np.maximum(a, c.rho)
        lo = np.full_like(a, c.rho)
        for _ in range(work):
            mid = 0.5 * (lo + hi)
            hmid = mid - a + p * mid ** (p - 1.0)
            hi = np.where(hmid > 0.0, mid, hi)
            lo = np.where(hmid > 0.0, lo, mid)
        x = np.where(active, np.sign(y) * 0.5 * (lo + hi), 0.0)
        return x, x
    if family == "fixed_series":
        x = np.zeros_like(y)
        aa = a[active]
        x[active] = np.sign(y[active]) * aa * series_normalized(p * aa ** (p - 2.0), p=p, n=work)
        return x, x
    raise ValueError(f"Unknown family {family!r}")


def objective_gap_statistics(y, x, ref, p):
    """Positive-part mean used in the paper; retain signed diagnostics too.

    Differences are computed after cancellation of the common y**2/2 term.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if y.shape != x.shape or y.shape != ref.shape:
        raise ValueError('Objective gap inputs must have matching shapes')
    dx = 0.5 * x*x - y*x + np.abs(x)**p
    dr = 0.5 * ref*ref - y*ref + np.abs(ref)**p
    gap = dx - dr
    if not np.all(np.isfinite(gap)):
        raise AssertionError('Nonfinite objective gap')
    return {
        'mean_prox_objective_gap': float(np.mean(np.maximum(gap, 0.0))),
        'mean_signed_objective_difference': float(np.mean(gap)),
        'max_prox_objective_gap': float(np.max(gap)),
        'min_signed_objective_gap': float(np.min(gap)),
        'max_abs_prox_objective_gap': float(np.max(np.abs(gap))),
    }


def replay_file(path, meta):
    rows = pd.read_csv(path)
    out = []
    if int(meta['n']) != 128 or not math.isclose(float(meta['lam']), 1.0):
        raise ValueError('This replay is written for the 128-input, lambda=1 Stage-3 record')
    for p, data in rows.groupby('p', sort=True):
        p = float(p)
        y = inputs_phase5(n=128, p=p, lam=1.0, seed=int(meta['seed']))
        digest = hashlib.sha256(y.tobytes()).hexdigest()
        if digest != meta['input_shas'][format(p, '.17g')]:
            raise AssertionError(f'p={p}: input SHA-256 mismatch')
        ref = positive_root_bisect(y, p=p, lam=1.0)
        c = constants(p, 1.0)
        cheb_cache = {}
        for row in data.to_dict('records'):
            family, conf = row['family'], row['configuration']
            if family == 'paper_MB_no_fallback':
                match = re.fullmatch(r'N(\d+)_T([0-9.]+)_nt(\d+)', conf)
                if not match: raise ValueError(f'Invalid MB config {conf}')
                N, T, nt = int(match[1]), float(match[2]), int(match[3])
                R = max(.9, (1 + c.qp) * .5) * c.ustar
                mb = MBPort(p=p, lam=1., n=N, T=T, nt=nt,
                            mode='paper', R=R, source_R=False)
                o = mb(y)
                x, z = o.x, o.x_complex
            else:
                x, z = evaluate_numpy(family, conf, p, y, cheb_cache)
            reproduced_e = float(np.max(np.abs(z - ref) / np.maximum(np.abs(y), c.tau)))
            saved_e = float(row['max_complex_normalized_error'])
            if not np.isclose(reproduced_e, saved_e, atol=1e-11, rtol=1e-6):
                raise AssertionError(f'p={p}, {family} {conf}: error does not reproduce: {reproduced_e} vs {saved_e}')
            out.append(dict(row, reproduced_max_complex_normalized_error=reproduced_e,
                            **objective_gap_statistics(y, x, ref, p),
                            objective_evaluation='real scalar candidate (MB: real part)'))
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', default='major7_cpu_all_methods.csv')
    parser.add_argument('--selected', default='major7_cpu_accuracy_matched.csv')
    parser.add_argument('--metadata', default='major7_cpu_metadata.json')
    parser.add_argument('--output-dir', default='major7_objective_gap_replay')
    args = parser.parse_args()
    meta = json.loads(Path(args.metadata).read_text())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for label, path in [('all_methods', args.all), ('selected', args.selected)]:
        df = replay_file(path, meta)
        dest = out / f'major7_{label}_with_objective_gaps.csv'
        df.to_csv(dest, index=False)
        print(f'PASS: reproduced all {len(df)} rows: {dest}')
    print('All normalized errors and source input hashes reproduced.')

if __name__ == '__main__':
    main()
