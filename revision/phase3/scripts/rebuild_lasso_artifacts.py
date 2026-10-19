#!/usr/bin/env python3
"""Rebuild publication-oriented curves and tables from Phase 3 raw CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize_run_table(summary: pd.DataFrame) -> pd.DataFrame:
    expected = {'seed', 'method', 'verified_stop_hit', 'first_verified_stop_iteration',
                'replay_budget_median_ms', 'final_verified_displacement',
                'final_objective', 'support_jaccard_vs_reference_FB'}
    if not expected.issubset(summary.columns):
        raise ValueError(f'Missing columns in run summary: {sorted(expected-set(summary.columns))}')
    rows = []
    for method, g in summary.groupby('method', sort=True):
        hits = g[g['verified_stop_hit'] == 1]
        row = {'method': method, 'seeds': len(g),
               'stop_hits': len(hits), 'all_hit': len(hits) == len(g),
               'median_final_objective': float(g['final_objective'].median()),
               'max_final_reference_displacement': float(g['final_verified_displacement'].max()),
               'min_support_jaccard_vs_reference': float(g['support_jaccard_vs_reference_FB'].min()),
               'median_support_jaccard_vs_reference': float(g['support_jaccard_vs_reference_FB'].median()),
               'median_iterations_to_target_if_hit': (float(hits['first_verified_stop_iteration'].median())
                                                      if len(hits) else float('nan')),
               'median_replay_time_ms_if_hit': (float(hits['replay_budget_median_ms'].median())
                                                if len(hits) else float('nan')),
               'median_checked_stop_wall_ms_if_hit': (float(hits['checked_stop_median_wall_ms'].median())
                                                      if len(hits) else float('nan'))}
        rows.append(row)
    return pd.DataFrame(rows)


def _style(ax):
    ax.grid(alpha=0.18)
    ax.spines[['top', 'right']].set_visible(False)


def make_seed_plots(traces: pd.DataFrame, refs: pd.DataFrame, out: Path,
                    seed: int, tol: float):
    df = traces[traces['seed'] == seed].copy()
    rd = refs[refs['seed'] == seed].copy()
    if df.empty:
        raise ValueError(f'Missing seed {seed}')
    methods = list(df['method'].unique())
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    minobj = min(df['objective'].min(), rd['objective'].min())
    # Main plot uses full F, not a gap to a potentially non-global reference.
    for method in methods:
        dd = df[df['method'] == method]
        ax.plot(dd['iteration'], dd['objective'], label=method, lw=1.8)
    if not rd.empty:
        ax.plot(rd['iteration'], rd['objective'], label='Reference FB (float64, untimed)',
                lw=1.25, ls='--', color='black')
    ax.set(xlabel='Forward--Backward iteration', ylabel='Objective F(x)',
           title=f'Objective vs iteration, seed={seed}')
    ax.set_ylim(bottom=minobj - 0.03 * max(1., abs(minobj)))
    _style(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f'objective_vs_iteration_seed{seed}.pdf')
    fig.savefig(out / f'objective_vs_iteration_seed{seed}.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for method in methods:
        dd = df[df['method'] == method]
        ax.plot(dd['cumulative_diagnostic_step_ms'], dd['objective'], label=method, lw=1.8)
    ax.set(xlabel='Cumulative diagnostic FB-step CUDA-event time (ms; checks excluded)',
           ylabel='Objective F(x)', title=f'Objective vs online-step time, seed={seed}')
    ax.set_ylim(bottom=minobj - 0.03 * max(1., abs(minobj)))
    _style(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f'objective_vs_online_time_seed{seed}.pdf')
    fig.savefig(out / f'objective_vs_online_time_seed{seed}.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for method in methods:
        dd = df[(df['method'] == method) & df['verified_displacement'].notna()]
        ax.semilogy(dd['iteration'], dd['verified_displacement'].clip(lower=1e-17),
                    marker='.', ms=3, label=method)
    if not rd.empty:
        dd = rd[rd['verified_displacement'].notna()]
        ax.semilogy(dd['iteration'], dd['verified_displacement'].clip(lower=1e-17),
                    ls='--', color='black', label='Reference FB')
    ax.axhline(tol, color='grey', ls=':', label=f'Target {tol:.0e}')
    ax.set(xlabel='Forward--Backward iteration', ylabel='Reference one-step displacement',
           title=f'Independent stopping test, seed={seed}')
    _style(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f'reference_displacement_seed{seed}.pdf')
    fig.savefig(out / f'reference_displacement_seed{seed}.png', dpi=180)
    plt.close(fig)


def run(root: Path):
    files = ['run_summary.csv', 'trajectories.csv', 'reference_FB_trajectories.csv',
             'support_comparison.csv', 'timings_raw.csv', 'metadata.json']
    for f in files:
        if not (root / f).is_file():
            raise FileNotFoundError(root / f)
    info = json.loads((root / 'metadata.json').read_text())
    summary = pd.read_csv(root / 'run_summary.csv', float_precision='round_trip')
    traces = pd.read_csv(root / 'trajectories.csv', float_precision='round_trip')
    refs = pd.read_csv(root / 'reference_FB_trajectories.csv', float_precision='round_trip')
    timing = pd.read_csv(root / 'timings_raw.csv', float_precision='round_trip')
    table = summarize_run_table(summary)
    table.to_csv(root / 'across_seed_summary.csv', index=False)
    out = root / 'figures'
    out.mkdir(exist_ok=True)
    for seed in sorted(traces.seed.unique()):
        make_seed_plots(traces, refs, out, int(seed), float(info['stopping_tolerance']))
    n_expected = (info['max_iterations']+1)*len(info['seeds'])*len(info['candidates'])
    if len(traces) != n_expected:
        raise ValueError(f'Trace row count {len(traces)} != expected {n_expected}')
    if not np.isfinite(traces['objective']).all():
        raise ValueError('Objective trace includes nonfinite values')
    if np.any(summary['support_jaccard_vs_reference_FB'] < 0) or np.any(summary['support_jaccard_vs_reference_FB'] > 1):
        raise ValueError('Jaccard outside [0,1]')
    lines = [
        '# Phase 3 end-to-end FB application: run summary', '',
        f'- Input dimensions: m={info["m"]}, n={info["n"]}; p={info["p"]}; lambda={info["lambda_reg"]}.',
        f'- Seeds: {info["seeds"]}; max iterations: {info["max_iterations"]}; independent verification every {info["verify_every"]} iterations.',
        f'- Common normalized reference-FB displacement tolerance: {info["stopping_tolerance"]:.2e}.',
        f'- Candidate methods: {info["candidates"]}.',
        f'- Raw objective traces and reference-FB curves: {len(traces)} candidate points, {len(refs)} reference points.',
        f'- Raw timing measurements: {len(timing)}.',
        '',
        '## Across-seed results', '',
        'A row with all_hit=False has no valid across-all-seeds time-to-target claim.', '',
        table.to_markdown(index=False, floatfmt='.5g'), '',
        '## Methodological restrictions', '',
        '- The objective-vs-online-time curves sum diagnostic step-event times and exclude reference verification.',
        '- The separately timed replay runs use iteration budgets selected by independent residual checks; these timings exclude checking cost.',
        '- The checked-stop wall time includes all high-accuracy reference evaluations and is substantially more expensive.',
        '- The float64 reference FB is a high-accuracy algorithmic comparator from the same initialization; it is not a global minimizer certificate.',
        '- Equality of nonzero counts is not equality of supports; compare the recorded Jaccard and exact-support indicators.',
        '- Multiple seeds share one code/hardware environment; intervals or generality require separate experimental evidence.',
    ]
    (root / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print(f'Rebuilt {len(table)} per-method aggregate rows, {len(traces)} trace rows, plots for {len(info["seeds"])} seeds')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    run(args.root)
