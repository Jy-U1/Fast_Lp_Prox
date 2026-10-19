"""Fast CPU tests for the application-level Major Revision benchmark."""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
import pytest
import torch

from revision.phase3.scripts.run_lasso_end_to_end import (
    Candidate, fraction_float, parse_candidates, parse_seeds, make_problem,
    reference_update, relative_verified_displacement, objective_tensor,
    support_stats, first_verified_hit, run,
)
from revision.phase3.scripts.rebuild_lasso_artifacts import summarize_run_table, run as rebuild
from fast_lp_prox.revision_bench import prox_bisection_reference_numpy


def test_fraction_and_candidates():
    assert abs(fraction_float('1/3') - 1/3) < 1e-15
    assert [c.label for c in parse_candidates('chebyshev:6,newton:3,bisection:20')] == [
        'chebyshev_D6','newton_K3','bisection_K20']
    with pytest.raises(ValueError):
        parse_candidates('newton:3,newton:3')
    with pytest.raises(argparse.ArgumentTypeError):
        fraction_float('1')
    with pytest.raises(ValueError):
        parse_seeds('1,1')


def test_support_checks_positions_not_only_count():
    a = np.array([0.,1.,0.,2.])
    s = support_stats(a, np.array([0,2]))
    assert s['size'] == 2
    assert s['exact_support'] == 0
    assert s['jaccard'] == 0.0
    assert support_stats(np.zeros(3), np.array([],dtype=int))['jaccard'] == 1


def test_independent_reference_update_is_finite_and_consistent():
    p = make_problem(m=12, n=18, support=3, sigma=.01, lam=.05,
                     p=1/3, seed=12, device=torch.device('cpu'),
                     dtype=torch.float32, power_iterations=20)
    assert torch.equal(p.Ad.to(torch.float32), p.A)
    assert torch.equal(p.bd.to(torch.float32), p.b)
    x = torch.zeros(p.n)
    ref = reference_update(x, p, bisect_iters=75)
    assert ref.shape == (18,)
    assert np.isfinite(ref).all()
    assert abs(relative_verified_displacement(x,p,bisect_iters=75)-
               np.linalg.norm(ref)) < 1e-10
    xref = torch.tensor(ref, dtype=torch.float64)
    assert np.isfinite(float(objective_tensor(xref, p)))


def test_reference_scalar_prox_stationarity():
    z = np.array([-3.,-2.,0.,2.,3.], dtype=np.float64)
    out = prox_bisection_reference_numpy(z,p=.4,lam=.25,iters=75)
    active = out != 0
    assert np.allclose(out, -out[::-1])
    assert np.max(np.abs(np.abs(out[active])+.25*.4*np.abs(out[active])**(-.6)-
                         np.abs(z[active]))) < 1e-12


def test_first_hit_skips_unverified_points():
    rows = [{'iteration':0,'verified_displacement':1.},
            {'iteration':1,'verified_displacement':float('nan')},
            {'iteration':2,'verified_displacement':0.0002}]
    assert first_verified_hit(rows, 1e-3) == 2
    assert first_verified_hit(rows, 1e-5) is None


def test_summarize_handles_missed_target():
    df = pd.DataFrame({'seed':[0,1], 'method':['newton_K3']*2,
        'verified_stop_hit':[1,0], 'first_verified_stop_iteration':[10,-1],
        'replay_budget_median_ms':[2,20],
        'final_verified_displacement':[1e-5,1e-3],
        'final_objective':[5,6],
        'support_jaccard_vs_reference_FB':[1,.5],
        'checked_stop_median_wall_ms':[100,200]})
    result = summarize_run_table(df).iloc[0]
    assert not result['all_hit']
    assert result['stop_hits'] == 1
    assert result['median_replay_time_ms_if_hit'] == 2


def test_end_to_end_smoke_and_rebuild(tmp_path):
    a = argparse.Namespace(device='cpu', dtype='float32', p=1/3,
        p_label='1/3', lambda_reg=.05, m=14, n=25,
        support=4, sigma=.01, seeds='1', max_iter=12,
        verify_every=3, tol=1e-2, power_iterations=20,
        ref_bisect_iters=65, candidates='chebyshev:6,newton:3,bisection:12',
        n_fit=96, n_val=121, repeats=1, warmup=0,
        checked_stop_repeats=1, output_dir=str(tmp_path/'small'),
        overwrite=False)
    run(a)
    root = tmp_path/'small'
    assert json.loads((root/'metadata.json').read_text())['m'] == 14
    traces = pd.read_csv(root/'trajectories.csv')
    assert len(traces)==3*(12+1)
    assert np.isfinite(traces['objective']).all()
    summary = pd.read_csv(root/'run_summary.csv')
    assert len(summary)==3
    assert summary.support_jaccard_vs_reference_FB.between(0,1).all()
    rebuild(root)
    assert len(pd.read_csv(root/'across_seed_summary.csv'))==3
    assert (root/'figures/objective_vs_iteration_seed1.png').is_file()
    with pytest.raises(FileExistsError):
        run(a)
