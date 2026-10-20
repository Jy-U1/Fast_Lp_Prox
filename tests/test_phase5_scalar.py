"""Tests are independent of private GitHub repo and do not require a GPU."""
import csv
from pathlib import Path
import math
import os
import subprocess
import sys

import mpmath as mp
import numpy as np
import pytest
import torch

from revision.phase5.backends import (
    FixedSeries, ResidualSeriesBisection, fixed_term_series,
    normalized_stationarity, proximal_constants, reference_bisection,
    scaled_series_coefficients, select_against_zero,
)
from revision.phase5.scripts.benchmark_phase5_scalar import (
    generate_shared_inputs, metrics,
)


@pytest.mark.parametrize('p', [1/3, 2/5, 3/4, .9, .99, .999])
def test_threshold_and_scaled_parameter(p):
    c = proximal_constants(p, lam=1.0)
    assert 0 < c.rho < c.tau
    assert 0 < c.qp < 1.0
    eps_over = p * c.tau ** (p - 2) / c.ustar
    assert abs(eps_over - c.qp) < 1e-13
    assert 0.5 * c.rho**2 == pytest.approx((1-p) * c.rho**p, rel=1e-10)


@pytest.mark.parametrize('p', [1/3, 3/4, .9, .99, .999])
def test_scaled_coefficients_match_formula(p):
    coefs = scaled_series_coefficients(p, 48)
    mp.mp.dps = 90
    u = mp.mpf(str(proximal_constants(p, 1.).ustar))
    pp = mp.mpf(str(p))
    for n in (1, 2, 7, 15, 48):
        b = mp.gamma((2-pp)*n-1)/(mp.factorial(n)*mp.gamma((1-pp)*n))*u**n
        assert coefs[n-1] == pytest.approx(float(b), rel=2e-12)


@pytest.mark.parametrize('p,N,tol', [(1/3, 32, 1e-11), (2/5, 64, 1e-12),
                                      (3/4, 256, 1e-10), (.9, 512, 1e-6)])
def test_fixed_series_converges_on_active_branch(p, N, tol):
    c = proximal_constants(p, .1)
    y = torch.tensor([0.0, c.tau*(1-1e-3), c.tau,
                      c.tau*(1+1e-4), c.tau*1.5, c.tau*10,
                      -c.tau*(1+1e-4)], dtype=torch.float64)
    out = FixedSeries(p=p, lam=.1, terms=N)(y)
    ref = reference_bisection(y, p=p, lam=.1)
    assert torch.allclose(out[:3], torch.zeros_like(out[:3]))
    assert out[-1] == pytest.approx(-out[3].item())
    assert float((out-ref).abs().max()) < tol


@pytest.mark.parametrize('p', [.3333333333333333, .75, .99])
def test_reference_against_mpmath(p):
    mp.mp.dps = 95
    c = proximal_constants(p, .04)
    for factor in [1.0001, 1.05, 5.0]:
        yf = c.tau * factor
        out = float(reference_bisection(torch.tensor([yf],dtype=torch.float64),
                                        p=p,lam=.04).item())
        pp, ll, yy = mp.mpf(str(p)), mp.mpf('0.04'), mp.mpf(str(yf))
        lo = (2*ll*(1-pp))**(1/(2-pp))
        hi = yy
        for _ in range(310):
            m = (lo+hi)/2
            r = m-yy+ll*pp*m**(pp-1)
            if r <= 0: lo = m
            else: hi = m
        exact = float((lo+hi)/2)
        assert out == pytest.approx(exact, rel=3e-13, abs=2e-13)


@pytest.mark.parametrize('p', [1/3,.75,.9,.99,.999])
def test_residual_checked_hybrid(p):
    c = proximal_constants(p, 1.0)
    y = torch.tensor([c.tau*(1+1e-6), c.tau*1.02, c.tau*3.0,
                      -c.tau*10],dtype=torch.float32)
    backend = ResidualSeriesBisection(p=p,lam=1.0,
                                      orders=(8,32,128),tol=1e-6,
                                      bisect_steps=75)
    out = backend(y)
    ref = reference_bisection(y, p=p,lam=1.0)
    err = (out.to(torch.float64)-ref).abs()/y.to(torch.float64).abs()
    assert float(err.max()) < 1.3e-6
    assert backend.last_stats.total_active == 4
    assert backend.last_stats.series_accepted + backend.last_stats.fallback_count == 4
    assert backend.last_stats.max_indicator_accepted <= 1e-6


def test_hard_case_forces_fallback_and_stays_finite():
    p=.999
    c=proximal_constants(p,lam=1.)
    y=torch.tensor([c.tau*(1+1e-7),c.tau*1.0001,c.tau*1.1],dtype=torch.float64)
    b=ResidualSeriesBisection(p=p,lam=1.,orders=(4,8),tol=1e-11)
    z=b(y)
    assert b.last_stats.fallback_count >= 1
    assert bool(torch.isfinite(z).all())
    r=reference_bisection(y,p=p,lam=1.)
    assert torch.allclose(z,r,rtol=1e-10,atol=1e-11)


def test_objective_guard_near_threshold_and_bad_candidates():
    c=proximal_constants(1/3, 1.0)
    y=torch.tensor([c.tau*1.000001, 12.0, -12.0, 0.0],dtype=torch.float32)
    cand=torch.tensor([3.0, 11.9, -11.9, 99.0],dtype=torch.float32)
    guarded=select_against_zero(y,cand,p=1/3,lam=1.)
    assert guarded[0] == 0
    assert guarded[1] == cand[1]
    assert guarded[2] == cand[2]
    assert guarded[3] == 0
    xd=guarded.to(torch.float64)
    yy=y.to(torch.float64)
    obj_delta=0.5*xd.square()-yy*xd+xd.abs().pow(1/3)
    assert bool((obj_delta <= 1e-13).all())


def test_generator_and_metrics_are_reproducible():
    device=torch.device('cpu')
    x=generate_shared_inputs(n=1000,p=1/3,lam=1.0,active_fraction=.9,
                             ymax_factor=10,seed=41,dtype=torch.float64,device=device)
    z=generate_shared_inputs(n=1000,p=1/3,lam=1.0,active_fraction=.9,
                             ymax_factor=10,seed=41,dtype=torch.float64,device=device)
    assert torch.equal(x,z)
    ref=reference_bisection(x,p=1/3,lam=1.)
    met=metrics(x,ref,ref,p=1/3,lam=1.)
    assert met['support_mismatch_count']==0
    assert met['max_norm_error']==0
    c=proximal_constants(1/3,1.)
    assert abs((x.abs()>c.tau).float().mean().item()-.9)<.01


def test_bad_parameters_rejected():
    for p,lam in [(1.,1.),(0.,1.),(.5,0.),(float('nan'),.1)]:
        with pytest.raises(ValueError): proximal_constants(p,lam)
    with pytest.raises(ValueError): scaled_series_coefficients(.5,0)
    with pytest.raises(ValueError): ResidualSeriesBisection(p=.5,lam=1.,orders=(8,8))
    with pytest.raises(ValueError): reference_bisection(torch.zeros(3),p=.5,lam=1.,iterations=0)


def test_smoke_script_creates_archive_and_refuses_overwrite(tmp_path, monkeypatch):
    from revision.phase5.scripts.benchmark_phase5_scalar import main
    out = tmp_path/'new_results'
    argv=['phase5', '--n','128','--warmup','0','--repeats','1','--device','cpu',
          '--dtype','float64','--p','0.75','--series-terms','8,16',
          '--hybrid-orders','8,32','--existing-methods','none',
          '--output-dir',str(out)]
    monkeypatch.setattr(sys,'argv',argv)
    main()
    with (out/'phase5_scalar_results.csv').open() as f:
        data=list(csv.DictReader(f))
    assert len(data)==3
    assert {r['family'] for r in data}=={'fixed_series','residual_series_bisect'}
    assert all(float(row['median_ms'])>=0 for row in data)
    assert (out/'phase5_metadata.json').exists()
    with pytest.raises(FileExistsError):
        main()
