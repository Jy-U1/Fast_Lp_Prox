#!/usr/bin/env python3
"""Reviewer-2 Major 7: released MATLAB MB evaluator vs paper formula on CPU.

CPU-only scipy complex special functions.  This study is NOT the already
archived Phase-5 million-input CUDA study.  Never merge its timings without
stating dtype/device/batch size and precomputation conventions.

Run: python -m revision.phase6.scripts.benchmark_mb_cpu --p 0.6666666666666666
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Callable

import numpy as np
import scipy

from revision.phase6.mb_port import MBPort, constants, positive_root_bisect, series_normalized, choose_matlab_parameters


def git_sha():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return None


def inputs_phase5(*,n:int,p:float,lam:float,seed:int,active_fraction:float=.90,ymax_factor:float=10.) -> np.ndarray:
    """The NumPy part of revision.phase5.scripts.generate_shared_inputs.

    Preserves Phase5 generator order and RNG; float64 CPU output is exact
    versus the torch.tensor(vals, dtype=torch.float64) pathway.
    """
    if n<32 or not 0<=active_fraction<=1 or ymax_factor<=1: raise ValueError("Bad generator options")
    c=constants(p,lam);rng=np.random.default_rng(seed)
    na=int(round(n*active_fraction));ni=n-na
    inactive=c.tau*rng.uniform(0,.999,ni)
    if ni:
        k=min(ni,max(1,ni//20))
        delta=np.logspace(-8,-3,k)
        inactive[:k]=c.tau*(1.-delta)
    if na:
        k=min(na,max(1,na//20))
        active=c.tau*np.exp(rng.uniform(np.log1p(1e-3),math.log(ymax_factor),na))
        active[:k]=c.tau*(1.+np.logspace(-8,-3,k))
    else:
        active=np.empty(0,dtype=np.float64)
    vals=np.concatenate((inactive,active))
    vals*=rng.choice([-1.,1.],size=n)
    rng.shuffle(vals)
    return vals


def elapsed_ms(f:Callable[[],object],warmup:int,repeats:int) -> tuple[float,list[float]]:
    for _ in range(warmup):f()
    times=[]
    for _ in range(repeats):
        t0=time.perf_counter();f();times.append(1000*(time.perf_counter()-t0))
    return float(np.median(times)),times


def _stats(y,reference,approx, *,c, rawcomplex=None, statuses=None):
    zz=np.asarray(rawcomplex if rawcomplex is not None else approx,dtype=np.complex128)
    x=np.asarray(approx,dtype=np.float64)
    denom=np.maximum(np.abs(y),c.tau)
    absrel=np.abs(x-reference)/denom
    complexrel=np.abs(zz-reference)/denom
    support=np.count_nonzero((x!=0)!=(reference!=0))
    active=np.abs(y)>c.tau
    xa=np.abs(x[active]);ya=np.abs(y[active])
    station=np.abs(xa-ya+c.lam*c.p*np.maximum(xa,1e-300)**(c.p-1))
    obj=0.5*(x-y)**2+c.lam*np.abs(x)**c.p
    objref=0.5*(reference-y)**2+c.lam*np.abs(reference)**c.p
    return dict(max_norm_error_real=float(np.max(absrel)),
                max_norm_error_complex=float(np.max(complexrel)),
                mean_norm_error_real=float(np.mean(absrel)),
                max_norm_imag=float(np.max(np.abs(np.imag(zz))/denom)),
                max_active_stationarity_residual=float(np.max(station)) if len(station) else 0.,
                max_prox_objective_gap=float(np.max(obj-objref)),
                support_mismatch_count=int(support),
                all_finite=bool(np.all(np.isfinite(x)) and np.all(np.isfinite(zz))),
                n_domain_rejected=int(np.sum(statuses=="rejected_R")) if statuses is not None else 0)


def run(args):
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    pvals=[float(v.strip()) for v in args.p.split(",")]
    out_rows=[]; metadata={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "git_commit":git_sha(),"source_repository":"https://github.com/lshen03/Lp-Proximity-Operator",
        "source_matlab_file":"xplus_lp_hybrid.m","source_paper":"arXiv:2606.31951v2, Sec. 5.2",
        "cpu":platform.processor(),"platform":platform.platform(),"python":platform.python_version(),
        "numpy":np.__version__,"scipy":scipy.__version__,"float_dtype":"float64",
        "quadrature":"fixed uniform trapezoid, not validated",
        "raw_matlab_sign":"+GammaRatio correction, as public MATLAB source",
        "paper_v2_sign":"-GammaRatio correction, as paper H(s) definition",
        "empirical_bounds_are_certificates":False,
        "warmup":args.warmup,"repeats":args.repeats,"n":args.n,
        "N":args.N,"T":args.T,"nt":args.nt,"seed":args.seed,
        "lam":args.lam,"p":pvals,"active_fraction":args.active_fraction,
        "ymax_factor":args.ymax_factor,"accuracy_target":args.tol,
        "matlab_parameters_are_fixed":not args.matlab_autoselect,"enable_bisection_fallback":True,
        "matlab_autoselect":args.matlab_autoselect,"selection_eta":args.selection_eta,
        "selection_maxN":args.selection_maxN,"selection_maxT":args.selection_maxT,
        "provenance_note":"Original MATLAB has no declared license; code is an independent reimplementation",
        "case_input_shas":{}}
    for p in pvals:
        c=constants(p,args.lam)
        y=inputs_phase5(n=args.n,p=p,lam=args.lam,seed=args.seed,
                        active_fraction=args.active_fraction,ymax_factor=args.ymax_factor)
        reference=positive_root_bisect(y,p=p,lam=args.lam)
        digest=hashlib.sha256(y.tobytes()).hexdigest()
        metadata['case_input_shas'][format(p,'.17g')]=digest
        R=(.9 if args.R_policy=='matlab-default' else max(.9,(1+c.qp)*.5))*c.ustar
        print(f"p={p:.6g} N={args.N} T={args.T:g} nt={args.nt} R/u*={R/c.ustar:.9f} max_eps/u*={c.qp:.9f}",flush=True)
        selection_start=time.perf_counter()
        if args.matlab_autoselect:
            choice=choose_matlab_parameters(p=p,R=R,eta=args.selection_eta,
                  maxN=args.selection_maxN,maxT=args.selection_maxT,
                  envelope_samples=args.selection_samples)
            selectedN, selectedT=choice.N, choice.T
            print(f"  MATLAB-style N/T selection: N={selectedN} T={selectedT:.3g}; "
                  f"sampled series={choice.series_estimate:.3g} "
                  f"sampled MB={choice.mb_estimate:.3g}; "
                  f"empirical targets={choice.meets_empirical_series_target}/{choice.meets_empirical_mb_target}",
                  flush=True)
        else:
            choice=None
            selectedN,selectedT=args.N,args.T
        selection_ms=1000*(time.perf_counter()-selection_start)
        # Each source mode uses the *same* contour grid, sample inputs,
        # and independent double bisection reference.
        for mode in ('matlab','paper'):
            tprep=time.perf_counter()
            backend=MBPort(p=p,lam=args.lam,n=selectedN,T=selectedT,nt=args.nt,
                           mode=mode,R=R,source_R=(args.R_policy=='matlab-default'),chunk=args.chunk)
            offline_ms=1000*(time.perf_counter()-tprep)
            rawname='mb_public_matlab_sign' if mode=='matlab' else 'mb_paper_v2_sign'
            for checked in (False,True) if mode=='paper' else (False,):
                key=rawname+('_residual_checked' if checked else '')
                def func():return backend(y,tol=args.tol if checked else None,
                                          imag_tol=args.imag_tol)
                try:
                    evalout=func()
                    m,vals=elapsed_ms(func,args.warmup,args.repeats)
                    met=_stats(y,reference,evalout.x,c=c,rawcomplex=(evalout.x_complex if not checked else None),statuses=evalout.statuses)
                    okay=met['max_norm_error_complex']<=args.tol and met['max_norm_imag']<=args.imag_tol and met['support_mismatch_count']==0 and met['all_finite']
                    out_rows.append(dict(p=p,method=key,N=selectedN,T=selectedT,nt=args.nt,
                                         R_over_ustar=R/c.ustar,offline_setup_ms=offline_ms,
                                         offline_parameter_selection_ms=selection_ms,
                                         empirical_series_estimate=(choice.series_estimate if choice else float('nan')),
                                         empirical_mb_estimate=(choice.mb_estimate if choice else float('nan')),
                                         empirical_targets_met=(choice.meets_empirical_series_target and choice.meets_empirical_mb_target if choice else None),
                                         median_ms=m,timing_repeats_raw_ms=json.dumps(vals),
                                         meets_accuracy_target=bool(okay),
                                         fallback_count=evalout.fallback_count,
                                         raw_candidate_max_norm_imag=evalout.max_imag_normalized,
                                         max_accepted_residual_indicator=evalout.max_accepted_residual_indicator,
                                         **met))
                    print(f"  {key:38} norm={met['max_norm_error_complex']:.3g} imag={met['max_norm_imag']:.3g} ms={m:.3f} fallback={evalout.fallback_count}",flush=True)
                except (ValueError,ArithmeticError,FloatingPointError) as exc:
                    out_rows.append(dict(p=p,method=key,N=selectedN,T=selectedT,nt=args.nt,
                                         R_over_ustar=R/c.ustar,offline_setup_ms=offline_ms,
                                         offline_parameter_selection_ms=selection_ms,
                                         median_ms=float('nan'),timing_repeats_raw_ms='[]',
                                         meets_accuracy_target=False,fallback_count=-1,
                                         error=str(exc),n_domain_rejected=int(np.count_nonzero((np.abs(y)>c.tau) & (args.lam*p*np.maximum(np.abs(y),c.tau)**(p-2) >R)))))
                    print(f"  {key:38} REJECTED: {exc}",flush=True)
        for sn in args.series_orders.split(','):
            N=int(sn)
            def series_eval():
                a=np.abs(y)
                result=np.zeros_like(a)
                active=a>c.tau
                eps=args.lam*p*a[active]**(p-2)
                result[active]=np.sign(y[active])*a[active]*series_normalized(eps,p=p,n=N)
                return result
            val=series_eval()
            m,times=elapsed_ms(series_eval,args.warmup,args.repeats)
            met=_stats(y,reference,val,c=c)
            out_rows.append(dict(p=p,method=f'fixed_series_N{N}',N=N,T=0.,nt=0,
                                 R_over_ustar=float('nan'),offline_setup_ms=float('nan'),
                                 median_ms=m,timing_repeats_raw_ms=json.dumps(times),
                                 fallback_count=0,meets_accuracy_target=bool(met['max_norm_error_complex']<=args.tol),
                                 **met))
        # Only numerical baselines on identical CPU dtype and identical inputs.
        # Bisection is 20 steps, not the 80-step reference. Newton uses the
        # existing project implementation if its package is installed.
        def bisection20():return positive_root_bisect(y,p=p,lam=args.lam,steps=20)
        b=bisection20();m,times=elapsed_ms(bisection20,args.warmup,args.repeats)
        met=_stats(y,reference,b,c=c)
        out_rows.append(dict(p=p,method='bisection_K20_cpu_numpy',N=20,T=0.,nt=0,
                             R_over_ustar=float('nan'),offline_setup_ms=0.,median_ms=m,
                             timing_repeats_raw_ms=json.dumps(times),fallback_count=0,
                             meets_accuracy_target=bool(met['max_norm_error_complex']<=args.tol),**met))
    if args.include_phase5 != "no":
        try:
            import torch
            from fast_lp_prox.revision_bench import fit_degree_family, make_preloaded_prox
        except (ImportError, ModuleNotFoundError) as exc:
            if args.include_phase5 == "yes":
                raise RuntimeError("Phase 5 repository backends unavailable; run within a pip-installed Fast_Lp_Prox checkout") from exc
            print(f"Phase 5 CPU backends not installed, skipping: {exc}",flush=True)
            metadata["phase5_torch_backends_available"] = False
        else:
            metadata["phase5_torch_backends_available"] = True
            metadata["torch"] = torch.__version__
            chosen_degrees={1/3:6,.4:6,.75:14,.9:24,.99:54,.999:128}
            chosen_iters={1/3:3,.4:3,.75:4,.9:4,.99:5,.999:4}
            for p in pvals:
                c=constants(p,args.lam)
                y=inputs_phase5(n=args.n,p=p,lam=args.lam,seed=args.seed,
                                active_fraction=args.active_fraction,ymax_factor=args.ymax_factor)
                reference=positive_root_bisect(y,p=p,lam=args.lam)
                yt=torch.as_tensor(y,dtype=torch.float64,device='cpu')
                degree=next((v for k,v in chosen_degrees.items() if abs(p-k)<1e-12), 14)
                nit=next((v for k,v in chosen_iters.items() if abs(p-k)<1e-12), 4)
                offline_start=time.perf_counter()
                fitted=fit_degree_family(p,[degree],n_fit=8192,n_validation=4001)
                cheb_offline=1000*(time.perf_counter()-offline_start)
                for family,work,coeff,setupms in (
                    ('chebyshev',degree,fitted.coefficients[:degree+1],cheb_offline),
                    ('newton',nit,None,0.),
                    ('bisection',20,None,0.),
                ):
                    backend=make_preloaded_prox(p=p,lam=args.lam,family=family,
                        device=torch.device('cpu'),dtype=torch.float64,
                        work=work,coefficients=coeff)
                    with torch.no_grad():
                        x=backend(yt).detach().cpu().numpy()
                        tms,reps=elapsed_ms(lambda:backend(yt),args.warmup,args.repeats)
                    met=_stats(y,reference,x,c=c)
                    out_rows.append(dict(p=p,method=f'phase5_{family}_K{work}' if family!='chebyshev' else f'phase5_chebyshev_D{work}',
                        N=work,T=0.,nt=0,R_over_ustar=float('nan'),
                        offline_setup_ms=setupms,median_ms=tms,
                        timing_repeats_raw_ms=json.dumps(reps),fallback_count=0,
                        meets_accuracy_target=bool(met['max_norm_error_complex']<=args.tol),**met))
    cols=[]
    for row in out_rows:
        for k in row:
            if k not in cols:cols.append(k)
    fname=out/'mb_major7_cpu_results.csv'
    with fname.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=cols);wr.writeheader();wr.writerows(out_rows)
    (out/'mb_major7_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    print(f"Saved: {fname} ({len(out_rows)} records)")
    return out_rows


def parser():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--p',default='0.3333333333333333,0.4,0.75,0.9,0.99,0.999')
    pa.add_argument('--lam',type=float,default=1.)
    pa.add_argument('--n',type=int,default=128)
    pa.add_argument('--seed',type=int,default=0)
    pa.add_argument('--active-fraction',type=float,default=.9)
    pa.add_argument('--ymax-factor',type=float,default=10.)
    pa.add_argument('--N',type=int,default=3)
    pa.add_argument('--T',type=float,default=60.)
    pa.add_argument('--matlab-autoselect',action='store_true',help='Use sampled N/T rules from MATLAB, not certified')
    pa.add_argument('--selection-eta',type=float,default=1e-8)
    pa.add_argument('--selection-maxN',type=int,default=200)
    pa.add_argument('--selection-maxT',type=float,default=50000.)
    pa.add_argument('--selection-samples',type=int,default=4000)
    pa.add_argument('--nt',type=int,default=4000)
    pa.add_argument('--chunk',type=int,default=32)
    pa.add_argument('--R-policy',choices=['full-active','matlab-default'],default='full-active')
    pa.add_argument('--tol',type=float,default=1e-6)
    pa.add_argument('--imag-tol',type=float,default=1e-6)
    pa.add_argument('--series-orders',default='16,64')
    pa.add_argument('--include-phase5',choices=['auto','yes','no'],default='auto')
    pa.add_argument('--warmup',type=int,default=1)
    pa.add_argument('--repeats',type=int,default=3)
    pa.add_argument('--output-dir',default='results/reviewer2_major7_mb_cpu_smoke')
    return pa

if __name__=='__main__':
    run(parser().parse_args())
