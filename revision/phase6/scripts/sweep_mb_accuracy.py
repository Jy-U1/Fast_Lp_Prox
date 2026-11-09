#!/usr/bin/env python3
"""Accuracy-cost grid for *raw*, paper-consistent Mellin--Barnes evaluation.

This extends the fixed-N,T functionality pilot without replacing the previous
benchmark or changing archived Phase 4/5 results.  SciPy CPU float64 only.
The raw MB candidate must meet the tolerance *without any fallback*.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from revision.phase6.mb_port import MBPort, constants, positive_root_bisect
from revision.phase6.scripts.benchmark_mb_cpu import inputs_phase5, elapsed_ms


def parse_grid(s: str, typ, *, positive: bool=True):
    try:
        vals = [typ(v.strip()) for v in s.split(",")]
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid grid {s!r}") from e
    if not vals or len(vals) != len(set(vals)) or any(not math.isfinite(v) or (positive and v<=0) for v in vals):
        raise ValueError(f"invalid grid {s!r}")
    return sorted(vals)


def _csv(path: Path, records: list[dict], *, fields: list[str]):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)


def run(args):
    pgrid=parse_grid(args.p,float)
    ngrid=parse_grid(args.N_grid,int)
    Tgrid=parse_grid(args.T_grid,float)
    if min(ngrid)<1 or max(ngrid)>2048 or args.n<32 or args.dt<=0 or not (0<args.tol<1):
        raise ValueError('invalid parameter range')
    if args.repeats<1 or args.warmup<0 or args.max_panels<16:
        raise ValueError('invalid timing or panel parameters')
    od=Path(args.output_dir);od.mkdir(parents=True, exist_ok=True)
    entries=[];best=[]
    metadata={'generated_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'raw MB contour accuracy matching (no bisection fallback)',
        'source':'arXiv:2606.31951v2 Sec 5.2, paper-consistent sign',
        'independent_matlab_equivalence_test':False,
        'platform':platform.platform(),'python':platform.python_version(),
        'numpy':np.__version__,'scipy':scipy.__version__,
        'p_grid':pgrid,'N_grid':ngrid,'T_grid':Tgrid,'n':args.n,
        'lam':args.lam,'seed':args.seed,'tol':args.tol,'imag_tol':args.imag_tol,
        'dt_max':args.dt,'max_panels':args.max_panels,'warmup':args.warmup,
        'repeats':args.repeats,'input_shas':{},
        'timing':'median CPU online evaluation of n inputs; MB gamma contour setup excluded',
        'caution':'Nonvalidated trapezoid; numerical pass is not formal certification.'}
    cols=['p','N','T','nt','dt_actual','n_inputs','R_over_ustar',
        'setup_ms','median_ms','max_complex_normalized_error',
        'max_real_normalized_error','max_normalized_imag',
        'max_active_stationarity_residual','max_prox_objective_gap',
        'support_mismatch_count','domain_rejected','all_finite','meets_accuracy_target',
        'timing_repeats_raw_ms','error']
    for p in pgrid:
        c=constants(p,args.lam)
        y=inputs_phase5(n=args.n,p=p,lam=args.lam,seed=args.seed)
        ref=positive_root_bisect(y,p=p,lam=args.lam)
        denom=np.maximum(abs(y),c.tau)
        metadata['input_shas'][format(p,'.17g')]=hashlib.sha256(y.tobytes()).hexdigest()
        R=max(.9,(1+c.qp)*.5)*c.ustar
        print(f'\np={p:.6g} {args.n} inputs R/u*={R/c.ustar:.9f} input_SHA256={metadata["input_shas"][format(p,".17g")][:12]}',flush=True)
        per=[]
        for N in ngrid:
            for T in Tgrid:
                nt=math.ceil(2*T/args.dt)
                if nt%2:nt+=1
                row={'p':p,'N':N,'T':T,'nt':nt,'dt_actual':2*T/nt,'n_inputs':args.n,
                     'R_over_ustar':R/c.ustar}
                if nt>args.max_panels:
                    row.update({'meets_accuracy_target':False,'error':f'panel cap: {nt}>{args.max_panels}'})
                    entries.append(row);per.append(row)
                    continue
                try:
                    t0=time.perf_counter()
                    b=MBPort(p=p,lam=args.lam,n=N,T=T,nt=nt,
                             mode='paper',R=R,source_R=False,chunk=args.chunk)
                    row['setup_ms']=1000*(time.perf_counter()-t0)
                    z=b(y)
                    m,times=elapsed_ms(lambda:b(y),args.warmup,args.repeats)
                    x=z.x
                    norm_err=np.abs(z.x_complex-ref)/denom
                    nreal=np.abs(x-ref)/denom
                    imag=np.abs(z.x_complex.imag)/denom
                    active=abs(y)>c.tau
                    xa=abs(x[active]);ya=abs(y[active]);
                    res=abs(xa-ya+args.lam*p*np.maximum(xa,1e-300)**(p-1))
                    obj=.5*(x-y)**2+args.lam*abs(x)**p
                    objref=.5*(ref-y)**2+args.lam*abs(ref)**p
                    err=float(norm_err.max());eim=float(imag.max())
                    finite=bool(np.all(np.isfinite(z.x_complex)))
                    supp=int(np.count_nonzero((x!=0)!=(ref!=0)))
                    passes=bool(finite and supp==0 and err<=args.tol and eim<=args.imag_tol)
                    row.update({'median_ms':m,'max_complex_normalized_error':err,
                        'max_real_normalized_error':float(nreal.max()),
                        'max_normalized_imag':eim,'max_active_stationarity_residual':float(res.max()),
                        'max_prox_objective_gap':float(np.max(obj-objref)),
                        'support_mismatch_count':supp,'domain_rejected':z.rejected_count,
                        'all_finite':finite,'meets_accuracy_target':passes,
                        'timing_repeats_raw_ms':json.dumps(times),'error':''})
                    print(f' N={N:3} T={T:6g} nt={nt:6d} E={err:.3g} imag={eim:.3g} time={m:.3f}ms PASS={passes}',flush=True)
                except (ValueError, ArithmeticError, FloatingPointError, OverflowError) as e:
                    row.update({'meets_accuracy_target':False,'error':repr(e)})
                    print(f' N={N:3} T={T:6g} ERROR={e}',flush=True)
                entries.append(row);per.append(row)
        passing=[v for v in per if v.get('meets_accuracy_target')]
        if passing:
            selected=min(passing,key=lambda v:v['median_ms'])
            selection='fastest_passing_raw_MB'
        else:
            usable=[v for v in per if v.get('all_finite')]
            selected=min(usable,key=lambda v:v['max_complex_normalized_error']) if usable else None
            selection='no_passing_config_best_error_only'
        if selected:
            best.append(dict(selected, selection_status=selection))
            print(f' SELECTED {selection}: N={selected["N"]},T={selected["T"]},E={selected.get("max_complex_normalized_error",float("nan")):.3g}',flush=True)
    _csv(od/'mb_raw_contour_grid.csv',entries,fields=cols)
    _csv(od/'mb_raw_contour_selected.csv',best,fields=cols+['selection_status'])
    metadata['completed_cases']=len(pgrid)
    (od/'mb_raw_contour_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    return entries,best


def parser():
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--p',default='0.3333333333333333,0.4,0.75,0.9,0.99,0.999')
    q.add_argument('--lam',type=float,default=1.)
    q.add_argument('--n',type=int,default=128)
    q.add_argument('--seed',type=int,default=0)
    q.add_argument('--N-grid',dest='N_grid',default='3,6,12,24,48')
    q.add_argument('--T-grid',dest='T_grid',default='60,240,960')
    q.add_argument('--dt',type=float,default=.03)
    q.add_argument('--max-panels',type=int,default=100000)
    q.add_argument('--chunk',type=int,default=32)
    q.add_argument('--tol',type=float,default=1e-6)
    q.add_argument('--imag-tol',type=float,default=1e-6)
    q.add_argument('--warmup',type=int,default=1)
    q.add_argument('--repeats',type=int,default=3)
    q.add_argument('--output-dir',default='results/reviewer2_major7_mb_cpu_grid')
    return q

if __name__=='__main__':run(parser().parse_args())
