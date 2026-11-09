#!/usr/bin/env python3
"""Major 7 final same-CPU comparison using previously recorded raw-MB grid.

Reads completed Stage-2 sweeps and independently re-times selected raw MB,
existing CPU float64 Chebyshev/Newton/Bisection and fixed-term series on the
*same generated input arrays*. Nothing is inferred from GPU timings.

The MB branch uses the published-paper sign and a full-active-domain extension
of the MATLAB source. It is not a certified floating-point implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from revision.phase6.mb_port import MBPort, constants, positive_root_bisect, series_normalized
from revision.phase6.scripts.benchmark_mb_cpu import inputs_phase5, elapsed_ms


P_DEFAULT = "0.3333333333333333,0.4,0.75,0.9,0.99,0.999"
CHEB_DEGREE = {1/3: 6, 0.4: 6, 0.75: 14, 0.9: 24, 0.99: 54, 0.999: 128}
NEWTON_STEPS = {1/3: 3, 0.4: 3, 0.75: 4, 0.9: 4, 0.99: 5, 0.999: 4}


def asbool(val):
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


def select_raw_mb(rows):
    """Prefer less quadrature work then fewer series terms, not noisy best time.

    On failure return smallest measured complex error, always flag unpassed.
    """
    usable = []
    for row in rows:
        try:
            if not asbool(row.get("all_finite")) or not math.isfinite(float(row["max_complex_normalized_error"])):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        usable.append(row)
    passing = [r for r in usable if asbool(r.get("meets_accuracy_target"))]
    if passing:
        selected = min(passing, key=lambda r: (int(r["nt"]), int(r["N"]), float(r["T"])))
        return selected, "minimum_grid_passing"
    if usable:
        selected = min(usable, key=lambda r: float(r["max_complex_normalized_error"]))
        return selected, "best_error_unpassed"
    return None, "no_usable_candidate"


def load_stage2(paths, expected_p, n, lam, seed, tol, expected_shas):
    grouped = {float(p): [] for p in expected_p}
    input_meta = []
    for folder in paths:
        folder = Path(folder)
        grid = folder / "mb_raw_contour_grid.csv"
        md = folder / "mb_raw_contour_metadata.json"
        if not (grid.is_file() and md.is_file()):
            raise FileNotFoundError(f"Missing Stage-2 grid or metadata in {folder}")
        metadata = json.loads(md.read_text())
        for key, expected in (("n", n), ("lam", lam), ("seed", seed), ("tol", tol)):
            val = metadata.get(key)
            if isinstance(expected, float):
                if val is None or not math.isclose(float(val), expected, rel_tol=0, abs_tol=1e-14):
                    raise ValueError(f"{folder}: metadata {key}={val} != {expected}")
            elif val != expected:
                raise ValueError(f"{folder}: metadata {key}={val} != {expected}")
        for p in metadata.get("p_grid", []):
            key = format(float(p), ".17g")
            digest = metadata.get("input_shas", {}).get(key)
            if digest is None:
                raise ValueError(f"{folder}: missing input hash for p={key}")
            for q, expected_sha in expected_shas.items():
                if math.isclose(float(p), q, rel_tol=0, abs_tol=1e-14) and digest != expected_sha:
                    raise ValueError(f"{folder}: inconsistent input SHA for p={p}")
        with grid.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for p in expected_p:
                    if math.isclose(float(row["p"]), p, rel_tol=0, abs_tol=1e-14):
                        grouped[p].append(row)
                        break
        input_meta.append({"path": str(folder), "metadata": metadata})
    if any(not v for v in grouped.values()):
        missing = [p for p, vals in grouped.items() if not vals]
        raise ValueError(f"Stage-2 sweeps missing p(s): {missing}")
    return grouped, input_meta


def stats(y, reference, approx, p, lam, *, complex_candidate=None):
    c = constants(p, lam)
    a = np.asarray(approx, dtype=np.float64)
    zz = np.asarray(a if complex_candidate is None else complex_candidate, dtype=np.complex128)
    denom = np.maximum(np.abs(y), c.tau)
    err = np.abs(zz-reference)/denom
    imag = np.abs(zz.imag)/denom
    aa = np.abs(y)>c.tau
    x = np.abs(a[aa]); ya = np.abs(y[aa])
    resid = np.abs(x - ya + lam*p*np.maximum(x,1e-300)**(p-1))
    return {
        "max_complex_normalized_error": float(np.max(err)),
        "max_real_normalized_error": float(np.max(np.abs(a-reference)/denom)),
        "max_normalized_imag": float(np.max(imag)),
        "max_active_stationarity_residual": float(np.max(resid)) if resid.size else 0.,
        "support_mismatch_count": int(np.count_nonzero((a!=0)!=(reference!=0))),
        "all_finite": bool(np.all(np.isfinite(zz)) and np.all(np.isfinite(a))),
    }


def writecsv(path, records):
    if not records:
        raise ValueError("No rows")
    keys = list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(records)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _candidate(p, candidates):
    return next((v for k,v in candidates.items() if math.isclose(k,p,rel_tol=0,abs_tol=1e-12)), None)


def run(args):
    from fast_lp_prox.revision_bench import make_preloaded_prox, fit_degree_family
    import torch

    pvals = [float(v) for v in args.p.split(",")]
    if len(pvals) != len(set(pvals)) or not all(0<p<1 for p in pvals):
        raise ValueError("Bad p grid")
    if args.n < 32 or args.repeats<1 or args.warmup<0 or not (0<args.tol<1):
        raise ValueError("Bad timing or accuracy parameters")
    torch.set_num_threads(args.torch_threads)
    out = Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    prepared={}
    hashes={}
    for p in pvals:
        y = inputs_phase5(n=args.n,p=p,lam=args.lam,seed=args.seed)
        prepared[p]=y
        hashes[p]=hashlib.sha256(y.tobytes()).hexdigest()
    grids, manifests=load_stage2(args.stage2, pvals,args.n,args.lam,args.seed,args.tol,hashes)
    allrows=[]
    by_family=[]
    metadata={
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "purpose":"Same-CPU raw MB versus comparators on identical float64 inputs",
        "source":"arXiv:2606.31951v2, paper sign; independently reimplemented MATLAB algorithm",
        "git_commit":_git_commit(), "python":platform.python_version(),
        "platform":platform.platform(), "numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__,
        "torch_threads":torch.get_num_threads(), "n":args.n,"seed":args.seed,"lam":args.lam,
        "accuracy_target":args.tol, "imag_tolerance":args.imag_tol,
        "warmup":args.warmup,"repeats":args.repeats,"p":pvals,
        "input_shas":{format(p,".17g"):v for p,v in hashes.items()},
        "stage2_directories":[str(v) for v in args.stage2],
        "stage2_metadata":manifests,
        "notes":[
            "MB raw estimates and real baselines independently timed in this session",
            "MB gamma contour setup and Chebyshev fit are offline and measured separately",
            "The tested CPU implementations use SciPy/NumPy for raw MB/series and PyTorch for legacy baselines",
            "MB uses source_R=False (full active domain), not source MATLAB default R restriction",
            "Numerical test pass is not a validated floating-point certificate",
            "Grid selection based on completed Stage-2 data, so timings are conditional on this finite grid",
        ],
    }
    for p in pvals:
        y=prepared[p]
        reference=positive_root_bisect(y,p=p,lam=args.lam)
        R=max(.9,(1+constants(p,args.lam).qp)*.5)*constants(p,args.lam).ustar
        choice, reason=select_raw_mb(grids[p])
        assert choice is not None
        N=int(choice["N"]); T=float(choice["T"]); nt=int(choice["nt"])
        print(f"\np={p:.6g}: Stage-2 raw MB N={N}, T={T:g}, nt={nt}, selection={reason}",flush=True)
        t0=time.perf_counter()
        mb=MBPort(p=p,lam=args.lam,n=N,T=T,nt=nt,mode="paper",R=R,source_R=False,chunk=args.chunk)
        setup_ms=1000*(time.perf_counter()-t0)
        z=mb(y)
        mbstat=stats(y,reference,z.x,p,args.lam,complex_candidate=z.x_complex)
        if not math.isclose(float(choice["max_complex_normalized_error"]),mbstat["max_complex_normalized_error"],rel_tol=1e-5,abs_tol=1e-11):
            raise AssertionError(f"p={p}: Stage-2 and rerun MB accuracy differ: {choice['max_complex_normalized_error']} vs {mbstat['max_complex_normalized_error']}")
        tim,raw=elapsed_ms(lambda:mb(y),args.warmup,args.repeats)
        mbrow={"p":p,"family":"paper_MB_no_fallback","configuration":f"N{N}_T{T:g}_nt{nt}",
            "source_selection":reason,"offline_setup_ms":setup_ms,"median_ms":tim,
            "timing_repeats_raw_ms":json.dumps(raw),"input_sha256":hashes[p],
            "source_stage2_passed":asbool(choice["meets_accuracy_target"]),
            **mbstat}
        mbrow["meets_accuracy_target"]=bool(mbstat["all_finite"] and mbstat["support_mismatch_count"]==0 and mbstat["max_complex_normalized_error"]<=args.tol and mbstat["max_normalized_imag"]<=args.imag_tol)
        allrows.append(mbrow)
        print(f"  MB raw: norm={mbstat['max_complex_normalized_error']:.3g}, imag={mbstat['max_normalized_imag']:.3g}, time={tim:.3f}ms, PASS={mbrow['meets_accuracy_target']}",flush=True)

        dtype=torch.float64
        device=torch.device("cpu")
        yt=torch.as_tensor(y,dtype=dtype,device=device)
        cdegree=_candidate(p,CHEB_DEGREE)
        nsteps=_candidate(p,NEWTON_STEPS)
        if cdegree is None or nsteps is None:
            raise ValueError(f"Missing Chebyshev/Newton configuration for p={p}")
        t0=time.perf_counter()
        fitted=fit_degree_family(p,[cdegree],n_fit=args.fit_n,n_validation=args.val_n)
        fit_ms=1000*(time.perf_counter()-t0)
        for family, work, coeff in [
            ("chebyshev",cdegree,fitted.coefficients[:cdegree+1]),
            ("newton",nsteps,None),
            ("bisection",20,None),
        ]:
            backend=make_preloaded_prox(p=p,lam=args.lam,family=family,device=device,dtype=dtype,work=work,coefficients=coeff)
            with torch.no_grad():
                result=backend(yt).detach().cpu().numpy()
                stat=stats(y,reference,result,p,args.lam)
                time_ms,raw_ms=elapsed_ms(lambda:backend(yt),args.warmup,args.repeats)
            row={"p":p,"family":family,"configuration":f"D{work}" if family=="chebyshev" else f"K{work}",
                 "offline_setup_ms":fit_ms if family=="chebyshev" else 0.,"median_ms":time_ms,
                 "timing_repeats_raw_ms":json.dumps(raw_ms),"input_sha256":hashes[p],**stat}
            row["meets_accuracy_target"]=bool(stat["all_finite"] and stat["support_mismatch_count"]==0 and stat["max_complex_normalized_error"]<=args.tol)
            allrows.append(row)
            print(f"  {family:10} norm={stat['max_complex_normalized_error']:.3g} time={time_ms:.3f}ms PASS={row['meets_accuracy_target']}",flush=True)
        active=(np.abs(y)>constants(p,args.lam).tau)
        def evaluate_series(order):
            x=np.zeros_like(y)
            a=np.abs(y[active]);eps=args.lam*p*np.power(a,p-2.)
            x[active]=np.sign(y[active])*a*series_normalized(eps,p=p,n=order)
            return x
        for order in args.series_terms:
            res=evaluate_series(order)
            stat=stats(y,reference,res,p,args.lam)
            time_ms,raw_ms=elapsed_ms(lambda:evaluate_series(order),args.warmup,args.repeats)
            row={"p":p,"family":"fixed_series","configuration":f"N{order}",
                 "offline_setup_ms":0.,"median_ms":time_ms,
                 "timing_repeats_raw_ms":json.dumps(raw_ms),"input_sha256":hashes[p],**stat}
            row["meets_accuracy_target"]=bool(stat["all_finite"] and stat["support_mismatch_count"]==0 and stat["max_complex_normalized_error"]<=args.tol)
            allrows.append(row)
        for family in ("paper_MB_no_fallback","chebyshev","newton","bisection","fixed_series"):
            candidates=[r for r in allrows if math.isclose(r["p"],p,abs_tol=1e-14) and r["family"]==family]
            passing=[r for r in candidates if r["meets_accuracy_target"]]
            if passing:
                selected=min(passing,key=lambda r:r["median_ms"])
                selected=dict(selected,table_status="accuracy_pass")
            else:
                selected=min(candidates,key=lambda r:r["max_complex_normalized_error"])
                selected=dict(selected,table_status="NO_PASSING_CONFIGURATION")
            by_family.append(selected)
    writecsv(out/"major7_cpu_all_methods.csv",allrows)
    writecsv(out/"major7_cpu_accuracy_matched.csv",by_family)
    (out/"major7_cpu_metadata.json").write_text(json.dumps(metadata,indent=2,default=str,allow_nan=False))
    print("\nFinal comparison",flush=True)
    for row in by_family:
        print(f" p={row['p']:5.3g} {row['family']:21} {row['configuration']:17} E={row['max_complex_normalized_error']:.3g} t={row['median_ms']:9.3f}ms {row['table_status']}")
    print(f"\nSaved to {out.resolve()}")
    return allrows,by_family


def parser():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--stage2",nargs="+",default=["results/reviewer2_major7_mb_stage2","results/reviewer2_major7_mb_stage2_longT"])
    pa.add_argument("--output-dir",default="results/reviewer2_major7_mb_stage3_final")
    pa.add_argument("--p",default=P_DEFAULT)
    pa.add_argument("--lam",type=float,default=1.)
    pa.add_argument("--seed",type=int,default=0)
    pa.add_argument("--n",type=int,default=128)
    pa.add_argument("--tol",type=float,default=1e-6)
    pa.add_argument("--imag-tol",type=float,default=1e-6)
    pa.add_argument("--fit-n",type=int,default=8192)
    pa.add_argument("--val-n",type=int,default=4001)
    pa.add_argument("--chunk",type=int,default=32)
    pa.add_argument("--torch-threads",type=int,default=1)
    pa.add_argument("--series-terms",type=lambda s:[int(v) for v in s.split(",")],default=[16,32,64,128,256])
    pa.add_argument("--warmup",type=int,default=3)
    pa.add_argument("--repeats",type=int,default=9)
    return pa

if __name__ == "__main__":
    run(parser().parse_args())
