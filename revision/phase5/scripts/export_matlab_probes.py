"""Export a small, same-input probe set for original MATLAB MB code from [24].

The MATLAB scalar routine is run SEPARATELY (not through PyTorch).
It is a different language/device, so the times are not claimed to be a
matched-implementation A100 throughput comparison.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from revision.phase5.backends import proximal_constants, reference_bisection


def export(path: Path, *, ps: tuple[float, ...], lam: float):
    if path.exists(): raise FileExistsError(path)
    factors=(1.000001,1.0001,1.01,1.25,3.0,10.0)
    result=[]
    for p in ps:
        c=proximal_constants(p,lam)
        for factor in factors:
            y=c.tau*factor
            ref=float(reference_bisection(torch.tensor([y],dtype=torch.float64),p=p,
                                          lam=lam,iterations=100)[0].item())
            result.append({'p':p,'lam':lam,'y':y,'factor':factor,'ref_x':ref,
                           'ref_semantics':'float64_100step_bisection'})
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=result[0]);w.writeheader();w.writerows(result)
    print('wrote',len(result),'scalar probes to',path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p-grid',default='0.3333333333333333,0.75,0.9')
    ap.add_argument('--lam',type=float,default=1.0)
    ap.add_argument('--output',type=Path,default=Path('results/phase5_matlab_probes.csv'))
    args=ap.parse_args()
    export(args.output,ps=tuple(map(float,args.p_grid.split(','))),lam=args.lam)


if __name__=='__main__':main()
