"""Combine Phase-5 experiment CSVs and select fastest *passing tested* option.

Unlike the archived Phase-1 results, these are NEW sample/timing trials and
must never be silently concatenated across distinct dtype/GPU/input generators.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(paths: list[Path]):
    rows=[]
    provenance=[]
    config=None
    for f in paths:
        if not f.is_file():
            raise FileNotFoundError(f)
        meta=f.parent/'phase5_metadata.json'
        if not meta.is_file():
            raise FileNotFoundError(meta)
        info=json.loads(meta.read_text())
        setting=(info['gpu'],info['torch'],info['args']['dtype'],
                 info['args']['n'],info['args']['active_fraction'],
                 info['args']['ymax_factor'],info['args']['seed'])
        if config is None: config=setting
        if setting!=config:
            raise ValueError('Cannot combine distinct hardware, precision or input protocols')
        with f.open(newline='') as handle:
            block=list(csv.DictReader(handle))
        if not block:
            raise ValueError(f'{f}: empty result')
        rows.extend(block)
        provenance.append({'input':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
                           'git_commit':info['git']['commit'],'input_tensor_sha256':info['input_sha256']})
    return rows,provenance


def fastest_passing(rows: list[dict]) -> list[dict]:
    groups={}
    for x in rows:
        key=(x['p'],x['family'])
        groups.setdefault(key,[]).append(x)
    chosen=[]
    for (p,family),items in sorted(groups.items(),key=lambda kv:(float(kv[0][0]),kv[0][1])):
        passing=[x for x in items if x['meets_accuracy_target']=='True' and x['all_finite']=='True'
                 and int(x['support_mismatch_count'])==0]
        if passing:
            win=min(passing,key=lambda x:float(x['median_ms']))
            chosen.append({'p':p,'family':family,'status':'PASS',
                           'configuration':win['configuration'],
                           'median_ms':win['median_ms'],'max_norm_error':win['max_norm_error'],
                           'max_active_stationarity_residual':win['max_active_stationarity_residual'],
                           'mean_prox_objective_gap':win['mean_prox_objective_gap'],
                           'fallback_count':win['fallback_count'],
                           'offline_family_seconds':win['offline_family_seconds']})
        else:
            chosen.append({'p':p,'family':family,'status':'NO_PASSING_CONFIGURATION',
                           'configuration':'','median_ms':'','max_norm_error':'',
                           'max_active_stationarity_residual':'','mean_prox_objective_gap':'',
                           'fallback_count':'','offline_family_seconds':''})
    return chosen


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csvs',nargs='+',type=Path)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f'Output exists: {args.output_dir}')
    records,provenance=read(args.csvs)
    selected=fastest_passing(records)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/'phase5_matched_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(selected[0]))
        writer.writeheader();writer.writerows(selected)
    (args.output_dir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Selected {len(selected)} family-by-p rows; passing {sum(x["status"]=="PASS" for x in selected)}')
    for s in selected:
        print(s['p'],s['family'],s['status'],s['configuration'],s['median_ms'])


if __name__=='__main__':
    main()
