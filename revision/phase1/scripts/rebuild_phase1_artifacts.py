#!/usr/bin/env python3
"""Recompute Phase 1 derived tables, audit reports and figures from raw CSVs.

This script never modifies raw CSV files and does not depend on the prox solver.
Uses matplotlib (and the Python standard library); ready for Colab/local Python.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FORMAL = [
    ('p0333', '1/3', 1/3, 'revision_phase1_A100_p0333_1M.csv'),
    ('p0400', '2/5', 2/5, 'revision_phase1_A100_p0400_1M.csv'),
    ('p0750', '3/4', 3/4, 'revision_phase1_A100_p0750_1M.csv'),
    ('p0900', '0.9', .9, 'revision_phase1_A100_p0900_1M.csv'),
    ('p0990', '0.99', .99, 'revision_phase1_A100_p0990_1M.csv'),
    ('p0999', '0.999', .999, 'revision_phase1_A100_p0999_1M.csv'),
]
FAMILIES = ('chebyshev','newton','bisection')
COLORS = {'chebyshev':'#137C8B', 'newton':'#D36B2D', 'bisection':'#697586'}


def load_csv(path: Path):
    with path.open('r', newline='', encoding='utf-8') as f:
        reader=csv.DictReader(f)
        rows=list(reader)
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    return rows


def save_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def is_true(x):
    return str(x).strip().lower() in {'true','1','yes'}


def passed(r):
    return (is_true(r['all_finite'])
            and int(r['support_mismatch_count'])==0
            and float(r['max_norm_error']) <= float(r['accuracy_target'])
            and is_true(r['meets_accuracy_target']))


def f(r, k):
    return float(r[k]) if r.get(k) not in (None,'') else float('nan')


def filehash(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        while block:=handle.read(65536):
            h.update(block)
    return h.hexdigest()


def audit_formal(path: Path, rows: list[dict], p_value: float):
    r=rows[0]
    required={'family','configuration','median_ms','max_norm_error','std_ms',
              'all_finite','support_mismatch_count','meets_accuracy_target',
              'accuracy_target','n_timing','n_accuracy','git_commit','p',
              'device_name','repeats','warmup','dtype','reference_method',
              'max_reference_crosscheck_norm_error'}
    missing=required.difference(r)
    if missing:
        raise ValueError(f'{path}: missing fields {sorted(missing)}')
    if any(abs(float(x['p'])-p_value)>1e-12 for x in rows):
        raise ValueError(f'{path}: inconsistent p')
    identity=['git_commit','p','n_timing','n_accuracy','device_name','dtype',
              'repeats','warmup','accuracy_target','seed','ymax_factor',
              'active_ratio_requested','fit_n','validation_n','reference_method',
              'torch_version','numpy_version','python_version']
    invariant_err=[]
    for k in identity:
        vals={x[k] for x in rows if x.get(k)!=''}
        if len(vals)>1:
            invariant_err.append(k)
    if invariant_err:
        raise ValueError(f'{path}: inconsistent metadata fields {invariant_err}')
    if r['n_timing']!='1000000' or r['n_accuracy']!='1000000' or r['dtype']!='float32':
        raise ValueError(f'{path}: formal dataset is not n=1e6 float32 with same accuracy sample')
    if not all(x['repeats']=='30' and x['warmup']=='10' for x in rows):
        raise ValueError(f'{path}: unexpected formal timing protocol')
    expected_flags = [is_true(x['all_finite']) and f(x,'max_norm_error')<=f(x,'accuracy_target') for x in rows]
    actual_flags = [is_true(x['meets_accuracy_target']) for x in rows]
    if expected_flags!=actual_flags:
        raise ValueError(f'{path}: inconsistent CSV pass flag vs recorded criterion')
    if any(not math.isfinite(f(x,'median_ms')) or f(x,'median_ms')<=0 for x in rows):
        raise ValueError(f'{path}: nonpositive or invalid runtime')
    return {
        'n_configurations':len(rows),
        'n_pass_accuracy':sum(passed(x) for x in rows),
        'n_nonfinite':sum(not is_true(x['all_finite']) for x in rows),
        'total_support_mismatches':sum(int(x['support_mismatch_count']) for x in rows),
        'max_reference_crosscheck_norm_error':max(f(x,'max_reference_crosscheck_norm_error') for x in rows),
        'max_abs_active_residual':max(f(x,'max_active_stationarity_residual') for x in rows),
        'commit':r['git_commit'],
        'device':r['device_name'],
        'torch':r.get('torch_version',''),
        'numpy':r.get('numpy_version',''),
        'python':r.get('python_version',''),
    }


def do_plots(plots, best_by_p, all_rows):
    plt.rcParams.update({'font.size':9, 'axes.titlesize':11, 'axes.labelsize':10,
                         'legend.fontsize':9,'savefig.bbox':'tight'})
    xs=list(range(len(best_by_p)))
    ps=[r['p_label'] for r in best_by_p]

    fig,ax=plt.subplots(figsize=(8.1,4.7))
    for fam in FAMILIES:
        name={'chebyshev':'Chebyshev','newton':'Projected Newton','bisection':'Bisection'}[fam]
        ys=[float(r[f'{fam}_median_ms']) for r in best_by_p]
        ax.plot(xs,ys,marker='o',ms=6,lw=2,color=COLORS[fam],label=name)
    ax.set_xticks(xs, ps)
    ax.set_ylabel('Median online runtime (ms)')
    ax.set_xlabel('Power exponent p')
    ax.set_title(r'Fastest observed configuration meeting $E_{\infty}\leq 10^{-6}$')
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='upper left')
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(plots/f'best_runtime.{ext}',dpi=220)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.1,4.7))
    ys=[float(r['cheb_over_newton']) for r in best_by_p]
    ax.axhline(1.,color='#333333',lw=1,ls='--')
    ax.plot(xs,ys,color=COLORS['chebyshev'],marker='o',lw=2,ms=6)
    for x,y in zip(xs,ys):
        ax.annotate(f'{y:.2f}x',(x,y), xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xticks(xs,ps)
    ax.set_ylim(0, max(ys)*1.19)
    ax.set_ylabel('Chebyshev time / Newton time')
    ax.set_xlabel('Power exponent p')
    ax.set_title('Accuracy-matched GPU time ratio (observed)')
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(plots/f'cheb_newton_time_ratio.{ext}',dpi=220)
    plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(13.4,7.6))
    for axis,(tag, label, pvalue, fname) in zip(axes.flat,FORMAL):
        rows=all_rows[tag]
        for fam in FAMILIES:
            sub=sorted((r for r in rows if r['family']==fam),key=lambda z:f(z,'median_ms'))
            t=[f(r,'median_ms') for r in sub]
            e=[max(f(r,'max_norm_error'),1e-16) for r in sub]
            axis.plot(t,e,marker='.',lw=1.4,ms=5,color=COLORS[fam],label=fam.capitalize())
            elig=[r for r in sub if passed(r)]
            if elig:
                best=min(elig,key=lambda x:f(x,'median_ms'))
                axis.scatter([f(best,'median_ms')],[f(best,'max_norm_error')],
                             marker='*',s=125,color=COLORS[fam],edgecolor='black',linewidth=.3,zorder=4)
        axis.axhline(1e-6,ls='--',lw=.9,color='black',alpha=.65)
        axis.set_yscale('log')
        axis.set_title(f'p = {label}')
        axis.grid(alpha=.16)
        axis.set_xlabel('Online median time (ms)')
        axis.set_ylabel('Maximum normalized error')
    legend_handles,legend_labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(legend_handles,legend_labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,1.025),frameon=False)
    fig.suptitle(r'Accuracy--runtime tradeoff, $n=10^6$, float32, A100 (stars = fastest passing)',y=1.07,fontsize=12)
    fig.tight_layout(pad=1.5)
    for ext in ('png','pdf'):
        fig.savefig(plots/f'accuracy_runtime_six_panel.{ext}',dpi=220)
    plt.close(fig)


def main(root: Path):
    raw=root/'data'
    tab=root/'tables'
    plots=root/'figures'
    reports=root/'reports'
    for directory in (tab,plots,reports):directory.mkdir(parents=True,exist_ok=True)
    all_rows={}
    audit=[]
    bestrows=[]
    full=[]
    best_by_p=[]
    metadata=[]
    for tag, label, pvalue, filename in FORMAL:
        path=raw/'formal'/filename
        rows=load_csv(path)
        all_rows[tag]=rows
        check=audit_formal(path,rows,pvalue)
        audit.append({'p_label':label,'source_file':filename,**check})
        first=rows[0]
        b={'p_label':label, 'p_value':pvalue, 'n_timing':int(first['n_timing']),
           'device_name':first['device_name'], 'torch_version':first['torch_version'],
           'accuracy_target':first['accuracy_target'], 'git_commit':first['git_commit']}
        for x in rows:
            full.append({'p_label':label,'source_file':filename,**x})
        for fam in FAMILIES:
            candidates=[x for x in rows if x['family']==fam and passed(x)]
            if not candidates:
                raise ValueError(f'No passing {fam} in {filename}')
            row=min(candidates,key=lambda x:f(x,'median_ms'))
            bestrows.append({
                'p_label':label,'p_value':pvalue,'family':fam,
                'configuration':row['configuration'], 'degree':row['degree'],
                'iterations':row['iterations'],'median_ms':row['median_ms'],
                'std_ms':row['std_ms'], 'max_norm_error':row['max_norm_error'],
                'max_active_stationarity_residual':row['max_active_stationarity_residual'],
                'sampled_float64_branch_error':row['sampled_float64_branch_error'],
                'support_mismatch_count':row['support_mismatch_count'],
                'source_file':filename,'git_commit':row['git_commit'],
                'n_timing':row['n_timing'], 'repeats':row['repeats'],
            })
            b[f'{fam}_configuration']=row['configuration']
            b[f'{fam}_median_ms']=row['median_ms']
            b[f'{fam}_max_norm_error']=row['max_norm_error']
            b[f'{fam}_std_ms']=row['std_ms']
        b['cheb_over_newton']=float(b['chebyshev_median_ms'])/float(b['newton_median_ms'])
        b['cheb_over_bisection']=float(b['chebyshev_median_ms'])/float(b['bisection_median_ms'])
        best_by_p.append(b)
        metadata.append({
            'tag':tag, 'p_label':label, 'p_float':pvalue, 'source_file':filename,
            'cheb_degrees':sorted(int(x['degree']) for x in rows if x['family']=='chebyshev'),
            'newton_iterations':sorted(int(x['iterations']) for x in rows if x['family']=='newton'),
            'bisection_iterations':sorted(int(x['iterations']) for x in rows if x['family']=='bisection'),
            'common':{k:first.get(k,'') for k in (
                'lambda','dtype','device','device_name','n_timing','n_accuracy',
                'active_ratio_requested','active_ratio_realized','ymax_factor','seed',
                'warmup','repeats','validation_n','accuracy_target','git_commit',
                'python_version','torch_version','numpy_version',
                'reference_method','reference_newton_crosscheck_n',
                'max_reference_crosscheck_norm_error','reference_seconds',
            )},
            'n_fit':next((int(x['fit_n']) for x in rows if x['family']=='chebyshev' and x['fit_n']),'')
        })
    if len({a['commit'] for a in audit})!=1:
        raise ValueError('Formal CSVs are from mixed code commits')
    fields_full=['p_label','source_file']+list(all_rows[FORMAL[0][0]][0].keys())
    save_csv(tab/'formal_all_configurations.csv',full,fields_full)
    save_csv(tab/'formal_best_by_method.csv',bestrows,list(bestrows[0].keys()))
    save_csv(tab/'formal_best_comparison.csv',best_by_p,list(best_by_p[0].keys()))
    save_csv(reports/'formal_quality_checks.csv',audit,list(audit[0].keys()))
    do_plots(plots,best_by_p,all_rows)

    # Save a machine-readable descriptor of parameter grids reconstructed from CSVs.
    (reports/'formal_parameter_grids.json').write_text(
        json.dumps(metadata,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    allcsv=sorted(raw.rglob('*.csv'))
    inventory=[]
    for path in allcsv:
        rr=load_csv(path)
        inv={'relative_path':str(path.relative_to(root)),'run_class':path.parent.name,
             'rows':len(rr), 'p':rr[0].get('p',''), 'n_timing':rr[0].get('n_timing',''),
             'repeats':rr[0].get('repeats',''),
             'git_commit':rr[0].get('git_commit',''),
             'device_name':rr[0].get('device_name',''),
             'sha256':filehash(path),'size_bytes':path.stat().st_size}
        inventory.append(inv)
    save_csv(reports/'dataset_inventory.csv',inventory,list(inventory[0].keys()))

    checks={'complete_formal_files':len(FORMAL),'formal_configurations':len(full),
            'formal_pass_count':sum(a['n_pass_accuracy'] for a in audit),
            'formal_nonfinite_count':sum(a['n_nonfinite'] for a in audit),
            'formal_total_support_mismatches':sum(a['total_support_mismatches'] for a in audit),
            'max_reference_crosscheck_disagreement':max(a['max_reference_crosscheck_norm_error'] for a in audit),
            'formal_git_commit':audit[0]['commit'],
            'has_all_six_p':True}
    (reports/'quality_summary.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    summary=[
        '# Phase 1: accuracy-matched proximal benchmark — data summary',
        '',
        '## Scope',
        '',
        'These are the **six recorded A100 GPU float32 runs** using 1,000,000 scalar',
        'inputs, seed 0, active ratio requested 0.9, 10 warmup calls, 30 repeats,',
        'and an accuracy target of $10^{-6}$. All six CSVs record Git commit',
        f'`{audit[0]["commit"]}`. The pilot and smoke runs are archived separately.',
        '',
        'A configuration passes if the recorded output is finite and the measured',
        'maximum normalized error is at most $10^{-6}$; the summary additionally',
        'requires zero support mismatches. This is an **empirical, sampled accuracy**',
        'criterion, not a certified uniform bound on the active branch.',
        '',
        'The normalization is',
        '$E_\\infty=\\max_i |\\widehat x_i-x_i^\\mathrm{ref}|/',
        '\\max(|y_i|,\\tau_p)$; the reference is the recorded float64',
        '75-step bracketed bisection, crosschecked against an independent',
        'float64 Newton calculation on a subset.',
        '',
        '## Fastest *observed passing* candidate per family',
        '',
        '| p | Chebyshev | Cheb ms | Newton | Newton ms | Bisection | Bisection ms | Cheb/Newton |',
        '|---|---|---:|---|---:|---|---:|---:|',
    ]
    for b in best_by_p:
        summary.append('| {p_label} | {chebyshev_configuration} | {chebyshev_median_ms:.3f} | {newton_configuration} | {newton_median_ms:.3f} | {bisection_configuration} | {bisection_median_ms:.3f} | {cheb_over_newton:.2f} |'.format(
            **{k:(float(v) if k in {'chebyshev_median_ms','newton_median_ms','bisection_median_ms','cheb_over_newton'} else v) for k,v in b.items()}))
    summary.extend([
        '',
        '## Integrity checks',
        '',
        f'- Formal data: {checks["formal_configurations"]} configurations across {checks["complete_formal_files"]} runs.',
        f'- Non-finite configurations: {checks["formal_nonfinite_count"]}.',
        f'- Aggregate support mismatches: {checks["formal_total_support_mismatches"]}.',
        f'- Largest reference crosscheck disagreement: {checks["max_reference_crosscheck_disagreement"]:.3e}.',
        '- Original CSVs are copied byte-for-byte; SHA-256 values are recorded in `reports/dataset_inventory.csv`.',
        '',
        '## Interpretation and important limits',
        '',
        '1. Within these **current PyTorch implementations** and the specified error',
        '   target, low p (1/3 and 2/5) favors a low-degree Chebyshev polynomial by',
        '   ~6–8% in observed time; at p=0.75 Newton is slightly faster, and the',
        '   performance gap in favor of Newton increases at p=0.9, 0.99, and 0.999.',
        '2. The fastest *observed* Chebyshev candidate can be close to the target;',
        '   for p=0.999, D=116 passes the online-input criterion although its',
        '   separately sampled float64 branch error exceeds 1e-6. Use D=120',
        '   as a better-margin diagnostic, not as a mathematical certificate.',
        '3. Hardware/runtime results are specific to the current implementation,',
        '   PyTorch version, GPU, input distribution, and timing protocol;',
        '   fused or specialized kernels are not evaluated here.',
        '4. Only summary timing statistics are recorded, not raw per-repeat event',
        '   timings; hence paired confidence intervals cannot be reconstructed.',
        '5. The offline coefficient-family build time was measured separately and',
        '   is **not** included in the online median times. Build times cover the',
        '   whole fitted coefficient family, not a single chosen degree.',
        '6. The scalar proximal test does **not** yet cover threshold-adjacent',
        '   deterministic inputs, full end-to-end Lasso convergence, different',
        '   input sizes, kernel-launch benchmarking, or float64 GPU sweeps.',
        '7. The maximum active stationarity residual is an unnormalized diagnostic;',
        '   it is not equivalent to the measured normalized output error.',
        '8. The original interactive Colab notebook and stdout logs were not',
        '   supplied. The new notebook reconstructs the workflow from the',
        '   recorded CSV metadata and the known CLI. It does not invent logs.',
        '',
        '## Companion artifacts',
        '',
        '- `tables/formal_all_configurations.csv`: all formal rows with source labels.',
        '- `tables/formal_best_by_method.csv`: 18 fastest passing candidates (6 p × 3 methods).',
        '- `tables/formal_best_comparison.csv`: 6-row speed/accuracy overview.',
        '- `reports/dataset_inventory.csv`: datasets, SHA-256 hashes and metadata.',
        '- `reports/formal_quality_checks.csv`: one row of checks per formal run.',
        '- `figures/`: high-resolution PNG and vector PDF plots.',
        '- `notebooks/Phase1_Accuracy_Runtime_A100.ipynb`: structured Colab workflow.',
        '',
    ])
    (reports/'phase1_summary.md').write_text('\n'.join(summary), encoding='utf-8')
    print('Rebuilt derived Phase 1 artifacts:')
    print(json.dumps(checks,indent=2))
    for b in best_by_p:
        print(f"p={b['p_label']:>5}: Cheb {b['chebyshev_configuration']:>15} "
              f"{float(b['chebyshev_median_ms']):7.4f}ms; "
              f"Newton {b['newton_configuration']:>10} "
              f"{float(b['newton_median_ms']):7.4f}ms; "
              f"ratio {b['cheb_over_newton']:.3f}")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    args=parser.parse_args()
    main(args.root.resolve())
