#!/usr/bin/env python3
"""Offline audit of the public Phase 4/5 experiment record. Stdlib only."""
from pathlib import Path
import csv, hashlib, json, zipfile, io

HERE = Path(__file__).resolve().parent

def check(cond, reason):
    if not cond:
        raise AssertionError(reason)

def load_csv_from_zip(zf, name):
    raw = zf.read(name).decode('utf-8-sig')
    return list(csv.DictReader(io.StringIO(raw)))

def is_true(x):
    return str(x).strip().lower() in {'1','true','yes'}

def verify():
    checksums = (HERE/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()
    for line in checksums:
        digest, rel = line.split('  ',1)
        path = HERE/rel
        check(path.is_file(), f'missing archived file {rel}')
        check(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'SHA mismatch {rel}')
    print(f'SHA-256: {len(checksums)} files verified')

    manifest = json.loads((HERE/'EXPERIMENT_MANIFEST.json').read_text())
    p4=HERE/'raw'/'FastLp_Phase4_A100_All6_2dec97b1_COMPLETE_20261020_165704.zip'
    p5=HERE/'raw'/'FastLp_Phase5_A100_formal_All6_acdef3f2_20261020_200356.zip'

    count_p4 = count_stop = count_replay = 0
    with zipfile.ZipFile(p4) as zf:
        check(zf.testzip() is None, 'Phase 4 ZIP CRC failure')
        names=set(zf.namelist())
        cases=sorted({name.split('/')[1] for name in names if name.startswith('reviewer2_phase4_lasso_all/p')})
        check(len(cases)==6,f'Phase 4 case count = {len(cases)}')
        for case in cases:
            base=f'reviewer2_phase4_lasso_all/{case}'
            s=load_csv_from_zip(zf,base+'/phase3_results/run_summary.csv')
            d=load_csv_from_zip(zf,base+'/phase4_diagnostics/diagnostics_summary.csv')
            check(len(s)==9 and len(d)==9, f'Phase 4 {case}: expected nine rows in each file')
            count_p4 += len(s)
            count_stop += sum(is_true(row['verified_stop_hit']) for row in s)
            count_replay += sum(is_true(row['phase3_replay_check_pass']) for row in d)
        check((count_p4,count_stop,count_replay)==(54,54,54),
            f'Phase 4 counts: rows={count_p4}, stop={count_stop}, replay={count_replay}')
    print(f'Phase 4: {len(cases)} cases, {count_p4} rows, {count_stop} stop checks, {count_replay} replay checks')

    with zipfile.ZipFile(p5) as zf:
        check(zf.testzip() is None, 'Phase 5 ZIP CRC failure')
        names=set(zf.namelist())
        cases=sorted({x.split('/')[1] for x in names if x.startswith('cases/p')})
        check(len(cases)==6, f'Phase 5 case count = {len(cases)}')
        for case in cases:
            check(f'cases/{case}/phase5_scalar_results.csv' in names, f'Phase 5 missing {case} CSV')
            check(f'cases/{case}/phase5_metadata.json' in names, f'Phase 5 missing {case} metadata')
        rows=load_csv_from_zip(zf,'summary/all6_raw_results.csv')
        check(len(rows)==83, f'Phase 5 rows={len(rows)}')
        support=sum(int(row['support_mismatch_count']) for row in rows)
        accuracy=sum(is_true(row['meets_accuracy_target']) for row in rows)
        finite=sum(is_true(row['all_finite']) for row in rows)
        check((support,accuracy,finite)==(0,54,83),
            f'Phase 5 support={support}, accuracy_pass={accuracy}, finite={finite}')
    print(f'Phase 5: {len(cases)} cases, {len(rows)} rows, {accuracy} passing configurations, support mismatch={support}')
    print('PASS: all experiment-only archival checks passed')

if __name__=='__main__':
    verify()
