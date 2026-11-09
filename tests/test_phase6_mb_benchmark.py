import argparse
import csv
import json
from pathlib import Path
from revision.phase6.scripts.benchmark_mb_cpu import run, parser


def test_end_to_end_small_fixed_parameters(tmp_path):
    opts=parser().parse_args(["--p","0.5","--n","32","--N","2",
        "--T","30","--nt","400","--include-phase5","no",
        "--warmup","0","--repeats","1","--output-dir",str(tmp_path)])
    rows=run(opts)
    assert len(rows)==6
    assert {r['method'] for r in rows}=={
        'mb_public_matlab_sign','mb_paper_v2_sign',
        'mb_paper_v2_sign_residual_checked',
        'fixed_series_N16','fixed_series_N64','bisection_K20_cpu_numpy'}
    assert (tmp_path/'mb_major7_cpu_results.csv').is_file()
    metadata=json.loads((tmp_path/'mb_major7_metadata.json').read_text())
    assert metadata['source_paper'].startswith('arXiv:2606.31951v2')
    assert len(metadata['case_input_shas'])==1
    for row in rows:
        assert row['support_mismatch_count']==0
    assert rows[2]['max_norm_error_complex']<1e-6


def test_matlab_default_rejection_is_reported(tmp_path):
    opts=parser().parse_args(["--p","0.99","--n","32","--N","2",
        "--T","25","--nt","400","--include-phase5","no",
        "--R-policy","matlab-default","--warmup","0","--repeats","1",
        "--output-dir",str(tmp_path)])
    rows=run(opts)
    a={r['method']:r for r in rows}
    assert a['mb_public_matlab_sign']['meets_accuracy_target'] is False
    assert a['mb_public_matlab_sign']['n_domain_rejected']>0
    assert 'REJECTED' not in a['mb_public_matlab_sign'].keys() # stored as error field
    assert 'epsilon>R' in a['mb_public_matlab_sign']['error']
