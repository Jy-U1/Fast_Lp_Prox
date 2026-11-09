from argparse import Namespace
import csv

import pytest
from revision.phase6.scripts.sweep_mb_accuracy import parse_grid, run


def test_parse_grid():
    assert parse_grid('12,3,6',int)==[3,6,12]
    for g in ('', '0,3', '2,2', 'invalid'):
        with pytest.raises(ValueError):parse_grid(g,int)


def test_grid_outputs_without_fallback(tmp_path):
    a=Namespace(p='0.3333333333333333',lam=1.,n=32,seed=0,
        N_grid='3',T_grid='60',dt=.5,max_panels=100000,
        chunk=32,tol=1e-6,imag_tol=1e-6,
        warmup=0,repeats=1,output_dir=str(tmp_path))
    records,best=run(a)
    assert len(records)==1 and len(best)==1
    assert records[0]['domain_rejected']==0
    assert records[0]['max_complex_normalized_error']>0
    assert records[0]['meets_accuracy_target'] is False
    assert records[0]['setup_ms']>=0
    with (tmp_path/'mb_raw_contour_grid.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==1 and rows[0]['p'].startswith('0.333')


def test_panel_cap_skips(tmp_path):
    a=Namespace(p='0.4',lam=1.,n=32,seed=0,
        N_grid='3',T_grid='960',dt=.005,max_panels=100000,
        chunk=32,tol=1e-6,imag_tol=1e-6,
        warmup=0,repeats=1,output_dir=str(tmp_path))
    records,best=run(a)
    assert len(records)==1 and not best
    assert 'panel cap' in records[0]['error']
