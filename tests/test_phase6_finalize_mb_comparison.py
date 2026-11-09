"""No-device tests of the Stage-3 selection and provenance checks."""
import csv
import json
import numpy as np
import pytest

from revision.phase6.scripts.finalize_mb_comparison import asbool, select_raw_mb, load_stage2


def row(p, N, T, nt, E, passed, t=20):
    return dict(p=str(p),N=str(N),T=str(T),nt=str(nt),
                max_complex_normalized_error=str(E),all_finite='True',
                meets_accuracy_target=str(passed),median_ms=str(t))


def test_boolean_csv_parser():
    assert asbool('True') and asbool('1') and asbool(True)
    assert not asbool('False') and not asbool('')


def test_stable_selection_prefers_minimum_grid_not_noisy_time():
    chosen,status=select_raw_mb([row(1/3,12,60,4000,1e-7,True,19.8),
                                 row(1/3,48,60,4000,1e-15,True,19.6),
                                 row(1/3,12,240,16000,1e-8,True,91)])
    assert status=='minimum_grid_passing' and int(chosen['N'])==12


def test_nonpassing_best_error():
    chosen,status=select_raw_mb([row(.9,48,960,64000,2e-5,False),
                                 row(.9,12,60,4000,4e-3,False)])
    assert status=='best_error_unpassed' and float(chosen['T'])==960


def test_reject_invalid_rows():
    chosen,status=select_raw_mb([dict(row(.9,6,60,4000,1e-5,True),all_finite='False'),
                                 dict(row(.9,12,60,4000,1e-6,True),max_complex_normalized_error='')])
    assert chosen is None and status=='no_usable_candidate'


def test_input_provenance_enforced(tmp_path):
    d=tmp_path/'stage2';d.mkdir()
    with (d/'mb_raw_contour_grid.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['p','N','T','nt','max_complex_normalized_error','all_finite','meets_accuracy_target','median_ms']);wr.writeheader()
        wr.writerow(row(1/3,12,60,4000,1e-7,True))
    (d/'mb_raw_contour_metadata.json').write_text(json.dumps(dict(p_grid=[1/3],n=128,lam=1.,seed=0,tol=1e-6,input_shas={format(1/3,'.17g'):'abcdef'})))
    with pytest.raises(ValueError,match='SHA'):
        load_stage2([d],[1/3],128,1.,0,1e-6,{1/3:'different'})
