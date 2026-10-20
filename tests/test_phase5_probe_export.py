import csv
import pytest

from revision.phase5.scripts.export_matlab_probes import export


def test_probe_export_deterministic_and_no_overwrite(tmp_path):
    out=tmp_path/'tiny.csv'
    export(out,ps=(1/3,0.9),lam=1.0)
    with out.open() as fh: rows=list(csv.DictReader(fh))
    assert len(rows)==12
    assert min(float(r['y']) for r in rows)>0
    assert all(float(r['ref_x'])>0 for r in rows)
    with pytest.raises(FileExistsError): export(out,ps=(1/3,),lam=1.0)
