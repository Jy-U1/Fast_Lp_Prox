"""Checks for read-only reconstruction of the revised manuscript's FB tables."""
from pathlib import Path
import csv

import pytest

from revision.paper_reproduction import DEFAULT_RECORD, rebuild_tables, main


def test_archived_tables_have_expected_shapes_and_selected_values():
    table9, table10 = rebuild_tables()
    assert len(table9) == 18
    assert len(table10) == 6
    assert {r["evaluator"].split("_")[0] for r in table9} == {
        "chebyshev", "newton", "bisection"
    }
    small = next(r for r in table9 if r["p"] == "1/3"
                 and r["m"] == "1000" and r["evaluator"] == "chebyshev_D6")
    assert small["mean_first_checked_k"] == "130.0"
    assert small["without_checks_ms"] == "78.06"
    assert small["gradient_forward_ms"] == "19.71"
    assert small["prox_ms"] == "67.82"
    recovery = next(r for r in table10 if r["p"] == "1/3" and r["m"] == "1000")
    assert recovery["relative_l2_error"] == "0.16278"
    assert recovery["precision"] == "1.0000"
    assert recovery["recall"] == "0.6867"
    assert recovery["active_percent"] == "3.43"
    assert recovery["amortization_runs"] == "35"


def test_cli_writes_only_a_new_directory(tmp_path):
    target = tmp_path / "tables"
    assert main(["--output-dir", str(target)]) == 0
    nine = list(csv.DictReader((target / "table9_lasso_timings.csv").open()))
    ten = list(csv.DictReader((target / "table10_lasso_recovery.csv").open()))
    assert len(nine) == 18 and len(ten) == 6
    with pytest.raises(FileExistsError):
        main(["--output-dir", str(target)])


def test_rejects_incomplete_archive_summary(tmp_path):
    dst = tmp_path / "record" / "summaries"
    dst.mkdir(parents=True)
    originals = DEFAULT_RECORD / "summaries"
    for fname in ("phase4_lasso_six_case_summary.csv", "phase4_lasso_all_methods_verified.csv"):
        (dst / fname).write_bytes((originals / fname).read_bytes())
    path = dst / "phase4_lasso_all_methods_verified.csv"
    content = path.read_text().splitlines()
    path.write_text("\n".join(content[:-1]) + "\n")
    with pytest.raises(ValueError, match="18 method rows"):
        rebuild_tables(tmp_path / "record")
