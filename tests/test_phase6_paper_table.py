"""Lightweight public-table consistency checks; no GPU or quadrature."""
import csv
import math
import pytest
from pathlib import Path

from revision.phase6.scripts.verify_stage3_objective_gaps import objective_gap_statistics

BASE = Path("revision/phase6/experiment_records/major7_stage3_20261021")


def read(name):
    with (BASE/name).open(newline="") as f:
        return list(csv.DictReader(f))


def test_table6_summary_matches_selected_rows():
    summary = read("major7_stage3_sixcase_summary.csv")
    selected = read("major7_cpu_accuracy_matched.csv")
    assert len(summary) == 6 and len(selected) == 30
    fam = {
        "paper_MB_no_fallback": ("MB_online_ms", "MB_passed"),
        "chebyshev": ("chebyshev_online_ms", "chebyshev_passed"),
        "newton": ("newton_online_ms", "newton_passed"),
        "fixed_series": ("fixed_series_online_ms", "fixed_series_passed"),
        "bisection": ("bisection_online_ms", "bisection_passed"),
    }
    for row in summary:
        candidates = [r for r in selected if math.isclose(float(r["p"]), float(row["p"]), abs_tol=1e-14)]
        assert len(candidates) == 5
        for r in candidates:
            time_key, pass_key = fam[r["family"]]
            assert math.isclose(float(row[time_key]), float(r["median_ms"]), rel_tol=1e-12)
            assert (row[pass_key] == "True") == (r["meets_accuracy_target"] == "True")
    assert sum(r["meets_accuracy_target"] == "True" for r in selected) == 25


def test_objective_gap_uses_nonnegative_mean():
    import numpy as np
    # The reference is deliberately inexact to exercise both gap signs.
    y = np.array([2.0, 0.5]); x = np.array([2.0, 1.0])
    ref = np.zeros_like(y)
    diag = objective_gap_statistics(y, x, ref, p=0.5)
    d = 0.5 * (x-y)**2 + np.abs(x)**0.5 - 0.5 * y**2
    assert np.any(d < 0) and np.any(d > 0)
    assert diag['mean_prox_objective_gap'] == pytest.approx(np.maximum(d, 0).mean())
    assert diag['mean_signed_objective_difference'] == pytest.approx(d.mean())
    assert diag['mean_prox_objective_gap'] > diag['mean_signed_objective_difference']
