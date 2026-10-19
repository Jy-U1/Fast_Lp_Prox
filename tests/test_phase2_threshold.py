"""Phase 2A numerical protocol tests (no GPU required)."""
from __future__ import annotations

from types import SimpleNamespace
import json
import numpy as np
import mpmath as mp
import pytest

from revision.phase2.scripts.run_threshold_robustness import (
    _parsed_p_grid, generate_threshold_points, high_precision_positive_prox,
    run, threshold_high_precision,
)


def test_exact_tie_convention_with_representable_threshold():
    # At p=1/2, lambda=1: rho=1, tau=3/2 exactly.
    with mp.workdps(80):
        t = threshold_high_precision(0.5, 1.0)
        assert t.rho == 1
        assert t.tau == mp.mpf("1.5")
        assert high_precision_positive_prox(1.5, t)[0] == "tie"
        assert high_precision_positive_prox(1.5, t)[1] == 0
        below = float(np.nextafter(np.float64(1.5), np.float64(-np.inf)))
        above = float(np.nextafter(np.float64(1.5), np.float64(np.inf)))
        assert high_precision_positive_prox(below, t)[0] == "below"
        side, x = high_precision_positive_prox(above, t)
        assert side == "above" and x > t.rho


@pytest.mark.parametrize("dtype", [np.dtype("float32"), np.dtype("float64")])
def test_input_grid_contains_both_sides_and_deduplicates(dtype):
    with mp.workdps(75):
        t = threshold_high_precision(0.999, 1.0)
        rows = generate_threshold_points(t, dtype)
    arr = np.array([r["y"] for r in rows], dtype=dtype)
    assert len(arr) == len(np.unique(arr))
    assert len(arr) > 15
    assert 0 in arr and np.any(arr < 0) and np.any(arr > 0)
    assert all(isinstance(r["origins"], str) for r in rows)
    assert any("ulp+1" in r["origins"] for r in rows)
    assert any("ulp-1" in r["origins"] for r in rows)


def test_reference_root_and_objective():
    with mp.workdps(70):
        t = threshold_high_precision(0.99, 1e-6)
        y = float(2 * t.tau)
        side, x = high_precision_positive_prox(y, t, iters=160)
        assert side == "above"
        a = mp.mpf(y)
        residual = abs(x + t.lam * t.p * x ** (t.p - 1) - a)
        assert residual < mp.mpf("1e-32") * max(a, 1)
        fx = (x-a)**2 / 2 + t.lam * x**t.p
        fzero = a**2 / 2
        assert fx < fzero


def test_p_grid_rejects_unsupported():
    assert len(_parsed_p_grid("1/3,0.99")) == 2
    with pytest.raises(ValueError):
        _parsed_p_grid("0.101")


def test_cpu_smoke_integrates_with_repo(tmp_path):
    args = SimpleNamespace(p_grid="1/3", lambda_grid="1", dtypes="float32,float64",
        device="cpu", fit_n=256, validation_n=501, mp_dps=65,
        ref_iterations=132, output_dir=str(tmp_path / "run"), overwrite=False)
    data = run(args)
    assert data["detail_rows"] > 100
    assert data["parameter_rows"] == 2
    assert (tmp_path / "run" / "threshold_detail.csv").exists()
    assert (tmp_path / "run" / "threshold_parameters.csv").exists()
    assert json.loads((tmp_path / "run" / "metadata.json").read_text())["detail_rows"] == data["detail_rows"]
