"""Preserve adjacent binary64 threshold inputs when rebuilding derived CSVs."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd

from revision.phase2.scripts.rebuild_threshold_artifacts import rebuild


def test_mismatch_export_preserves_adjacent_float64_values(tmp_path: Path) -> None:
    a = 1.0
    b = math.nextafter(a, math.inf)
    assert a != b
    records = [
        {
            "p_label": "1/3",
            "lambda": 1.0,
            "dtype": "float64",
            "configuration": "newton_K3",
            "family": "newton",
            "y": value,
            "norm_output_error": 0.8,
            "norm_stationarity_residual": 0.0,
            "norm_objective_regret": 0.0,
            "oracle_side": "above",
            "support_mismatch": True,
            "all_finite": True,
            "is_near_threshold": True,
            "threshold_quantization_disagrees_with_oracle": True,
            "output_agrees_with_quantized_threshold": True,
        }
        for value in (a, b)
    ]
    pd.DataFrame.from_records(records).to_csv(tmp_path / "threshold_detail.csv", index=False)
    rebuild(tmp_path)
    with (tmp_path / "support_mismatch_details.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    # Parse with Python's correctly rounded binary64 float conversion.
    assert {float(row["y"]) for row in rows} == {a, b}
    assert rows[0]["y"] != rows[1]["y"]
