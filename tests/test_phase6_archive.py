"""Unit tests for the add-on public archive, independent of contour evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from revision.phase6.scripts.archive_stage2_records import validate_source
from revision.phase6.scripts.verify_phase6_archive import verify_hashes


def test_stage2_validates_exact_input_sha(tmp_path):
    p = 0.9
    key = format(p, ".17g")
    folder = tmp_path / "stage2"
    folder.mkdir()
    meta = {"p_grid": [p], "N_grid": [6], "T_grid": [60.0],
            "input_shas": {key: "abc"}, "n": 128, "seed": 0,
            "lam": 1.0, "tol": 1e-6}
    (folder / "mb_raw_contour_metadata.json").write_text(json.dumps(meta))
    for filename in ("mb_raw_contour_grid.csv", "mb_raw_contour_selected.csv"):
        with (folder / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["p", "N", "T"])
            writer.writeheader()
            writer.writerow({"p": p, "N": 6, "T": 60.0})
    stage3 = {"input_shas": {key: "abc"}, "n": 128, "seed": 0,
              "lam": 1.0, "accuracy_target": 1e-6}
    assert validate_source(folder, stage3, 1) == (1, 1)
    stage3["input_shas"][key] = "not_the_same_input"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_source(folder, stage3, 1)


def test_sha_manifest_detects_changes(tmp_path):
    target = tmp_path / "results.csv"
    target.write_text("a,b\n1,2\n")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS.txt").write_text(f"{digest}  results.csv\n")
    assert verify_hashes(tmp_path) == 1
    target.write_text("a,b\n3,4\n")
    with pytest.raises(AssertionError, match="SHA256 mismatch"):
        verify_hashes(tmp_path)


def test_sha_manifest_rejects_unsafe_path(tmp_path):
    (tmp_path / "SHA256SUMS.txt").write_text("a" * 64 + "  ../secret.csv\n")
    with pytest.raises(ValueError, match="Invalid archive path"):
        verify_hashes(tmp_path)
