"""Verify the complete public Phase-6 code-and-results archive.

Use from the Git repository root after archiving both Stage-2 source grids:
    python -m revision.phase6.scripts.verify_phase6_archive

The original Stage-3 raw data are checked without replaying expensive MB integrals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

STAGE2 = Path("revision/phase6/experiment_records/major7_stage2_20261021")
STAGE3 = Path("revision/phase6/experiment_records/major7_stage3_20261021")
REQUIRED_SOURCE = (
    "revision/phase6/mb_port.py",
    "revision/phase6/scripts/benchmark_mb_cpu.py",
    "revision/phase6/scripts/sweep_mb_accuracy.py",
    "revision/phase6/scripts/finalize_mb_comparison.py",
    "tests/test_phase6_mb_port.py",
    "tests/test_phase6_mb_benchmark.py",
    "tests/test_phase6_mb_sweep.py",
    "tests/test_phase6_finalize_mb_comparison.py",
)


def records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_hashes(folder: Path) -> int:
    manifest = folder / "SHA256SUMS.txt"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"Invalid SHA256 line: {line!r}")
        name = Path(match[2])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Invalid archive path: {name}")
        target = folder / name
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != match[1]:
            raise AssertionError(f"SHA256 mismatch: {target}")
        count += 1
    return count


def _flag(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def verify(*, stage2: Path = STAGE2, stage3: Path = STAGE3,
           check_stage2: bool = True, check_source: bool = True) -> None:
    if check_source:
        missing = [name for name in REQUIRED_SOURCE if not Path(name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing Phase-6 source/test files: {missing}")
    count3 = verify_hashes(stage3)
    full = records(stage3 / "major7_cpu_all_methods.csv")
    selected = records(stage3 / "major7_cpu_accuracy_matched.csv")
    meta = json.loads((stage3 / "major7_cpu_metadata.json").read_text(encoding="utf-8"))
    if len(full) != 54 or len(selected) != 30 or len(meta["input_shas"]) != 6:
        raise AssertionError("Stage-3 expected 54 all-method rows, 30 selected rows, 6 input SHA records")
    if int(meta["n"]) != 128 or float(meta["accuracy_target"]) != 1e-6:
        raise AssertionError("Unexpected Stage-3 input size or accuracy threshold")
    full_keys = set()
    for row in full:
        p = float(row["p"])
        k = format(p, ".17g")
        if row["input_sha256"] != meta["input_shas"].get(k):
            raise AssertionError(f"Stage-3 input SHA mismatch, p={p}")
        if int(float(row["support_mismatch_count"])) != 0:
            raise AssertionError(f"Stage-3 support mismatch, p={p}")
        if not _flag(row["all_finite"]):
            raise AssertionError(f"Stage-3 nonfinite output, p={p}")
        samples = json.loads(row["timing_repeats_raw_ms"])
        if len(samples) != 9 or not math.isclose(float(row["median_ms"]),
                                                  statistics.median(samples),
                                                  rel_tol=1e-7, abs_tol=1e-6):
            raise AssertionError(f"Stage-3 timing mismatch, p={p}")
        key = (k, row["family"], row["configuration"])
        if key in full_keys:
            raise AssertionError(f"Duplicate Stage-3 configuration: {key}")
        full_keys.add(key)
    for row in selected:
        key = (format(float(row["p"]), ".17g"), row["family"], row["configuration"])
        if key not in full_keys:
            raise AssertionError(f"Stage-3 selected row not present in all methods: {key}")
    print(f"Stage 3: {len(full)} raw rows, {len(selected)} selected, "
          f"{count3} SHA-256 files, 0 support mismatches: PASS")

    if not check_stage2:
        print("Stage 2: NOT VERIFIED (--skip-stage2)")
        return

    from revision.phase6.scripts.archive_stage2_records import validate_source
    count2 = verify_hashes(stage2)
    stages = (("standard", 6, 48, 6), ("longT", 4, 12, 4))
    all_grid = []
    for subdir, n_p, n_rows, n_selected in stages:
        folder = stage2 / subdir
        grid_rows, selected_rows = validate_source(folder, meta, n_p)
        if (grid_rows, selected_rows) != (n_rows, n_selected):
            raise AssertionError(f"Unexpected Stage-2 record count: {subdir}")
        all_grid.extend(records(folder / "mb_raw_contour_grid.csv"))

    # All Stage-3 MB configurations must come from one of the original grids.
    for row in full:
        if row["family"] != "paper_MB_no_fallback":
            continue
        match = re.fullmatch(r"N(\d+)_T([0-9.]+)_nt(\d+)", row["configuration"])
        if not match:
            raise AssertionError(f"Invalid Stage-3 MB configuration: {row['configuration']}")
        N, T, nt = int(match[1]), float(match[2]), int(match[3])
        candidates = [g for g in all_grid if math.isclose(float(g["p"]), float(row["p"]))
                      and int(g["N"]) == N and float(g["T"]) == T and int(g["nt"]) == nt]
        if len(candidates) != 1:
            raise AssertionError(f"Stage-3 MB configuration missing from Stage-2 grid: {row['configuration']}")
        grid_error = float(candidates[0]["max_complex_normalized_error"])
        saved_error = float(row["max_complex_normalized_error"])
        if not math.isclose(grid_error, saved_error, abs_tol=1e-11, rel_tol=1e-6):
            raise AssertionError(f"Stage-2/3 MB error mismatch for p={row['p']}")

    print(f"Stage 2: 48+12 grid rows, 6+4 selected rows, "
          f"{count2} SHA-256 files, linked to Stage-3: PASS")
    print("PASS: Phase-6 public experimental archive is independently restorable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-stage2", action="store_true",
                        help="Preflight only; final public archive MUST include Stage-2 data")
    args = parser.parse_args()
    verify(check_stage2=not args.skip_stage2)


if __name__ == "__main__":
    main()
