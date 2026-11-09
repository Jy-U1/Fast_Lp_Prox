"""Archive the two original Stage-2 MB contour grids into revision/phase6.

Usage from the Git repository root:
python -m revision.phase6.scripts.archive_stage2_records

Only measured CSV/JSON records are copied.  All input SHA-256 hashes must agree
with the completed Stage-3 same-CPU benchmark, or no files are published.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

STAGE3 = Path("revision/phase6/experiment_records/major7_stage3_20261021")
STAGE2 = Path("revision/phase6/experiment_records/major7_stage2_20261021")
FILENAMES = (
    "mb_raw_contour_grid.csv",
    "mb_raw_contour_selected.csv",
    "mb_raw_contour_metadata.json",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_source(folder: Path, stage3_meta: dict, expected_cases: int) -> tuple[int, int]:
    for filename in FILENAMES:
        if not (folder / filename).is_file():
            raise FileNotFoundError(f"Missing Stage-2 source: {folder / filename}")
    metadata = json.loads((folder / FILENAMES[2]).read_text(encoding="utf-8"))
    if (int(metadata["n"]), int(metadata["seed"])) != (
        int(stage3_meta["n"]), int(stage3_meta["seed"])
    ):
        raise ValueError(f"Stage-2 n/seed mismatch: {folder}")
    if float(metadata["lam"]) != float(stage3_meta["lam"]):
        raise ValueError(f"Stage-2 lambda mismatch: {folder}")
    if float(metadata["tol"]) != float(stage3_meta["accuracy_target"]):
        raise ValueError(f"Stage-2 tolerance mismatch: {folder}")

    grid = read_csv(folder / FILENAMES[0])
    selected = read_csv(folder / FILENAMES[1])
    num_p = len(metadata["p_grid"])
    expected_rows = num_p * len(metadata["N_grid"]) * len(metadata["T_grid"])
    if num_p != expected_cases or len(grid) != expected_rows or len(selected) != num_p:
        raise ValueError(
            f"Stage-2 incomplete: {folder}; got {len(grid)} grid rows and "
            f"{len(selected)} selected rows; expected {expected_rows} / {num_p}"
        )
    for p in metadata["p_grid"]:
        key = format(float(p), ".17g")
        observed = metadata["input_shas"].get(key)
        reference = stage3_meta["input_shas"].get(key)
        if not observed or observed != reference:
            raise ValueError(f"Input SHA-256 mismatch for p={p} in {folder}")
    return len(grid), len(selected)


def archive(stage2: Path, long_t: Path, *, dest: Path = STAGE2,
            stage3: Path = STAGE3) -> None:
    stage3_meta_path = stage3 / "major7_cpu_metadata.json"
    if not stage3_meta_path.is_file():
        raise FileNotFoundError(f"Missing Stage-3 metadata: {stage3_meta_path}")
    metadata = json.loads(stage3_meta_path.read_text(encoding="utf-8"))
    inputs = ((stage2, "standard", 6), (long_t, "longT", 4))
    summaries = []
    for source, subdir, ncases in inputs:
        grid_n, selected_n = validate_source(source, metadata, ncases)
        summaries.append((source, subdir, grid_n, selected_n))

    # No output is written until both source folders have passed validation.
    for source, subdir, grid_n, selected_n in summaries:
        folder = dest / subdir
        folder.mkdir(parents=True, exist_ok=True)
        for filename in FILENAMES:
            shutil.copy2(source / filename, folder / filename)
        print(f"ARCHIVED {subdir}: {grid_n} grid rows, {selected_n} selected rows")

    readme = (
        "# Original Stage-2 MB parameter scans\n\n"
        "These are unmodified measurements produced on the user's Mac. "
        "`standard/` contains 6 exponents x 4 N x 2 T = 48 rows; "
        "`longT/` contains 4 exponents x 3 N x 1 T = 12 rows. "
        "Input SHA-256 hashes and all parameter metadata are retained.\n\n"
        "The paper-consistent raw Mellin--Barnes candidate is tested without "
        "bisection fallback. The integration is ordinary floating-point "
        "quadrature, not a formally certified enclosure. These experiments "
        "use the full-active radius policy, not the original MATLAB default "
        "R=0.9*u_star. Compare with the Stage-3 same-CPU results in the "
        "adjacent folder; never mix these CPU timings with the A100 benchmarks.\n"
    )
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "README.md").write_text(readme, encoding="utf-8")
    digest_lines = []
    for path in sorted(dest.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            relative = path.relative_to(dest).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            digest_lines.append(f"{digest}  {relative}")
    (dest / "SHA256SUMS.txt").write_text("\n".join(digest_lines) + "\n", encoding="utf-8")
    print(f"PASS: source SHA-256 matches Stage-3 metadata; archive at {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2", type=Path, default=Path("results/reviewer2_major7_mb_stage2"))
    parser.add_argument("--longT", type=Path, default=Path("results/reviewer2_major7_mb_stage2_longT"))
    parser.add_argument("--destination", type=Path, default=STAGE2)
    parser.add_argument("--stage3-record", type=Path, default=STAGE3)
    args = parser.parse_args()
    archive(args.stage2, args.longT, dest=args.destination, stage3=args.stage3_record)


if __name__ == "__main__":
    main()
