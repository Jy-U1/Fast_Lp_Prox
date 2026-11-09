#!/usr/bin/env python3
"""Rebuild the two revised-manuscript Lasso tables from frozen public CSVs.

This is a reporting utility, not a GPU benchmark. It does not modify archived
results, and it never substitutes terminal (500-step) recovery for the first
verified stopping checkpoint.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


DEFAULT_RECORD = (
    Path(__file__).resolve().parent
    / "experiment_records"
    / "phase4_phase5_20261020"
)
FAMILIES = ("chebyshev", "newton", "bisection")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fmt(value: str, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def rebuild_tables(record_root: Path = DEFAULT_RECORD) -> tuple[list[dict], list[dict]]:
    summary_dir = record_root / "summaries"
    methods = read_csv(summary_dir / "phase4_lasso_all_methods_verified.csv")
    cases = read_csv(summary_dir / "phase4_lasso_six_case_summary.csv")
    if len(methods) != 18 or len(cases) != 6:
        raise ValueError("Expected 18 method rows and six problem cases")

    by_case: dict[str, dict[str, dict]] = {}
    for row in methods:
        case, family = row["case"], row["family"]
        if family not in FAMILIES or family in by_case.setdefault(case, {}):
            raise ValueError(f"Unexpected or duplicate method: {case}/{family}")
        if any(int(row[field]) != 3 for field in ("n_seeds", "stops", "checks")):
            raise ValueError(f"Incomplete stopping/replay checks: {case}/{family}")
        by_case[case][family] = row

    if set(by_case) != {c["case"] for c in cases}:
        raise ValueError("The six-case and all-method summaries disagree")

    table9, table10 = [], []
    for case in sorted(cases, key=lambda r: (float(r["p"].replace("1/3", str(1/3)).replace("3/4", str(3/4))), int(r["m"]))):
        code = case["case"]
        family_rows = by_case[code]
        if set(family_rows) != set(FAMILIES):
            raise ValueError(f"Missing one or more method families for {code}")
        for family in FAMILIES:
            r = family_rows[family]
            if (int(r["m"]), int(r["n"])) != (int(case["m"]), int(case["n"])):
                raise ValueError(f"Dimension mismatch: {code}/{family}")
            table9.append({
                "p": case["p"],
                "m": case["m"],
                "n": case["n"],
                "evaluator": r["method"],
                "mean_first_checked_k": fmt(r["mean_check_k"], 1),
                "without_checks_ms": fmt(r["mean_replay_ms"], 2),
                "with_checks_ms": fmt(r["mean_checked_ms"], 2),
                "gradient_forward_ms": fmt(r["mean_instrumented_gradient_ms"], 2),
                "prox_ms": fmt(r["mean_instrumented_prox_ms"], 2),
            })

        ch = family_rows["chebyshev"]
        if abs(float(case["mean_rel_rec_C"]) - float(ch["mean_atstop_rec"])) > 1e-12:
            raise ValueError(f"Recovery summary mismatch: {code}")
        nstar = case["amortization_solves_vs_newton"]
        table10.append({
            "p": case["p"],
            "m": case["m"],
            "n": case["n"],
            "relative_l2_error": fmt(ch["mean_atstop_rec"], 5),
            "precision": fmt(ch["mean_precision"], 4),
            "recall": fmt(ch["mean_recall"], 4),
            "active_percent": fmt(str(100 * float(ch["mean_active_frac"])), 2),
            "offline_fit_ms": fmt(ch["fit_ms"], 1),
            "amortization_runs": str(int(float(nstar))) if nstar.strip() else "--",
        })
    return table9, table10


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    nine, ten = rebuild_tables(args.record_root)
    if args.output_dir is not None:
        if args.output_dir.exists():
            raise FileExistsError(f"Output path already exists: {args.output_dir}")
        args.output_dir.mkdir(parents=True)
        save_csv(args.output_dir / "table9_lasso_timings.csv", nine)
        save_csv(args.output_dir / "table10_lasso_recovery.csv", ten)
        print(f"Wrote two derived tables to {args.output_dir}")
    else:
        print("PASS: six Phase-4 cases, 18 methods, independent stop checks and replay checks")
        print("Use --output-dir NEW_DIRECTORY to write two paper-ready CSV tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
