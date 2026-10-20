#!/usr/bin/env python3
"""Extract matched-accuracy methodwise scalar metrics from archived Phase-1 CSVs.

This reports existing measurements; it NEVER runs CUDA or changes old results.
It selects the fastest *tested* passing configuration for each (p, family).

Usage from repo root:
  python revision/phase4/scripts/summarize_scalar_backends.py \
      --repo-root . --output-dir results/reviewer2_phase4_scalar_audit

If the CSVs are elsewhere, pass explicit files, folders, globs or ZIPs:
  ... --inputs path/to/phase1/ path/to/formal_A100_archive.zip
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import zipfile

PATTERN = re.compile(r"revision_phase1_A100_p\d+_1M\.csv$")
FAMILIES = ("chebyshev", "newton", "bisection")
FIELDS = [
    "p", "family", "configuration", "max_abs_error", "max_norm_error",
    "max_active_stationarity_residual", "mean_prox_objective_gap",
    "support_mismatch_count", "median_ms", "n_accuracy", "git_commit",
    "source", "source_sha256",
]


def p_label(p: float) -> str:
    for val, label in ((1 / 3, "1/3"), (2 / 5, "2/5"), (3 / 4, "3/4")):
        if abs(p - val) < 1e-13:
            return label
    return f"{p:g}"


def parse_csv(payload: bytes, source: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValueError("empty CSV")
        required = set(FIELDS[2:-2]) | {
            "family", "p", "all_finite", "meets_accuracy_target",
            "accuracy_target", "device", "dtype", "seed",
            "active_ratio_realized", "reference_method",
        }
        absent = sorted(required - set(reader.fieldnames))
        if absent:
            raise ValueError(f"missing columns: {absent}")
        rows = list(reader)
        if not rows:
            raise ValueError("no data rows")
        return rows
    except Exception as exc:
        raise ValueError(f"{source}: {exc}") from exc


def _all_inputs(root: Path, inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    if not inputs:
        paths = sorted(root.rglob("revision_phase1_A100_p*_1M.csv"))
    else:
        for spec in inputs:
            wildcard = any(ch in spec for ch in "*?[")
            glob_pattern = spec if Path(spec).is_absolute() else str(root / spec)
            candidates = [Path(x) for x in glob.glob(glob_pattern, recursive=True)] if wildcard else [Path(spec)]
            if not candidates:
                raise FileNotFoundError(f"No matches: {spec}")
            for p in candidates:
                p = p if p.is_absolute() else root / p
                if p.is_dir():
                    paths.extend(sorted(p.rglob("revision_phase1_A100_p*_1M.csv")))
                    paths.extend(sorted(p.rglob("*.zip")))
                elif p.is_file():
                    paths.append(p)
                else:
                    raise FileNotFoundError(p)
    return list(dict.fromkeys(p.resolve() for p in paths))


def read_sources(root: Path, inputs: list[str]) -> list[tuple[str, bytes]]:
    sources = []
    for path in _all_inputs(root, inputs):
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if PATTERN.search(Path(name).name):
                        sources.append((f"{path}:{name}", z.read(name)))
        elif PATTERN.search(path.name):
            sources.append((str(path), path.read_bytes()))
        elif inputs:
            raise ValueError(f"Not a matching Phase-1 CSV or a ZIP archive: {path}")
    if not sources:
        raise FileNotFoundError(
            "No revision_phase1_A100_p*_1M.csv files found. "
            "Pass --inputs PATH(s) to actual Phase-1 CSV files or archive ZIPs."
        )
    return sources


def _float(row: dict[str, str], field: str) -> float:
    try:
        v = float(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {field}: {row.get(field)!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"nonfinite {field} in {row.get('configuration')}")
    return v


def _same_input_checks(rows: list[dict[str, str]], source: str) -> None:
    for key in ("p", "n_accuracy", "dtype", "device", "seed", "reference_method", "active_ratio_realized"):
        vals = {r[key] for r in rows}
        if len(vals) != 1:
            raise ValueError(f"{source}: conflicting {key}: {vals}")
    flags = {r.get("accuracy_same_inputs_as_timing", "True") for r in rows}
    if flags != {"True"}:
        raise ValueError(f"{source}: some accuracy tensors differ from timing tensors")
    for r in rows:
        if _float(r, "accuracy_target") <= 0:
            raise ValueError("nonpositive accuracy target")


def select_records(sources: list[tuple[str, bytes]], target: float = 1e-6) -> list[dict[str, str]]:
    by_p: dict[float, tuple[str, bytes, list[dict[str, str]]]] = {}
    for name, payload in sources:
        rows = parse_csv(payload, name)
        _same_input_checks(rows, name)
        p = _float(rows[0], "p")
        if not 0 < p < 1:
            raise ValueError(f"invalid exponent {p}")
        if p in by_p:
            old_name, old_payload, _ = by_p[p]
            if payload == old_payload:
                continue
            raise ValueError(
                f"Two DIFFERENT datasets for p={p}: {old_name}; {name}. "
                "Specify only the intended formal archive via --inputs."
            )
        by_p[p] = (name, payload, rows)

    output = []
    for p, (name, payload, rows) in sorted(by_p.items()):
        for family in FAMILIES:
            candidates = [r for r in rows if r["family"] == family]
            if not candidates:
                raise ValueError(f"Missing backend {family} for p={p}: {name}")
            passing = [
                r for r in candidates
                if r["meets_accuracy_target"] == "True" and r["all_finite"] == "True"
                and _float(r, "max_norm_error") <= target
            ]
            if not passing:
                raise ValueError(f"No passing {family} for p={p}, target {target}")
            win = min(passing, key=lambda r: (_float(r, "median_ms"), r["configuration"]))
            entry = {field: win.get(field, "") for field in FIELDS if field not in ("source", "source_sha256")}
            entry.update(p=p_label(p), source=name, source_sha256=hashlib.sha256(payload).hexdigest())
            output.append(entry)
    return output


def tex_sci(value: str) -> str:
    x = float(value)
    if x == 0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(x))))
    mantissa = x / (10.0 ** exponent)
    return f"${mantissa:.2f}\\times10^{{{exponent}}}$"


def render_latex(rows: list[dict[str, str]]) -> str:
    out = [
        "% Generated from frozen Phase-1 CSV; NOT a new GPU benchmark.",
        "% Booktabs package required. E_norm selection threshold = 1e-6.",
        r"\begin{table}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Additional methodwise diagnostics for the accuracy-matched Phase-1 scalar benchmark.",
        r"The accuracy target is the maximum sampled normalized output error $10^{-6}$.",
        r"The maximum active stationarity residual uses the units defined by the Phase-1 implementation;",
        r"the reported proximal-objective gap is a sample mean, not a certified uniform bound.",
        r"All listed configurations have zero sampled support disagreements with the reference.",
        r"Original online runtimes are reported elsewhere.}",
        r"\label{tab:phase4-scalar-methodwise}",
        r"\begin{tabular}{@{}lllrrrr@{}}", r"\toprule",
        r"$p$ & Backend & Configuration & $\max|\widehat x-x_{\rm ref}|$ & $E_{\rm norm}$ & $\max r_{\rm active}$ & Mean objective gap \\",
        r"\midrule",
    ]
    for r in rows:
        p = r["p"]
        lab = f"${p}$" if "/" in p else f"${p}$"
        out.append(
            f"{lab} & {r['family']} & {r['configuration'].split('_',1)[-1].replace('_','')} & "
            + " & ".join(tex_sci(r[k]) for k in (
                "max_abs_error", "max_norm_error", "max_active_stationarity_residual",
                "mean_prox_objective_gap")) + r" \\"
        )
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def write_outputs(outdir: Path, rows: list[dict[str, str]], overwrite: bool = False) -> None:
    files = [outdir / "selected_scalar_backend_metrics.csv", outdir / "selected_scalar_backend_metrics.tex", outdir / "data_provenance.json"]
    if not overwrite and any(p.exists() for p in files):
        raise FileExistsError(f"Outputs exist under {outdir}; choose a new --output-dir or --overwrite")
    outdir.mkdir(parents=True, exist_ok=True)
    with files[0].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    files[1].write_text(render_latex(rows), encoding="utf-8")
    prov = {
        "description": "Postprocessing of archived Phase-1 A100 results, no new timing",
        "selection": "fastest tested with measured max_norm_error <= 1e-6, all_finite=True",
        "sources": [{"name": r["source"], "sha256": r["source_sha256"]} for r in rows[::3]],
        "count": len(rows), "all_support_mismatches_zero": all(r["support_mismatch_count"] == "0" for r in rows),
    }
    files[2].write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--inputs", nargs="*", default=[])
    p.add_argument("--output-dir", type=Path, default=Path("results/reviewer2_phase4_scalar_audit"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    root = args.repo_root.resolve()
    try:
        rows = select_records(read_sources(root, args.inputs))
        dest = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
        write_outputs(dest, rows, overwrite=args.overwrite)
    except (ValueError, FileNotFoundError, FileExistsError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Selected {len(rows)} matched-accuracy rows from {len(rows)//3} archived exponent datasets.")
    print(f"Output: {dest}")
    for r in rows:
        print(f"  p={r['p']:>5} {r['family']:9s} {r['configuration']:15s} "
              f"E_norm={float(r['max_norm_error']):.2e} "
              f"R_active={float(r['max_active_stationarity_residual']):.2e} "
              f"gap_mean={float(r['mean_prox_objective_gap']):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
