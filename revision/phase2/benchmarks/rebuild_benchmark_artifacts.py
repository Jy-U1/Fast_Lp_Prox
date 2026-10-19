#!/usr/bin/env python3
"""Analyze Phase 2B per-call interleaved timings; preserve raw measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .run_benchmark_reliability import CANDIDATE_GRID, configuration
except ImportError:  # direct `python path/to/rebuild_benchmark_artifacts.py`
    from run_benchmark_reliability import CANDIDATE_GRID, configuration


def clustered_median_ci(df: pd.DataFrame, metric: str, *, resamples: int = 1000,
                        seed: int = 41) -> tuple[float, float]:
    """Two-level empirical bootstrap: resample sessions, then within-session blocks.

    Descriptive interval only; with few sessions it is not a significance test.
    """
    sessions = [v[metric].to_numpy(dtype=np.float64) for _, v in df.groupby("session")]
    if len(sessions) < 2 or resamples < 1:
        return float("nan"), float("nan")
    if any(len(values) == 0 or not np.all(np.isfinite(values)) for values in sessions):
        raise ValueError("Invalid session samples for bootstrap")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    for i in range(resamples):
        chosen = rng.integers(0, len(sessions), size=len(sessions))
        pieces = [v[rng.integers(0, len(v), size=len(v))] for idx in chosen
                  for v in [sessions[idx]]]
        estimates[i] = np.median(np.concatenate(pieces))
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pairings(raw: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    keys = ["p_label", "n", "session", "block"]
    if raw.duplicated(keys + ["configuration"]).any():
        raise ValueError("Duplicate candidate within randomized block")
    if raw.empty:
        raise ValueError("No per-invocation timings")
    if not (raw["wall_ms"].gt(0) & raw["wall_ms"].notna()).all():
        raise ValueError("Missing/invalid wall timings")
    full = []
    for label in raw["p_label"].unique():
        specs = CANDIDATE_GRID.get(label)
        if specs is None:
            raise ValueError(f"Unexpected p: {label}")
        newton_spec = next((configuration(f,w) for f,w in specs if f=="newton"), None)
        if newton_spec is None:
            raise ValueError("No Newton competitor")
        r = raw[raw["p_label"] == label]
        req = {configuration(f,w) for f,w in specs}
        actual = set(r["configuration"].unique())
        if req != actual:
            raise ValueError(f"p={label} candidate mismatch: expected {req} got {actual}")
        ndf = r[r["configuration"] == newton_spec][keys + ["cuda_event_ms", "wall_ms", "order_in_block"]]
        for f, w in specs:
            if f != "chebyshev":
                continue
            c_name = configuration(f,w)
            cdf = r[r["configuration"] == c_name][keys + ["cuda_event_ms", "wall_ms", "order_in_block"]]
            merged = cdf.merge(ndf, on=keys, suffixes=("_cheb", "_newton"), validate="one_to_one")
            if len(merged) != len(cdf) or len(merged) != len(ndf):
                raise ValueError("Incomplete pairing in randomized blocks")
            if not (merged["order_in_block_cheb"] != merged["order_in_block_newton"]).all():
                raise ValueError("Candidate order collision")
            merged["cheb_config"] = c_name
            merged["newton_config"] = newton_spec
            merged["ratio_wall_cheb_over_newton"] = merged["wall_ms_cheb"] / merged["wall_ms_newton"]
            if merged["cuda_event_ms_cheb"].notna().all() and merged["cuda_event_ms_newton"].notna().all():
                merged["ratio_event_cheb_over_newton"] = (
                    merged["cuda_event_ms_cheb"] / merged["cuda_event_ms_newton"]
                )
            else:
                merged["ratio_event_cheb_over_newton"] = float("nan")
            full.append(merged)
    paired = pd.concat(full, ignore_index=True)
    if paired.duplicated(keys + ["cheb_config"]).any():
        raise ValueError("Duplicate paired rows")
    qual = quality.set_index(["p_label", "n", "session", "configuration"])["accuracy_pass"]
    def _qualified(row):
        return bool(qual.loc[(row.p_label, row.n, row.session, row.cheb_config)] and
                    qual.loc[(row.p_label, row.n, row.session, row.newton_config)])
    paired["accuracy_matched_in_checked_subset"] = paired.apply(_qualified, axis=1)
    return paired


def summarize_pairs(paired: pd.DataFrame, *, resamples: int = 1000, seed: int = 41):
    rows = []
    session_rows = []
    for (label, n, cheb), group in paired.groupby(["p_label", "n", "cheb_config"], sort=False):
        is_cuda = group["cuda_event_ms_cheb"].notna().all()
        ratio = "ratio_event_cheb_over_newton" if is_cuda else "ratio_wall_cheb_over_newton"
        cheb_time = "cuda_event_ms_cheb" if is_cuda else "wall_ms_cheb"
        newton_time = "cuda_event_ms_newton" if is_cuda else "wall_ms_newton"
        lo, hi = clustered_median_ci(group, ratio, resamples=resamples, seed=seed)
        row = {
            "p_label": label, "n": int(n), "cheb_config": cheb,
            "newton_config": str(group["newton_config"].iloc[0]),
            "primary_timer": "cuda_event_ms" if is_cuda else "wall_ms",
            "n_sessions": int(group["session"].nunique()),
            "n_paired_blocks": len(group),
            "all_accuracy_checks_pass": bool(group["accuracy_matched_in_checked_subset"].all()),
            "median_cheb_ms": float(group[cheb_time].median()),
            "median_newton_ms": float(group[newton_time].median()),
            "median_paired_ratio_cheb_over_newton": float(group[ratio].median()),
            "ci95_low_ratio": lo, "ci95_high_ratio": hi,
            "mean_paired_ratio": float(group[ratio].mean()),
            "fraction_blocks_cheb_faster": float((group[ratio] < 1).mean()),
            "median_paired_wall_ratio": float(group["ratio_wall_cheb_over_newton"].median()),
        }
        rows.append(row)
        for s, sub in group.groupby("session"):
            session_rows.append({
                "p_label": label, "n": int(n), "cheb_config": cheb,
                "session": int(s), "n_blocks": len(sub),
                "median_paired_ratio": float(sub[ratio].median()),
                "accuracy_checks_pass": bool(sub["accuracy_matched_in_checked_subset"].all()),
            })
    return pd.DataFrame(rows).sort_values(["p_label", "n", "cheb_config"]), pd.DataFrame(session_rows)


def rebuild(root: Path, *, bootstrap_resamples: int = 1000, seed: int = 41):
    root = Path(root)
    raw = pd.read_csv(root / "timings_raw.csv", float_precision="round_trip")
    checks = pd.read_csv(root / "accuracy_checks.csv", float_precision="round_trip")
    metadata = json.loads((root / "metadata.json").read_text())
    if len(raw) != metadata["n_raw_timing_rows"] or len(checks) != metadata["n_accuracy_rows"]:
        raise RuntimeError("Raw data row counts do not match metadata")
    expected = sum(len(CANDIDATE_GRID[p]) * len(metadata["sizes"]) *
                   metadata["sessions"] * metadata["blocks_per_session"]
                   for p in metadata["p_labels"])
    if len(raw) != expected:
        raise RuntimeError(f"Incomplete timing records: {len(raw)}, expected {expected}")
    is_cuda = str(metadata["device"]).startswith("cuda")
    if is_cuda and (raw["cuda_event_ms"].isna().any() or not raw["cuda_event_ms"].gt(0).all()):
        raise ValueError("A CUDA benchmark has missing/nonpositive CUDA event times")
    if len(checks) != sum(len(CANDIDATE_GRID[p]) * len(metadata["sizes"]) * metadata["sessions"]
                          for p in metadata["p_labels"]):
        raise RuntimeError("Incomplete accuracy checks")
    paired = _pairings(raw, checks)
    summary, per_session = summarize_pairs(paired, resamples=bootstrap_resamples, seed=seed)
    paired.to_csv(root / "paired_blocks.csv", index=False)
    summary.to_csv(root / "paired_summary.csv", index=False)
    per_session.to_csv(root / "per_session_summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = root / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for (label, cfg), sub in summary.groupby(["p_label", "cheb_config"], sort=False):
        sub = sub.sort_values("n")
        ax.plot(sub["n"], sub["median_paired_ratio_cheb_over_newton"],
                marker="o", label=f"p={label}: {cfg}")
        valid = sub["ci95_low_ratio"].notna() & sub["ci95_high_ratio"].notna()
        if valid.any():
            ax.fill_between(sub.loc[valid, "n"].to_numpy(dtype=float),
                            sub.loc[valid, "ci95_low_ratio"].to_numpy(dtype=float),
                            sub.loc[valid, "ci95_high_ratio"].to_numpy(dtype=float), alpha=0.12)
    ax.axhline(1.0, linestyle="--", linewidth=1, color="black")
    ax.set_xscale("log")
    ax.set_xlabel("Number of scalar inputs per batch")
    ax.set_ylabel("Paired median time ratio: Chebyshev / Newton")
    ax.set_title("Phase 2B interleaved timing (primary GPU events, CPU wall)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=.22)
    fig.tight_layout()
    fig.savefig(figure_dir / "paired_ratio_vs_size.png", dpi=180)
    fig.savefig(figure_dir / "paired_ratio_vs_size.pdf")
    plt.close(fig)

    failures = checks[~checks["accuracy_pass"]]
    failed_note = ("Some accuracy spot-checks FAILED. Speed ratios for those parameter groups "
                   "must NOT be described as accuracy-matched.\n" if len(failures) else
                   "All tested candidate/configuration/session subsets passed the accuracy checks.\n")
    report = (
        "# Phase 2B: benchmark reliability — diagnostic summary\n\n"
        f"- Git commit: `{metadata['git_commit']}`\n"
        f"- Device: `{metadata['device']}`; dtype: `{metadata['dtype']}`\n"
        f"- Input sizes: `{metadata['sizes']}`\n"
        f"- Repeated sessions in this runtime: {metadata['sessions']}; randomized blocks/session: {metadata['blocks_per_session']}\n"
        f"- Candidate accuracy failures on checked input subsets: {len(failures)} / {len(checks)}\n"
        f"- Bootstrapped intervals: hierarchical empirical 95% percentiles across sessions and blocks; "
        f"{bootstrap_resamples} resamples. These are DESCRIPTIVE, not formal evidence of portable speed differences.\n\n"
        + failed_note +
        "\nCaution: CUDA event times capture stream work and possible host submission gaps, "
        "while wall time includes Python overhead and synchronization. "
        "No fused kernel or CUDA Graph optimization was performed. "
        "Changing input size may reveal fixed overhead but does not uniquely identify its source.\n"
    )
    (root / "SUMMARY.md").write_text(report)
    checksums = {path.relative_to(root).as_posix(): _sha(path) for path in sorted(root.rglob("*"))
                 if path.is_file() and path.name != "manifest.json"}
    (root / "manifest.json").write_text(json.dumps({"sha256": checksums}, indent=2) + "\n")
    print(f"Rebuilt {len(summary)} comparison summaries from {len(raw)} individual timings; "
          f"accuracy failures={len(failures)}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = ap.parse_args()
    rebuild(Path(args.root), bootstrap_resamples=args.bootstrap_resamples)


if __name__ == "__main__":
    main()
