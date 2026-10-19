#!/usr/bin/env python3
"""Rebuild public Phase 2A summaries/plots from threshold_detail.csv."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rebuild(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / "threshold_detail.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run threshold experiment first")
    # ULP-neighbor tests require exact float64 round trips in derived CSVs.
    # The pandas default C parser can merge adjacent binary64 values.
    df = pd.read_csv(path, float_precision="round_trip")
    needed = {"p_label", "lambda", "dtype", "family", "configuration", "norm_output_error",
              "support_mismatch", "all_finite", "is_near_threshold", "oracle_side",
              "norm_stationarity_residual", "norm_objective_regret",
              "threshold_quantization_disagrees_with_oracle", "output_agrees_with_quantized_threshold"}
    if missing := needed.difference(df.columns):
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")
    group = ["p_label", "lambda", "dtype", "configuration", "family"]

    def summarize(x: pd.DataFrame) -> pd.Series:
        near = x[x["is_near_threshold"]]
        return pd.Series({
            "tested_inputs": len(x),
            "near_inputs": len(near),
            "zero_ties": int((x["oracle_side"] == "tie").sum()),
            "all_finite": bool(x["all_finite"].all()),
            "support_mismatch_count": int(x["support_mismatch"].sum()),
            "threshold_quantization_disagreement_count": int(x["threshold_quantization_disagrees_with_oracle"].sum()),
            "output_disagrees_with_quantized_threshold_count": int((~x["output_agrees_with_quantized_threshold"]).sum()),
            "near_support_mismatch_count": int(near["support_mismatch"].sum()),
            "max_norm_output_error": float(x["norm_output_error"].max()),
            "max_near_norm_output_error": float(near["norm_output_error"].max()) if len(near) else np.nan,
            "max_norm_stationarity_residual": float(x["norm_stationarity_residual"].max()),
            "max_norm_objective_regret": float(x["norm_objective_regret"].max()),
        })
    aggregate_rows = []
    for key, group_df in df.groupby(group, sort=False, dropna=False):
        row = dict(zip(group, key))
        row.update(summarize(group_df).to_dict())
        aggregate_rows.append(row)
    summary = pd.DataFrame(aggregate_rows)
    summary = summary.sort_values(["p_label", "lambda", "dtype", "family", "configuration"])
    summary.to_csv(root / "threshold_summary.csv", index=False)

    mismatches = df[df["support_mismatch"]].copy()
    # Keep all mismatches, not just those with the largest errors.
    mismatches.to_csv(root / "support_mismatch_details.csv", index=False)
    audit = pd.DataFrame([{
        "detail_rows": len(df), "cases": int(df[["p_label", "lambda", "dtype"]].drop_duplicates().shape[0]),
        "configs": len(summary), "all_finite": bool(df["all_finite"].all()),
        "total_support_mismatches": int(df["support_mismatch"].sum()),
        "total_threshold_quantization_disagreements": int(df["threshold_quantization_disagrees_with_oracle"].sum()),
        "outputs_inconsistent_with_quantized_threshold": int((~df["output_agrees_with_quantized_threshold"]).sum()),
        "near_support_mismatches": int(df.loc[df["is_near_threshold"], "support_mismatch"].sum()),
        "max_norm_output_error": float(df["norm_output_error"].max()),
        "max_norm_stationarity_residual": float(df["norm_stationarity_residual"].max()),
        "max_norm_objective_regret": float(df["norm_objective_regret"].max()),
    }])
    audit.to_csv(root / "threshold_quality_checks.csv", index=False)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)

    # Lambda=1 view for the two precisions, near-threshold sampled max errors.
    porder = ["1/3", "2/5", "3/4", "0.9", "0.99", "0.999"]
    families = ["chebyshev", "newton", "bisection"]
    types = [x for x in ("float32", "float64") if x in summary["dtype"].unique()]
    if types:
        fig, axes = plt.subplots(1, len(types), figsize=(7 * len(types), 4.7), squeeze=False)
        for j, dtype in enumerate(types):
            ax = axes[0][j]
            ss = summary[(summary["dtype"] == dtype) & np.isclose(summary["lambda"], 1.0)]
            if ss.empty:
                ax.set_axis_off()
                continue
            # For the Chebyshev family, show the conservative degree (larger D).
            selected = []
            for (pl, fam), h in ss.groupby(["p_label", "family"], sort=False):
                h = h.sort_values("configuration")
                if fam == "chebyshev":
                    h = h.loc[h["configuration"].str.extract(r"D(\d+)")[0].astype(int).idxmax()]
                else:
                    h = h.iloc[0]
                selected.append(h)
            plot_df = pd.DataFrame(selected)
            categories = [x for x in porder if x in ss["p_label"].values]
            for family in families:
                sub = plot_df[plot_df["family"] == family].set_index("p_label")
                vals = [float(sub.loc[p, "max_near_norm_output_error"]) if p in sub.index else np.nan for p in categories]
                ax.plot(categories, np.maximum(vals, 1e-18), marker="o", label=family)
            ax.set_title(f"{dtype}; lambda=1; near threshold")
            ax.set_yscale("log")
            ax.set_ylabel("Max normalized output error")
            ax.set_xlabel("p")
            ax.grid(True, which="both", alpha=0.2)
            ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figures / f"near_threshold_error.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    by_dtype = df.groupby(["p_label", "dtype", "family"], sort=False)["support_mismatch"].sum().reset_index()
    by_dtype.to_csv(root / "support_mismatch_counts.csv", index=False)

    report = [
        "# Phase 2A: Threshold and scaling robustness — data summary", "",
        "This is a numerical test, not a uniform error certificate or a timing benchmark.",
        "The oracle uses high-precision threshold comparisons and a separate bracketed root solver.",
        "At the exact mathematical threshold the reference selects the zero minimizer (the true prox is set-valued).", "",
        f"- Cases (p, lambda, dtype): **{int(audit.loc[0, 'cases'])}**",
        f"- Evaluations: **{len(df)}**",
        f"- Non-finite results: **{int((~df['all_finite']).sum())}**",
        f"- Support mismatches: **{int(df['support_mismatch'].sum())}**",
        f"- Predicted threshold-quantization disagreements: **{int(df['threshold_quantization_disagrees_with_oracle'].sum())}**",
        f"- Outputs inconsistent with quantized repo threshold: **{int((~df['output_agrees_with_quantized_threshold']).sum())}**",
        f"- Near-threshold support mismatches: **{int(df.loc[df['is_near_threshold'], 'support_mismatch'].sum())}**",
        "", "## Interpretation notes", "",
        "- Oracle side is based on the exact representable input compared to a high-precision threshold; it is not the side of the rounded device threshold.",
        "- Input grid includes floating-point neighbors, relative threshold shifts, and farther points on both signs.",
        "- Normalized output error is |xhat-xref|/max(|y|,tau).",
        "- Relative threshold distance is (|y|-tau)/tau; distance in ULP uses spacing at the nearest representable threshold.",
        "- Stationarity residual is only defined when the computed output is nonzero; zeros have blank residuals.",
        "- Objective regret is evaluated at high precision to reveal branch differences near the threshold, then divided by max(|y|,tau)^2.",
        "- This experiment does not compare run times or prove uniform-in-y error bounds.",
        "", "## Files", "",
        "- `threshold_detail.csv`: all signed inputs, labels, solutions, support and residual diagnostics.",
        "- `threshold_parameters.csv`: high-precision rho, tau and float rounding information.",
        "- `threshold_summary.csv`: per-configuration diagnostics.",
        "- `support_mismatch_details.csv`: all support discrepancies, including zero if absent.",
        "- `support_mismatch_counts.csv`: mismatch counts by p, dtype and method.",
        "- `threshold_quality_checks.csv`: aggregate test checks.",
        "- `metadata.json`: code/device versions and run parameters.",
    ]
    (root / "SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary, mismatches


def main():
    ap = argparse.ArgumentParser(description="Recompute summaries and public Phase 2A plots")
    ap.add_argument("--root", default="results/phase2/threshold")
    args = ap.parse_args()
    summary, mismatch = rebuild(Path(args.root))
    print(f"Rebuilt {len(summary)} summary rows; support mismatches={len(mismatch)}")


if __name__ == "__main__":
    main()
