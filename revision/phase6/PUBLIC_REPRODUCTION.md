# Phase 6 — CPU Mellin–Barnes comparator for the revised manuscript

This is the self-contained source/data index for the **revised manuscript's Table 6**.
The original source and unchanged reference results are retained in
`revision/phase6/experiment_records/`. The comparison does **not** run the
complete certified hybrid evaluator from the reference article.

## Quick verification, without rerunning CPU timing

Run from the repository root after `python -m pip install -e ".[dev]"`:

```bash
python -m pytest -q tests/test_phase6_*.py
python -m revision.phase6.scripts.verify_phase6_archive
```

Stage 2 contains 48 standard and 12 long-contour scans. Stage 3 contains
54 candidate records and **30 selected rows, of which 25 pass the common
sampled 1e-6 accuracy criterion**. The five nonpassing representatives are
three MB settings at p=0.9, 0.99, 0.999, and two fixed-series settings at
p=0.99, 0.999; Table 6 labels these as non-comparable or missing.

`major7_stage3_sixcase_summary.csv` in the Stage-3 record corresponds to
Table 6; `major7_cpu_all_methods.csv` contains all candidate results;
`major7_cpu_accuracy_matched.csv` contains the per-family representatives.
The immutable data and SHA256SUMS records are preserved byte-for-byte.

## How the comparison was conducted

- Six exponents, n=128, lambda=1, float64, seed=0, about 90% active.
- CPU timings: three warmups, nine calls, median; single PyTorch thread for
  the polynomial/Newton/bisection baselines, SciPy/NumPy for raw MB and series.
- Reported MB uses the *paper's correction sign* and raw contour quadrature
  with **no** bisection fallback or machine-verifiable interval bounds.
- MB grid search uses N={6,12,24,48}, T={60,240}, extended to T=960 for
  p>=0.75; the actual `nt` is recorded in the CSV.
- No passing MB configuration was observed for p=0.9,0.99,0.999 **within
  that tested grid**. This does not rule out other grids/implementations.
- The public MATLAB source and the paper have a sign discrepancy in the MB
  correction; `MBPort` exposes both interpretations and documents the
  full-active radius choice.

## Optional objective-gap replay

The Stage-3 frozen results did not originally archive objective gaps.
`verify_stage3_objective_gaps.py` can derive them only after reproducing the
original exact input arrays; the archived environment used **NumPy 1.26.4**.
The script refuses to report gaps when an input SHA-256 differs.
It now distinguishes the nonnegative mean objective gap from the signed
mean objective difference: the paper's `mean_prox_objective_gap` is
`mean(max(Phi(x;y)-Phi(x_ref;y),0))`.

The original timings, input records and source-code provenance are not
rewritten by this optional calculation. No new GPU experiments are needed
for this archive.
