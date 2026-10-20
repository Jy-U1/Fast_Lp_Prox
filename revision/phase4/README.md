# Phase 4 - additional JOMP Reviewer 2 experiments (non-destructive)

**Scope.** Reuse validated Phase-1 and Phase-3 results. Do not overwrite the published A100 logs. Phase-1 GPU scalars were at short SHA `7a1e32b`; Phase-3 FB replay measurements were at `40f8c07`. A new Phase-4 SHA is required for new measurements.

## Mac VS Code: start at repository root

```bash
pwd
git status -sb
git branch --show-current
git rev-parse --short HEAD
python -m pytest -q tests/test_phase3_lasso.py tests/test_phase4_reporting.py
```

The archive overlays only new paths under `revision/phase4/` and a new test file `tests/test_phase4_reporting.py`; it does not modify `src/fast_lp_prox` or any historical result.

## 1. Reuse stored Phase-1 scalar results, no GPU needed

```bash
python revision/phase4/scripts/summarize_scalar_backends.py --repo-root .
```

The script finds `revision_phase1_A100_p*_1M.csv` recursively. If your files are archived elsewhere, explicitly supply one or more CSV files, a directory or an archive ZIP with `--inputs`:

```bash
python revision/phase4/scripts/summarize_scalar_backends.py \
  --repo-root . --inputs revision/phase1 \
  --output-dir results/reviewer2_phase4_scalar_audit
```

Three generated outputs: `selected_scalar_backend_metrics.csv`, `selected_scalar_backend_metrics.tex`, and `data_provenance.json` with SHA-256 hashes of the source CSVs. The selection is the fastest *tested* method satisfying `max_norm_error <= 1e-6`. All reported errors are **sampled**. The objective gap column is a mean; it is not a certified uniform regret bound. The original Phase-1 CSV also records maximum active stationarity residual, and support mismatches. Any failure to locate all six exponents is an explicit incomplete-input warning, not evidence of zero error.

A summary extracted independently from the six original Phase-1 A100 CSVs shared in the project is included as `examples/historical_phase1_matched_metrics.csv`; this is a **postprocessing result**, not a new GPU run. Run the script on your repository's frozen raw CSVs to verify matching SHA and selection.

## 2. Lasso extension - check the command matrix before running

```bash
python revision/phase4/scripts/run_reviewer2_lasso_matrix.py --mode smoke
python revision/phase4/scripts/run_reviewer2_lasso_matrix.py --mode smoke --execute
```

The smoke run uses CPU, a tiny 80x160 matrix, one seed, 20 iterations. It is for API correctness only, **not** for timing conclusions. Outputs are under `results/reviewer2_phase4_lasso/`.

For a formal A100 run, use:

```bash
python revision/phase4/scripts/run_reviewer2_lasso_matrix.py --mode formal --device cuda
python revision/phase4/scripts/run_reviewer2_lasso_matrix.py --mode formal --device cuda --execute
```

To start with one formal case, add `--only-p 1/3 --only-size 1000x2000`.

Formal grid has `p in {1/3, 3/4, 0.9}`, two sizes `(m,n)=(1000,2000),(2000,4000)`, planted support fraction 5%, seeds `0,1,2`, common `R_ref <= 1e-6` checks every 10 iterations, max 500. Method budgets reuse the accuracy-matched Phase-1 scalar choices `Cheb D = 6/12/24`, Newton K = 3/4/4, bisection K = 20. This does **not** mean these configurations are universally optimal or that observed Lasso accuracy automatically passes at new scales. The driver must actually run and produce logs.

The runner stores command, branch, full git SHA, dirty status, stdout, stderr and return code separately for each case; it refuses to overwrite any existing case output. Formal timing requires a clean tree by default and CUDA. Do not mix Mac CPU smoke timings with A100 timing tables.

## 3. Still missing after this starter

- The fixed-term Lagrange-Buermann and [24] certified-hybrid baselines are not included, and still need implementation and fair timing protocols.
- Existing Lasso driver must be inspected before instrumenting recovery error against planted truth, support precision/recall, active fraction, and gradient-vs-prox CUDA stage-time breakdowns. The runner does not claim to produce these new metrics if the driver does not export them.
- Optional `--objective-guard` patch is separate and not applied here. Guarded timings must be benchmarked on their own and cannot replace old unsafeguarded A100 values.
- No paper or response text was automatically overwritten. Copy the generated TeX supplement only after checking data provenance and mathematical definitions.
