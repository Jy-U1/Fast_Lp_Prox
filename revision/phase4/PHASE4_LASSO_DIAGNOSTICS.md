# Phase 4: additional Lasso diagnostics (Reviewer 2, Major 8)

This is an **additive, read-only extension of Phase 3**. It does not modify
`revision/phase3`, `src/fast_lp_prox`, existing results, benchmark timing
semantics, or historical commits. It adds recovery and active-coordinate
metrics and an instrumented gradient/prox timing **diagnostic**.

## Why this extra program is needed

The Phase-3 `run_summary.csv` contains objective values, independent reference
FB displacements, and support Jaccard against the planted support, but **not**
relative L2 recovery error: the coefficients of `xtrue` and the final
iterate vectors were not archived. The new program reproduces the original
NumPy random sequence (matrix first, then support and nonzero coefficients)
and reconstructs the same problem with the existing `make_problem` API.
It replays the named candidate configurations. The replay must match the
archived **final objective**, **nonzero count**, and **truth-support Jaccard**
before it writes any new results; otherwise the script stops with an error.

## Mac / VS Code: run against your successful CPU smoke result

From the existing repository root, with `.venv` active:

```bash
python -m pytest -q tests/test_phase4_lasso_diagnostics.py \
  tests/test_phase4_reporting.py tests/test_phase3_lasso.py

python -m revision.phase4.scripts.diagnose_lasso_outputs \
  --input-dir results/reviewer2_phase4_lasso_retry1/p03333_m80_n160/phase3_results \
  --device cpu --stage-repeats 2 --stage-warmup 1

head -n 5 results/reviewer2_phase4_lasso_retry1/p03333_m80_n160/phase4_diagnostics/diagnostics_summary.csv
```

Default output is a **new** sibling directory named `phase4_diagnostics`. The
script refuses to overwrite it. If a previous attempt created that directory,
choose a new path with `--output-dir`, rather than deleting results.

### Outputs

* `recovery_trajectories.csv`: selected untimed checkpoints from each replay,
  including `recovery_l2_relative`, `recovery_l2_absolute`, nonzero fraction,
  fraction of proximal **inputs** above the activation threshold, support
  precision/recall/F1/Jaccard, objective and iteration.
* `diagnostics_summary.csv`: terminal reconstruction metrics, Phase-3 replay
  integrity checks, the original independent stop criterion status, the original
  whole-loop replay time, and medians of separate instrumented stage timings.
* `stage_timings_raw.csv`: one record per repeat, separating the gradient +
  forward step from the proximal evaluator, measured with no objective or
  independent reference check in either timed stage.
* `diagnostics_metadata.json`: SHA-256 of the input CSV/metadata files,
  implementation version, precision/device, diagnostic definitions.

**Time definitions:** Instrumented stage times measure time spent in the
matrix-vector gradient and the prox kernels with extra CUDA events in the
execution stream. They are diagnostic estimates. They are **not** a replacement
for `replay_budget_median_ms` or `checked_stop_median_wall_ms` archived by
Phase 3; adding instrumented times need not reproduce uninstrumented totals.
For CPU timing, Python timing overhead and thread scheduling further limit
interpretation. **Do not use Mac CPU smoke timings to support a GPU speedup.**

**Stop definition:** `phase3_stop_hit` is unchanged from Phase 3; when 0, the
reported time corresponds to a fixed iteration budget, **not** time-to-tolerance.

## Formal campaign on A100

First run the existing Phase-4 launcher for a new, clean, non-overwriting
campaign as previously described. For each of its six experiment cases, run
this new diagnostic on the resulting `phase3_results` directory in the **same
CUDA environment** and use `--device cuda`, `--stage-repeats 3`.

Example for the `p=1/3`, `(m,n)=(1000,2000)` case:

```bash
python -m revision.phase4.scripts.diagnose_lasso_outputs \
  --input-dir results/reviewer2_phase4_lasso_formal/p03333_m1000_n2000/phase3_results \
  --device cuda --stage-repeats 3
```

The path is illustrative: substitute the actual `--output-root` used by the
formal launcher. Refuse to mix CPU diagnostics with archived CUDA timing.

## Important limitations

* This records finite-run recovery diagnostics. It is not a proof that an
  approximate nonconvex proximal method converges to a critical point.
* The optional objective safeguard is **not** enabled. Guarded GPU runtimes
  require their own benchmark and must not be substituted for historical data.
* The new code relies on the **current** Phase-3 planted-data RNG order. It
  verifies the regenerated support and final objective/support statistics.
  On an incompatible Phase-3 revision it intentionally fails rather than
  silently recording invented recovery errors.
* The Phase-3 metadata does not record dtype or power-iteration count. A
  `phase4_launch.json` in the parent supplies dtype; otherwise pass `--dtype`
  explicitly. The script assumes the Phase-3 default `--power-iterations 100`
  unless overridden. Similarly, `--n-fit 8192 --n-validation 80001` are the
  original defaults for this campaign. Exact fit sampled errors are rechecked
  when Chebyshev is present.
* Neither the fixed-term Lagrange-Buermann nor certified-hybrid baseline has
  been added here. They are a separate task under Reviewer 2 Major 7.
