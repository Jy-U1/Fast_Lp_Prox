# FastLp — Phase 4/5 public experimental record (2026-10-20)

This directory holds **experimental evidence only**. Source code and tests remain in the existing `src/`, `revision/phase1`–`revision/phase5`, `scripts/`, and `tests/` directories of this repository. **No manuscript, reviewer response, reviewer report, LaTeX draft, bibliography or private Overleaf material is included here.**

The record is designed to make the experiments independently recoverable from GitHub even if the original Colab runtime or local computer is lost.

## Pinned source commits

| Experiment | Git commit | Raw archive |
|---|---|---|
| Phase 4 multi-size FB Lasso | `2dec97b158f8f4d935616ee1842ca5f5f8ebff5d` | `raw/FastLp_Phase4_A100_All6_2dec97b1_COMPLETE_20261020_165704.zip` |
| Phase 5 scalar proximity benchmark, corrected float32 threshold comparisons | `acdef3f2902cdb67ab07006647a97dbe41f56601` | `raw/FastLp_Phase5_A100_formal_All6_acdef3f2_20261020_200356.zip` |

Use the corresponding commit to rerun a campaign. The commit that adds this **record** is not the same as the two **source** commits. The raw archives keep original per-run metadata and timings, and the notebooks preserve the execution configuration.

## Experimental protocols and results

**Phase 4**: `p ∈ {1/3, 3/4, 0.9}`, `(m,n) ∈ {(1000,2000), (2000,4000)}`, random seeds `0,1,2`; Chebyshev/Newton/Bisection methods; float32 kernels, independent reference-FB displacement tolerance `1e-6`, 500-iteration cap. All **54/54** method–seed records met the stopping threshold and passed replay diagnostics. Results include raw trajectories, objective values, support diagnostics, recovery errors, gradient/prox stage timings, and summary tables. First-stop recovery values must not be confused with the final (500-iteration) diagnostic values.

**Phase 5**: six exponents `1/3, 2/5, 3/4, 0.9, 0.99, 0.999`; **1,000,000** inputs per exponent, float32 GPU inputs, 10 warmups, 30 timed repeats on A100-SXM4-40GB. **83/83** configurations had finite output and zero support mismatches, and **54/83** met a sampled *max normalized output error* target of `1e-6`. The raw table also records stationarity residuals, proximal objective gaps, fitting/setup cost, fallback counts, and timing repeats. `1e-6` is **not** asserted to be a uniform or validated floating-point guarantee.

`residual_series_bisect` is a separately implemented residual-checked series with bisection fallback; it **must not be identified as an implementation of the original Mellin–Barnes hybrid**. The `chebyshev_guarded` timing measures a scalar objective guard; it does not establish convergence of a globally safeguarded inexact FB method.

## File map

- `raw/*.zip`: **original complete A100 experimental outputs**, versioned here because the two archives together are smaller than 1 MB.
- `notebooks/*.ipynb`: rerunnable, pinned A100/Colab campaigns (including per-case Drive checkpoints).
- `summaries/*.csv`: derived, machine-readable experiment tables. These are convenient summaries; the ZIP files remain the source of truth.
- `figures/*.png`: plots generated from public benchmark data, not manuscript pages.
- `EXPERIMENT_MANIFEST.json`: machine-readable experiment parameters and source commits.
- `SHA256SUMS.txt`: integrity checks for all files, excluding itself.
- `verify_experiment_record.py`: offline verification using only Python's standard library.

## Verify the archived results after cloning

From the repository root:

```bash
python revision/experiment_records/phase4_phase5_20261020/verify_experiment_record.py
```

This checks SHA-256, ZIP integrity, six-case coverage, method/seed counts, Phase 4 stopping/replay flags, and Phase 5 accuracy/support counts. No A100 GPU is needed to verify the *archived records*.

## Reproduce the computations (optional)

Use the supplied notebooks to reproduce the exact experimental settings on a supported NVIDIA A100 environment. For example, a new clone may check out the source commit before running its corresponding notebook:

```bash
git checkout --detach 2dec97b158f8f4d935616ee1842ca5f5f8ebff5d  # Phase 4
# OR
git checkout --detach acdef3f2902cdb67ab07006647a97dbe41f56601  # Phase 5
```

**Note:** Checking out an older commit may mean the record directory itself is absent in that checkout. Save/copy the notebook first, or fetch it from the branch commit that contains this record.

The Phase 4 execution uses `revision/phase4/scripts/run_reviewer2_lasso_matrix.py` followed by `python -m revision.phase4.scripts.diagnose_lasso_outputs`; Phase 5 uses `python -m revision.phase5.scripts.benchmark_phase5_scalar`. The notebooks are authoritative for the full argument grids and recovery procedure.

## Privacy and publication boundary

This is an experiment-only archive. Keep complete paper and response drafts in a **private Overleaf project**, not in `revision/experiment_records`, `results/`, notebook outputs, Git tags, GitHub Releases, or Git commit messages. Only commit specific public experimental files; avoid `git add .` during this archival step. Raw archives have been inspected for LaTeX/PDF/DOCX/bibliography files.
