# Fast ℓp Proximity Operator — Phase 1 benchmark archive

This is a self-contained, **review-free experiment data package** prepared from recorded experimental CSV files. It does not contain manuscripts, correspondence, reviewer reports, or private response drafts.

## What's inside

```text
FastLp_Phase1_Experiment_Package/
  notebooks/Phase1_Accuracy_Runtime_A100.ipynb   Clean Colab workflow
  data/formal/                                    6 original A100 1M CSVs
  data/pilot/                                     2 original A100 10K CSVs
  data/smoke/                                     3 original CPU/GPU test CSVs
  tables/formal_all_configurations.csv           112 source-labeled formal rows
  tables/formal_best_by_method.csv               18 best observed passing rows
  tables/formal_best_comparison.csv              6-row formal summary
  figures/                                       Rebuilt publication-size PNG & PDF
  reports/phase1_summary.md                      Objective experimental summary
  reports/formal_quality_checks.csv              Metadata and error audit
  reports/quality_summary.json                   Aggregate check results
  reports/dataset_inventory.csv                  SHA-256 of each original CSV
  reports/formal_parameter_grids.json            Exact tested grids recorded in CSVs
  scripts/rebuild_phase1_artifacts.py            Recompute all derived outputs
  README.md                                       This file
  manifest.json                                   Source-file integrity metadata
```

## Google Colab workflow

1. Open `notebooks/Phase1_Accuracy_Runtime_A100.ipynb` in Colab.
2. Run Section A and **upload this ZIP archive** when prompted.
3. Run Sections B–D to regenerate the formal comparison tables/plots. **No GPU needed.**
4. Optionally switch `RUN_BENCHMARKS = True` in Section E, select an A100/CUDA runtime and choose specific `p` values to rerun. These write to `data/rerun/` only.
5. Run Section F to download an updated ZIP.

## Local analysis (doesn't re-run the GPU experiment)

From the extracted package root:

```bash
python -m pip install matplotlib pandas nbformat
python scripts/rebuild_phase1_artifacts.py --root .
```

Then inspect `tables/formal_best_comparison.csv` and `reports/phase1_summary.md`.

## Formal experiment protocol and provenance

Six A100-SXM4-40GB, PyTorch 2.11.0+cu130, Python 3.13.15 runs, recorded by repository commit `7a1e32b` (`revision/major-experiments` branch at the time of execution). Each CSV records `n_timing=1_000_000`, `n_accuracy=1_000_000`, `dtype=float32`, `warmup=10`, `repeats=30`, `lambda=1`, `seed=0`, and target maximum normalized output error `1e-6`. Their numerical parameters, measured values, and implementation versions are in the raw CSVs; the tested degree and iteration grids are extracted into `reports/formal_parameter_grids.json`.

The reference uses independent float64 bracketed bisection for 75 iterations, crosschecked with float64 Newton on a subset. The online time **excludes** offline coefficient construction and reference generation. The Chebyshev offline time refers to constructing the complete tested coefficient family in a run.

A *passing* configuration in the derived summary requires measured error `max_i |xhat_i-xref_i|/max(|y_i|, tau_p) <= 1e-6`, all outputs finite, and **zero support mismatches**. It is the fastest observed passing configuration within the **tested discrete grid**. These are empirical sampled bounds, **not** whole-domain certificates. The independently reported active stationarity residual is not the same metric and is unnormalized.

The 2 pilot CSVs and 3 smoke CSVs are retained under separate directories. The CPU smoke results were obtained from a different code commit (`fbcc975`); do **not** combine their timings with the formal A100 comparison.

## Limitations and upcoming experiments

- Paired timing samples and original Colab execution logs were not provided; reported standard deviations summarize the run, but per-repeat confidence intervals cannot be reconstructed.
- Substantial GPU launch overhead may be present (this is a PyTorch operation-level comparison, not optimized fused CUDA kernels).
- Tests near the activation threshold, different input sizes, sensitivity to multiple random seeds, float64 GPU performance, and an end-to-end ℓp-Lasso experiment remain separate work.
- The original exploratory Colab `.ipynb` was not supplied; the included notebook is a fresh, reproducible reconstruction based on the actual CSV metadata and benchmark script interface.
- Rerunning may change timings and the installed environment compared with the archived A100 run. The Colab rerun section checks out commit `7a1e32b` before executing the benchmark to fix the code revision.

## Archive integrity

`manifest.json` and `reports/dataset_inventory.csv` contain the SHA-256 hashes of original CSV files, which were copied unchanged. Generated tables and charts can be rebuilt from raw data with the included script. All data package contents are suitable for sharing as experiment code and results; no private reviewer materials are included.
