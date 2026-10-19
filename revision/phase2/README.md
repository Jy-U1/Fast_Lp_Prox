# Phase 2 — numerical robustness and benchmark reliability

Phase 1 is frozen at `revision/phase1/`. **Do not modify archived Phase 1 CSVs or performance figures.** This directory contains public experiment code and instructions only. Private correspondence, manuscript drafts and replies are not part of this package.

## Phase 2A: threshold and scale robustness (current)

The scalar objective is `0.5*(x-y)^2 + lambda*abs(x)^p` for `0<p<1`.

Define the exact (real arithmetic) threshold parameters for the *binary64 numerical p and lambda inputs*:

`rho = [2 lambda (1-p)]^(1/(2-p))`, `tau = [(2-p)/(2(1-p))] rho`.

Below `tau`, the proximal minimizer is zero; at exactly `tau` the proximal map is **set-valued** `{0, sign(y)*rho}`; above `tau` it is the larger stationary root. For reproducibility this experiment **chooses zero at ties**. An 80-digit mpmath threshold comparison determines the reference side of each **representable** float32/float64 input. A separate bracketed high-precision root computation supplies the reference active value. This deliberately avoids accepting the candidate's rounded `tau` comparison as ground truth.

The input construction combines: signed nearest representable values around `tau` (1, 2, 4, 8, 16 and 64 ULP), relative perturbations down to `1e-14`, and far-field factors `0, 0.5, 0.99, 1.01, 2, 10, 100` times the threshold. Duplicates created by float32 rounding are removed and their generation labels combined.

It tests the original **unmodified** Phase 1 online methods with degrees/steps drawn from the Phase 1 measured configurations (fast and higher-degree Chebyshev, Newton, Bisection). Six exponents and lambda values `1e-6,1,1e6` yield 36 precision/scale cases with two dtypes. No throughput claims are made here; performance benchmarking belongs in Phase 2B.

Output distinguishes: support mismatch against the high-precision threshold oracle; disagreement explained by rounding the repository threshold into the tensor dtype; output error normalized by `max(|y|,tau)`; positive-branch stationarity residual; and high-precision objective regret (objective loss relative to reference). A zero output has **no active stationarity residual**. Output/threshold errors on a finite collection of test inputs **are not** uniform error certificates.

### Run on CPU (VS Code)

From the project root, in your editable installation:

```bash
python -m pip install mpmath
python -m pytest -q tests/test_phase2_threshold.py
python revision/phase2/scripts/run_threshold_robustness.py \
  --device cpu --p-grid 1/3,0.99 --lambda-grid 1 \
  --fit-n 2048 --validation-n 4001 \
  --output-dir results/phase2/threshold_smoke
python revision/phase2/scripts/rebuild_threshold_artifacts.py \
  --root results/phase2/threshold_smoke
```

### Formal threshold/scale experiment (Colab or local)

```bash
python revision/phase2/scripts/run_threshold_robustness.py \
  --device cuda --p-grid 1/3,2/5,3/4,0.9,0.99,0.999 \
  --lambda-grid 1e-6,1,1e6 --dtypes float32,float64 \
  --fit-n 8192 --validation-n 80001 \
  --mp-dps 80 --ref-iterations 155 \
  --output-dir results/phase2/threshold_formal
python revision/phase2/scripts/rebuild_threshold_artifacts.py \
  --root results/phase2/threshold_formal
```

The formal run may take several minutes because its reference is arbitrary-precision; this is **not included in candidate online evaluation cost**. The notebook `notebooks/Phase2_Threshold_Robustness.ipynb` contains both modes and a ZIP export stage. Existing output directories are not overwritten unless `--overwrite` is specified.

### Report fields and constraints

- `threshold_parameters.csv` records exact oracle thresholds as decimal strings, original library float64 `tau`, rounded dtype center, and the mathematical side of the center.
- `threshold_detail.csv` records all signed inputs/method outputs/reference values and support/accuracy diagnostics, including near-vs-far separation.
- `threshold_summary.csv` and `support_mismatch_details.csv` are derived; use the rebuild script.
- `metadata.json` records numerical precision, code revision, implementation versions and experiment parameters.
- `SUMMARY.md` and `figures/near_threshold_error.png` summarize numerical behavior, *not* timings.
- Do not mix these results with Phase 1 runtime comparisons: Phase 2A addresses numerical robustness only.

## Phase 2B (planned next)

Independent repeated, interleaved timing for the closest Phase 1 competitions (p=1/3,2/5,3/4), including input-size scaling to identify PyTorch CUDA kernel-launch overhead. Code and datasets for Phase 2B will be placed in a separate subdirectory after reviewing Phase 2A.

## Follow-up work (not part of Phase 2A)

An end-to-end ell_p-Lasso test with objective-vs-iteration/runtime and high-accuracy reference proximal gradient residual is a distinct next experiment. Manuscript and private response drafting will follow validated experiments.
