# Phase 5 — Reviewer 2 Major Comments 6–7

This is a **non-destructive experiment overlay** for the existing Fast_Lp_Prox
GitHub project. It does not overwrite Phase 1–4 or claim that any new GPU
results have already been obtained. The main comparison is built on a *new*,
deterministic, same-input benchmark across all backends.

## What is implemented, and what is not

| Backend | Status | What can be claimed |
|---|---|---|
| Fixed-term Lagrange–Bürmann series | Implemented, tested | Direct N-term series, **not** error controlled |
| Residual-checked series + bisection | Implemented, tested | Practical a posteriori branch-error indicator and robust fallback; **not** the Mellin–Barnes hybrid from [24] |
| Existing Chebyshev / Newton / bisection | Reuses the preloaded *Phase-3* API | Original implementations, when `--existing-methods all` is used |
| Chebyshev + objective guard | Opt-in in the new scalar benchmark | Scalar candidate compared with zero in float64; **not** a proof of exact criticality |
| Original [24] MB hybrid | Separate MATLAB source-compatible probe driver supplied | Requires author's original `xplus_lp_hybrid.m` and a MATLAB/Octave installation; **not integrated into the A100 kernel benchmark** |

**Important scientific distinction:** In [24]'s 2026-09-16 arXiv v2,
Section 5.2 explicitly says that the analytic gamma-tail constants are not
numerically explicit; a floating-point certificate also needs validated
quadrature and directed rounding/interval arithmetic. A sampled MATLAB
'envelope bound' is NOT such a certificate. No code in this package is
advertised as a mathematically certified floating-point evaluator.

## Step A: Mac VS Code CPU smoke (repository root)

```bash
unzip -n ~/Downloads/FastLp_Reviewer2_Phase5_Scalar_And_Safeguard_Starter.zip -d .
python -m pytest -q tests/test_phase5_scalar.py tests/test_phase5_existing_adapter.py \
  tests/test_phase5_reporting.py tests/test_phase5_probe_export.py

python -m revision.phase5.scripts.benchmark_phase5_scalar \
  --device cpu --dtype float64 --p 0.3333333333333333 --lam 1 \
  --n 2000 --series-terms 8,16,32,64 \
  --hybrid-orders 8,32,128 --existing-methods all \
  --cheb-degrees 6 --newton-iters 3,4 --bisect-iters 20 \
  --warmup 2 --repeats 5 \
  --output-dir results/reviewer2_phase5_cpu_p03333
```

The `--existing-methods all` line is **the integration check** against your
local Phase-3 source. Package-only tests cannot replace it. If the existing
backend API has changed, stop at this step and inspect the traceback before
running A100 (do not rewrite your original code blindly).

## Step B: A100 pilot, after reviewing changes and pinning a new Git commit

The branch must include your newly added Phase-5 files; the old Phase-4
commit `2dec97b1` does not contain Phase 5. In a NEW output directory, run:

```bash
python -m revision.phase5.scripts.benchmark_phase5_scalar \
  --device cuda --dtype float32 --p 0.3333333333333333 --lam 1 \
  --n 50000 --series-terms 8,16,32,64,128 \
  --hybrid-orders 8,32,128 --existing-methods all \
  --cheb-degrees 6 --newton-iters 3,4 --bisect-iters 20 \
  --warmup 5 --repeats 10 \
  --output-dir results/reviewer2_phase5_A100_pilot_p03333
```

Then consider all six exponents in the Phase-1 grid:
`1/3,2/5,3/4,0.9,0.99,0.999`. Choose Chebyshev degrees and iteration counts
from the tested Phase-1 accuracy-matched grid, **but do not use the Phase-1
runtimes as if they were obtained on the new Phase-5 tensor**. All methods
in this experiment use exactly the same one tensor. Coefficient fitting,
series coefficient precomputation and reference solving are outside the online
CUDA timing block; separately recorded as offline costs.

A method that fails the shared `max_norm_error <= 1e-6` criterion is *not*
selected as an accuracy-matched winner. A fixed N series often fails near
activation when p approaches 1: this is a finding, not a script failure.

## Step C: objective guard in the same benchmark

Add `--guard-chebyshev` to Step B, using another NEW output directory. This
creates an additional backend `chebyshev_D..._objectiveguard` and measures its
incremental online cost on exactly the same tensor. The guard only tests
`Phi(xhat;y) <= Phi(0;y)` with a small finite-precision safety margin; it
is neither a global optimality condition nor the two-step model-descent and
vanishing-residual conditions needed for convergence of inexact FB.

To integrate that separate safeguard inside Phase-3 Lasso, use the previously
provided **optional** `Reviewer2_Objective_Safeguard_Code_Patch.zip`, and run
its tests and new guarded experiments. Do not merge guarded and unguarded
timings, and do not silently overwrite the frozen 6-case Phase-4 archive.

## Step D: original [24] MATLAB hybrid (optional independent validation)

The original MATLAB routine is at
https://github.com/lshen03/Lp-Proximity-Operator (main, xplus_lp_hybrid.m).

```bash
python -m revision.phase5.scripts.export_matlab_probes \
  --output results/phase5_matlab_probes.csv
# Download the original MATLAB repo (separate from the FastLp project),
# and set FPLP_ORIGINAL_MATLAB_REPO, FPLP_MB_PROBES_FILE and
# FPLP_MB_OUTPUT_FILE to absolute paths. Then in MATLAB:
# run('revision/phase5/matlab/run_original_mb_hybrid.m')
```

This tests near-threshold cases against an independent bisection reference.
MATLAB/Octave code has not been executed inside this package's test container;
run it on the intended platform and validate outputs before citing.
CPU MATLAB scalar times and GPU PyTorch throughput are distinct protocols and
are **not** an accuracy-matched implementation-level speed contest.

## Outputs and reproducibility

Each Phase-5 Python benchmark creates a new, otherwise nonexistent directory:

- `phase5_scalar_results.csv`: all tested backends (passing and failing),
  absolute and normalized errors, stationarity residuals, proximal-objective
  gaps, support mismatches, offline time, GPU online times and raw repetitions.
- `phase5_metadata.json`: complete parameters, software/device versions,
  input generation rule and tensor SHA-256, Git branch/commit/dirty status,
  and precise timing/error semantics.

After running a collection of configurations in *the same hardware, dtype,
input protocol and generator seed*, generate a matched-accuracy summary:

```bash
python -m revision.phase5.scripts.summarize_phase5 \
  results/reviewer2_phase5_A100_*/phase5_scalar_results.csv \
  --output-dir results/reviewer2_phase5_summary
```

**No publication values are inserted automatically.** Freeze source commit and
raw results before updating the paper or response.
