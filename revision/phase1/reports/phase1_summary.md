# Phase 1: accuracy-matched proximal benchmark — data summary

## Scope

These are the **six recorded A100 GPU float32 runs** using 1,000,000 scalar
inputs, seed 0, active ratio requested 0.9, 10 warmup calls, 30 repeats,
and an accuracy target of $10^{-6}$. All six CSVs record Git commit
`7a1e32b`. The pilot and smoke runs are archived separately.

A configuration passes if the recorded output is finite and the measured
maximum normalized error is at most $10^{-6}$; the summary additionally
requires zero support mismatches. This is an **empirical, sampled accuracy**
criterion, not a certified uniform bound on the active branch.

The normalization is
$E_\infty=\max_i |\widehat x_i-x_i^\mathrm{ref}|/
\max(|y_i|,\tau_p)$; the reference is the recorded float64
75-step bracketed bisection, crosschecked against an independent
float64 Newton calculation on a subset.

## Fastest *observed passing* candidate per family

| p | Chebyshev | Cheb ms | Newton | Newton ms | Bisection | Bisection ms | Cheb/Newton |
|---|---|---:|---|---:|---|---:|---:|
| 1/3 | chebyshev_D6 | 0.479 | newton_K3 | 0.511 | bisection_K20 | 2.269 | 0.94 |
| 2/5 | chebyshev_D6 | 0.470 | newton_K3 | 0.507 | bisection_K20 | 2.226 | 0.93 |
| 3/4 | chebyshev_D12 | 0.663 | newton_K4 | 0.617 | bisection_K20 | 2.287 | 1.07 |
| 0.9 | chebyshev_D24 | 1.028 | newton_K4 | 0.621 | bisection_K20 | 2.306 | 1.66 |
| 0.99 | chebyshev_D54 | 1.910 | newton_K5 | 0.734 | bisection_K20 | 2.249 | 2.60 |
| 0.999 | chebyshev_D116 | 3.732 | newton_K4 | 0.618 | bisection_K20 | 2.336 | 6.04 |

## Integrity checks

- Formal data: 112 configurations across 6 runs.
- Non-finite configurations: 0.
- Aggregate support mismatches: 0.
- Largest reference crosscheck disagreement: 4.297e-16.
- Original CSVs are copied byte-for-byte; SHA-256 values are recorded in `reports/dataset_inventory.csv`.

## Interpretation and important limits

1. Within these **current PyTorch implementations** and the specified error
   target, low p (1/3 and 2/5) favors a low-degree Chebyshev polynomial by
   ~6–8% in observed time; at p=0.75 Newton is slightly faster, and the
   performance gap in favor of Newton increases at p=0.9, 0.99, and 0.999.
2. The fastest *observed* Chebyshev candidate can be close to the target;
   for p=0.999, D=116 passes the online-input criterion although its
   separately sampled float64 branch error exceeds 1e-6. Use D=120
   as a better-margin diagnostic, not as a mathematical certificate.
3. Hardware/runtime results are specific to the current implementation,
   PyTorch version, GPU, input distribution, and timing protocol;
   fused or specialized kernels are not evaluated here.
4. Only summary timing statistics are recorded, not raw per-repeat event
   timings; hence paired confidence intervals cannot be reconstructed.
5. The offline coefficient-family build time was measured separately and
   is **not** included in the online median times. Build times cover the
   whole fitted coefficient family, not a single chosen degree.
6. The scalar proximal test does **not** yet cover threshold-adjacent
   deterministic inputs, full end-to-end Lasso convergence, different
   input sizes, kernel-launch benchmarking, or float64 GPU sweeps.
7. The maximum active stationarity residual is an unnormalized diagnostic;
   it is not equivalent to the measured normalized output error.
8. The original interactive Colab notebook and stdout logs were not
   supplied. The new notebook reconstructs the workflow from the
   recorded CSV metadata and the known CLI. It does not invent logs.

## Companion artifacts

- `tables/formal_all_configurations.csv`: all formal rows with source labels.
- `tables/formal_best_by_method.csv`: 18 fastest passing candidates (6 p × 3 methods).
- `tables/formal_best_comparison.csv`: 6-row speed/accuracy overview.
- `reports/dataset_inventory.csv`: datasets, SHA-256 hashes and metadata.
- `reports/formal_quality_checks.csv`: one row of checks per formal run.
- `figures/`: high-resolution PNG and vector PDF plots.
- `notebooks/Phase1_Accuracy_Runtime_A100.ipynb`: structured Colab workflow.
