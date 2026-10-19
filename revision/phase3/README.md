# Phase 3: end-to-end nonconvex Lp-Lasso application benchmark

This directory contains openly shareable reproducibility code only. It **does not** contain manuscript response letters or internal discussions.

The benchmark reuses the already tested `fast_lp_prox.revision_bench` online scalar proximal evaluators. It does not change their formulas. It measures whether differences in a scalar proximal kernel persist inside a complete forward--backward method.

## Problem and methods

\[
F(x)=\tfrac12\|Ax-b\|_2^2+\lambda\sum_i|x_i|^p,
\qquad x^{k+1}=\operatorname{prox}_{\alpha\lambda\|\cdot\|_p^p}
(x^k-\alpha A^T(Ax^k-b)).
\]

The main workload uses `m=2000`, `n=4000`, `p=1/3`, `lambda=0.05`, Gaussian design `A_ij~N(0,1/m)`, 200 ground-truth nonzeros, Gaussian noise `sigma=0.01`, `x0=0`, and `alpha=0.95/L_est` with seeded power iteration. Default seeds are 0,1,2. A common synthetic instance, initialization and stepsize are shared by every candidate for each seed.

Principal backends, selected according to the independent scalar-output accuracy experiments: `chebyshev:6`, `newton:3`, `bisection:20`. Optional legacy-budget comparison: `chebyshev:8,newton:8,bisection:40`. The reusable Chebyshev coefficients are fitted once, outside the iteration timings.

## Independent stopping check

At iteration `k`, define

\[
R(x^k)=\frac{\|x^k-T^{\mathrm{ref}}_\alpha(x^k)\|_2}{\max(1,\|x^k\|_2)},
\]

where `T_ref` evaluates the least-squares gradient in float64 using the *same rounded input matrix and observations*, and solves the scalar active-root problem using an independent NumPy float64 bisection. The default check is every 10 iterations, tolerance `1e-4` and maximum of 500 iterations. This normalized one-step displacement is **not** the gradient mapping norm (the latter has an additional `1/alpha` factor). `R<=tol` does not guarantee a global optimum or the same local optimum across solvers.

## Why diagnostics and timing are separate

The diagnostic trajectory logs F(x) at every iteration and high-accuracy residual at verification checkpoints. A reference float64 FB trajectory from the same x0 is also computed. None of these expensive diagnostic operations is used to claim solver-only runtime.

Two timing columns are reported:

- `replay_budget_median_ms`: rerun the complete online FB kernel for the number of iterations that reached the first independently verified checkpoint; high-accuracy checks are **excluded**. This is a fixed-budget replay selected using a separate diagnostic pass, not a wall-clock adaptive stopping method.
- `checked_stop_median_wall_ms`: run the FB loop with high-accuracy checks every 10 iterations, stopping when the same `R<=tol` rule succeeds. This time **includes** all expensive validation overhead.

The plots of objective versus online time use cumulative per-step diagnostic CUDA events with checks and objective computation excluded. The separately replayed full-loop times should be used for numerical comparisons. All raw times and failure-to-reach-tolerance flags are retained.

Support locations are compared to the float64 reference FB endpoint (same initialization and same maximum iteration budget), using exact equality and Jaccard similarity. These values are not interchangeable with support size or support recovery relative to the unknown true coefficients.

## Installation and tests

From repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_phase3_lasso.py
```

## Small CPU test

```bash
OMP_NUM_THREADS=1 python revision/phase3/scripts/run_lasso_end_to_end.py \
  --device cpu --m 60 --n 100 --support 10 --seeds 0 \
  --max-iter 60 --verify-every 10 --tol 1e-3 \
  --n-fit 512 --n-val 1001 --repeats 2 --warmup 0 \
  --checked-stop-repeats 1 \
  --output-dir results/phase3/lasso_smoke

python revision/phase3/scripts/rebuild_lasso_artifacts.py \
  --root results/phase3/lasso_smoke
```

The smoke run can legitimately fail to reach a stringent stopping tolerance in its short iteration budget; this is recorded, not silently considered a success.

## A100 full experiment

```bash
python revision/phase3/scripts/run_lasso_end_to_end.py \
  --device cuda --dtype float32 --p 1/3 --p-label 1/3 \
  --m 2000 --n 4000 --support 200 --sigma 0.01 --lambda-reg 0.05 \
  --seeds 0,1,2 --max-iter 500 --verify-every 10 --tol 1e-4 \
  --power-iterations 100 --ref-bisect-iters 80 \
  --candidates chebyshev:6,newton:3,bisection:20 \
  --n-fit 8192 --n-val 80001 --warmup 2 --repeats 5 \
  --checked-stop-repeats 2 \
  --output-dir results/phase3/lasso_formal_p0333

python revision/phase3/scripts/rebuild_lasso_artifacts.py \
  --root results/phase3/lasso_formal_p0333
```

Inspect `run_summary.csv`, `across_seed_summary.csv`, `support_comparison.csv` and the three plots per seed before making performance claims. If not every backend reaches the common tolerance, increase the common iteration cap or report an explicit non-convergence; **do not silently change the tolerance for one method**.

Optional p=3/4 application example: use `--p 3/4 --p-label 3/4 --candidates chebyshev:14,newton:4,bisection:20` and a different output directory. Degree 14 is used for a more conservative sampled-error margin than degree 12. This is a separate experiment with its own offline coefficients.

## Output organization

- `trajectories.csv`: sampled objective and independent stopping residual vs iteration, diagnostic per-step time.
- `reference_FB_trajectories.csv`: same-seed high-accuracy FB comparison (untimed).
- `run_summary.csv`: convergence hit, selected iteration budget, final objective/residual, exact support metrics, replay and checked-stop times.
- `timings_raw.csv`: individual repeats and timing definitions.
- `support_comparison.csv`: set overlap, exact equality, truth-support Jaccard.
- `metadata.json`: hardware, code commit, problem seeds, timing semantics, fitting details.
- `across_seed_summary.csv` and `figures/`: regenerated tables and figures.

Do not include temporary result ZIP files in source commits. Preserve the original result exports in a separate archive. This benchmark does not establish global optimality of the nonconvex objective.
