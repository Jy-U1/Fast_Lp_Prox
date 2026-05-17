# fast-lp-prox

`fast-lp-prox` is a research-oriented Python toolkit for fast evaluation of the nonconvex scalar and vector Lp proximal operator

```math
\operatorname{prox}_{\lambda |\cdot|^p}(y)
=
\arg\min_x \frac{1}{2}(x-y)^2 + \lambda |x|^p,
\qquad 0 < p < 1.
```

For vector inputs, the operator is applied coordinatewise.

The main goal of this project is to provide reusable, testable, and benchmarkable implementations of Lp proximal operators. Proximal-gradient and Forward-Backward routines are included only as lightweight application helpers and demos; they are not the main contribution of the package.

---

## Highlights

- PyTorch implementation of scalar and vector Lp proximal operators
- Fixed-control-flow Chebyshev polynomial evaluator for the active branch
- Newton and bisection baselines for comparison and validation
- Built-in coefficient tables for selected rational exponents
- Accuracy validation against high-iteration float64 reference solvers
- Stationary residual and objective-gap validation
- CPU and CUDA benchmark scripts
- Visualization scripts for prox curves, approximation error, runtime, throughput, and validation metrics
- Lp-Lasso Forward-Backward demo as an application workload

---

## Project status

This repository is currently an early research prototype.

The implementation is intended for experimentation, benchmarking, and reproducibility. APIs, coefficient tables, benchmark scripts, and documentation may still change.

Benchmark results should be treated as development results unless explicitly marked as official.

---

## Motivation

For nonconvex Lp regularization with `0 < p < 1`, the scalar proximal operator has a threshold structure:

```math
\operatorname{prox}_{\lambda |\cdot|^p}(y) = 0
\quad \text{when } |y| \le \tau_{p,\lambda}.
```

For active entries, the nonzero solution is the larger positive root of a scalar nonlinear equation. Traditional implementations often use iterative root solvers such as Newton or bisection.

This project explores an alternative approach:

1. Normalize the active root equation.
2. Approximate the active branch offline using Chebyshev polynomials.
3. Evaluate the polynomial approximation at runtime using fixed control flow.

The intended engineering advantage is that the polynomial evaluator can be more suitable for large batched tensor workloads than repeatedly applying iterative scalar root solvers.

This project does **not** claim that Newton or bisection cannot run on GPUs. They can. The goal is to provide and benchmark a fixed-control-flow polynomial evaluator for the Lp prox operator.

---

## Installation

For development:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a lighter editable install:

```bash
python -m pip install -e .
```

Run tests:

```bash
python -m pytest -q
```

---

## Quick start

### Build a reusable prox evaluator

```python
import torch
from fast_lp_prox import make_lp_prox

y = torch.linspace(-5.0, 5.0, 1001)

prox = make_lp_prox(
    p=1/3,
    lam=1.0,
    method="poly",
)

x = prox(y)
```

For repeated calls, prefer `make_lp_prox(...)` because it builds the callable once and then reuses it.

### One-off prox call

```python
import torch
from fast_lp_prox import lp_prox

y = torch.linspace(-5.0, 5.0, 1001)

x = lp_prox(
    y,
    p=1/3,
    lam=1.0,
    method="poly",
)
```

---

## Supported methods

| Method | Description |
|---|---|
| `poly` | Chebyshev polynomial evaluator for the active branch |
| `newton8` | 8-step safeguarded Newton baseline |
| `newton12` | 12-step safeguarded Newton baseline |
| `bisect40` | 40-step bisection baseline |

Example:

```python
from fast_lp_prox import make_lp_prox

prox_poly = make_lp_prox(p=1/3, lam=1.0, method="poly")
prox_newton = make_lp_prox(p=1/3, lam=1.0, method="newton12")
prox_bisect = make_lp_prox(p=1/3, lam=1.0, method="bisect40")
```

---

## Built-in coefficient tables

The polynomial evaluator uses Chebyshev coefficient tables for selected rational exponents.

Currently included target cases:

```text
p = 1/3
p = 2/5
p = 3/4
```

Coefficient tables are stored in:

```text
src/fast_lp_prox/coeff_tables/
```

When `method="poly"` is used, `make_lp_prox(...)` first tries to load an existing coefficient table. If no suitable table is available, it falls back to building the Chebyshev approximation.

Generate a coefficient table:

```bash
python scripts/generate_coeffs.py \
  --p 1/3 \
  --target-error 2e-7 \
  --max-degree 128 \
  --n-fit 8192 \
  --n-val 80001 \
  --verbose
```

Generate the main development tables:

```bash
python scripts/generate_coeffs.py \
  --p 1/3 \
  --target-error 2e-7 \
  --max-degree 128 \
  --n-fit 8192 \
  --n-val 80001 \
  --verbose

python scripts/generate_coeffs.py \
  --p 2/5 \
  --target-error 2e-7 \
  --max-degree 128 \
  --n-fit 8192 \
  --n-val 80001 \
  --verbose

python scripts/generate_coeffs.py \
  --p 3/4 \
  --target-error 2e-7 \
  --max-degree 160 \
  --n-fit 8192 \
  --n-val 80001 \
  --verbose
```

---

## CUDA benchmark snapshot

The isolated prox benchmark compares the polynomial evaluator `method="poly"` against iterative reference methods `newton8`, `newton12`, and `bisect40`.

One CUDA benchmark was run on:

```text
GPU: NVIDIA A100-SXM4-80GB
dtype: float32
n: 1,000,000
lambda: 1.0
active_ratio: 0.9
```

| p | poly median time | poly throughput | speedup vs newton8 | speedup vs newton12 | speedup vs bisect40 |
|---:|---:|---:|---:|---:|---:|
| 1/3 | 0.663 ms | 1508 M elems/s | 1.87x | 2.64x | 7.25x |
| 2/5 | 0.663 ms | 1509 M elems/s | 1.90x | 2.63x | 7.29x |
| 3/4 | 0.928 ms | 1078 M elems/s | 1.32x | 1.86x | 5.14x |

All benchmarked methods returned finite outputs. Accuracy and residual validation show float32-level agreement with the reference methods.

See [docs/benchmark_a100.md](docs/benchmark_a100.md) for the full A100 benchmark report, including environment details, validation results, figures, and reproduction commands.

## Large-scale CUDA prox scaling

A dedicated CUDA scaling benchmark evaluates the isolated Lp prox evaluator for vector sizes from `n = 10,000` to `n = 10,000,000`.

At `n = 10,000,000` on an NVIDIA A100-SXM4-40GB GPU, the polynomial evaluator reaches approximately 1.87B to 3.00B elements/s across `p in {1/3, 2/5, 3/4}`. The speedup over iterative baselines becomes more pronounced at large vector sizes, reaching about 2.25x over `newton8`, 3.29x over `newton12`, and 9.11x over `bisect40` for `p = 1/3`.

See [docs/prox_scaling_cuda.md](docs/prox_scaling_cuda.md) for the full scaling report.

---

## Prox benchmark

Run a small local CPU benchmark:

```bash
python scripts/benchmark_prox.py \
  --device cpu \
  --p 0.3333333333333333 \
  --n 10000 \
  --repeats 5 \
  --warmup 2
```

Save results:

```bash
python scripts/benchmark_prox.py \
  --device cpu \
  --p 0.3333333333333333 \
  --n 10000 \
  --repeats 5 \
  --warmup 2 \
  --output results/prox_benchmark_p_1_3_dev.csv
```

Run a CUDA benchmark:

```bash
python scripts/benchmark_prox.py \
  --device cuda \
  --dtype float32 \
  --p 0.3333333333333333 \
  --n 1000000 \
  --methods poly,newton8,newton12,bisect40 \
  --repeats 30 \
  --warmup 10 \
  --output results/prox_benchmark_p_1_3_cuda.csv
```

Benchmark outputs include:

- median runtime
- mean runtime
- runtime standard deviation
- throughput in million elements per second
- finite-value check
- output summary statistics

---

## Accuracy validation

Validate methods against a high-iteration float64 Newton reference:

```bash
python scripts/validate_accuracy.py \
  --device cpu \
  --p 0.3333333333333333 \
  --n 10000 \
  --sample-n 2000
```

Save validation results:

```bash
python scripts/validate_accuracy.py \
  --device cpu \
  --p 0.3333333333333333 \
  --n 10000 \
  --sample-n 2000 \
  --output results/accuracy_p_1_3_dev.csv
```

On CUDA:

```bash
python scripts/validate_accuracy.py \
  --device cuda \
  --dtype float32 \
  --p 0.3333333333333333 \
  --n 1000000 \
  --sample-n 100000 \
  --methods poly,newton8,newton12,bisect40 \
  --output results/accuracy_p_1_3_cuda.csv
```

Accuracy outputs include:

- maximum absolute error
- mean absolute error
- median absolute error
- maximum relative error
- mean relative error
- median relative error

---

## Residual and objective validation

In addition to comparing outputs against a reference solver, the package can check the active-branch stationary residual:

```math
R(x;y) = |x| - |y| + \lambda p |x|^{p-1}.
```

For active entries, the validation script reports residual statistics, residual-based error bounds, and objective gaps relative to a high-iteration float64 Newton reference.

Run residual validation:

```bash
python scripts/validate_residual.py \
  --device cpu \
  --p 0.3333333333333333 \
  --n 10000 \
  --sample-n 2000
```

Save residual validation results:

```bash
python scripts/validate_residual.py \
  --device cpu \
  --p 0.3333333333333333 \
  --n 10000 \
  --sample-n 2000 \
  --output results/residual_p_1_3_dev.csv
```

On CUDA:

```bash
python scripts/validate_residual.py \
  --device cuda \
  --dtype float32 \
  --p 0.3333333333333333 \
  --n 1000000 \
  --sample-n 100000 \
  --methods poly,newton8,newton12,bisect40 \
  --output results/residual_p_1_3_cuda.csv
```

Residual validation outputs include:

- maximum active-branch residual
- mean active-branch residual
- maximum residual-based error bound
- mean residual-based error bound
- objective gap relative to reference

---

## Plot benchmark and validation results

After saving benchmark or validation results to CSV, figures can be generated with:

```bash
python scripts/plot_benchmark_results.py \
  --input results/prox_benchmark_p_1_3_dev.csv \
  --output-dir figures
```

```bash
python scripts/plot_validation_results.py \
  --input results/accuracy_p_1_3_dev.csv \
  --kind accuracy \
  --output-dir figures
```

```bash
python scripts/plot_validation_results.py \
  --input results/residual_p_1_3_dev.csv \
  --kind residual \
  --output-dir figures
```

For a local development run, use:

```bash
./scripts/make_dev_figures.sh
```

This generates development benchmark results, validation results, and figures under:

```text
results/
figures/
```

These outputs are intended as local artifacts and are ignored by Git.

---

## Visualization examples

### Prox curve for one p

```bash
python scripts/plot_prox_curves.py \
  --p 1/3 \
  --output figures/prox_curve_p_1_3.png
```

This plots:

```math
y \mapsto \operatorname{prox}_{\lambda |\cdot|^p}(y)
```

and marks the activation threshold.

### Polynomial approximation error

```bash
python scripts/plot_prox_error.py \
  --p 1/3 \
  --output figures/prox_error_p_1_3.png
```

This compares the polynomial evaluator against a Newton baseline and plots the absolute error.

### Prox curves for several p values

```bash
python scripts/plot_p_sweep_curves.py
```

This generates a comparison plot for:

```text
p = 1/3
p = 2/5
p = 3/4
```

---

## Examples

### Basic usage

```bash
python examples/basic_usage.py
```

This prints Lp prox constants and evaluates the prox operator on a small tensor.

### Compare prox methods

```bash
python examples/compare_methods.py
```

This compares the outputs of:

```text
poly
newton8
newton12
bisect40
```

against a selected reference method.

---

## Lp-Lasso Forward-Backward demo

The repository includes an application demo for the nonconvex Lp-Lasso problem:

```math
\min_x \frac{1}{2}\|Ax-b\|_2^2 + \lambda \sum_i |x_i|^p,
\qquad 0 < p < 1.
```

A Forward-Backward step is:

```math
y_k = x_k - \alpha A^\top(Ax_k-b),
```

```math
x_{k+1}
=
\operatorname{prox}_{\alpha \lambda |\cdot|^p}(y_k).
```

The goal of this demo is to show how different Lp prox evaluators behave when inserted into the same iterative optimization loop.

Run the demo:

```bash
python examples/lp_lasso_forward_backward_demo.py
```

This generates a convergence plot:

```text
figures/lp_lasso_fb_convergence_demo.png
```

Run a local CPU benchmark:

```bash
python scripts/benchmark_lp_lasso_fb.py \
  --device cpu \
  --p 1/3 \
  --m 400 \
  --n 800 \
  --sparsity 40 \
  --num-iters 100 \
  --repeats 5 \
  --warmup 1 \
  --output results/lp_lasso_fb_p_1_3_cpu_dev.csv
```

Plot the benchmark:

```bash
python scripts/plot_lp_lasso_fb_results.py \
  --input results/lp_lasso_fb_p_1_3_cpu_dev.csv \
  --output-dir figures
```

This benchmark measures end-to-end Forward-Backward runtime, including both gradient computation and prox evaluation. For dense design matrices on CPU, matrix multiplication may dominate runtime, so the speedup can be smaller than in isolated prox benchmarks.

The isolated prox benchmark and the Lp-Lasso Forward-Backward benchmark answer different questions:

| Benchmark | Question |
|---|---|
| `benchmark_prox.py` | Is the prox evaluator itself fast? |
| `benchmark_lp_lasso_fb.py` | Does the prox evaluator improve an end-to-end optimization workload? |

---

## Proximal-gradient helper

The package also includes a simple proximal-gradient step helper:

```python
import torch
from fast_lp_prox import make_pg_step

b = torch.randn(100_000)
x = torch.zeros_like(b)

step = make_pg_step(
    p=1/3,
    lam=1.0,
    alpha=0.8,
    method="poly",
)

for _ in range(100):
    x = step(x, b)
```

This is included as an application helper showing how the Lp prox evaluator can be used inside iterative optimization algorithms.

The proximal-gradient method is not the main contribution of this repository.

---

## Project structure

```text
fast-lp-prox/
├── src/
│   └── fast_lp_prox/
│       ├── __init__.py
│       ├── constants.py
│       ├── coeffs.py
│       ├── torch_impl.py
│       ├── residual.py
│       ├── pg.py
│       ├── benchmark_utils.py
│       └── coeff_tables/
├── scripts/
│   ├── generate_coeffs.py
│   ├── benchmark_prox.py
│   ├── validate_accuracy.py
│   ├── validate_residual.py
│   ├── plot_prox_curves.py
│   ├── plot_prox_error.py
│   ├── plot_p_sweep_curves.py
│   ├── plot_benchmark_results.py
│   ├── plot_validation_results.py
│   ├── benchmark_lp_lasso_fb.py
│   ├── plot_lp_lasso_fb_results.py
│   └── make_dev_figures.sh
├── examples/
│   ├── basic_usage.py
│   ├── compare_methods.py
│   └── lp_lasso_forward_backward_demo.py
├── tests/
├── docs/
│   ├── benchmark_a100.md
│   └── assets/
├── pyproject.toml
└── README.md
```

Local benchmark outputs are typically written to:

```text
results/
figures/
```

These directories are ignored by Git.

---

## Development notes

The original exploratory implementation was developed in Colab. This repository restructures that prototype into a reusable Python package with tests, scripts, examples, coefficient tables, benchmark utilities, validation scripts, and visualization tools.

The current implementation prioritizes clarity and reproducibility over aggressive low-level optimization.

Planned future improvements include:

- Colab-based GPU benchmark reproduction notebook
- Better benchmark summary tables
- Residual-based certified mode
- More rational exponent tables
- Optional Triton backend
- More complete mathematical documentation
- Cleaner separation between core prox implementation and application demos
- Packaging improvements for broader distribution

---

## Current limitations

- The project is still a research prototype.
- Benchmark results depend strongly on hardware, tensor size, dtype, backend, and workload shape.
- Built-in coefficient tables currently focus on a small set of rational exponents.
- The PyTorch implementation is the current main backend.
- CPU application benchmarks may be dominated by dense matrix multiplication rather than prox evaluation.
- CUDA benchmark results should be reproduced in a controlled environment before being used as official claims.
- The mathematical documentation is still under development.
- The project is not yet prepared for stable package distribution.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

Citation information is not available yet.

If this project becomes part of a paper, preprint, or technical report, add a `CITATION.cff` file and update this section.