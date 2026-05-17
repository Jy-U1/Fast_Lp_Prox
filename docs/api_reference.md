# API reference

This document summarizes the main public APIs in `fast-lp-prox`.

The package is currently a research prototype. The APIs below are the main interfaces intended for examples, scripts, benchmarks, and downstream experiments.

---

## Core prox evaluator

### `make_lp_prox`

Build a reusable Lp proximal-operator evaluator.

```python
from fast_lp_prox import make_lp_prox

prox = make_lp_prox(
    p=1/3,
    lam=1.0,
    method="poly",
)

x = prox(y)
```

Typical use case:

- repeated prox evaluation with fixed `p`, `lambda`, and backend method
- benchmark loops
- proximal-gradient / Forward-Backward iterations

Parameters:

| Parameter | Description |
|---|---|
| `p` | Lp exponent, with `0 < p < 1` |
| `lam` | regularization parameter |
| `method` | prox backend, such as `poly`, `newton8`, `newton12`, or `bisect40` |

Returns:

| Return value | Description |
|---|---|
| callable | A function that maps a PyTorch tensor `y` to `prox_{lambda |.|^p}(y)` coordinatewise |

Notes:

- For repeated calls, prefer `make_lp_prox(...)` over repeatedly calling `lp_prox(...)`.
- The returned callable preserves the input tensor's device and dtype when possible.
- For `method="poly"`, the implementation first tries to load a built-in coefficient table. If no suitable table is available, it falls back to constructing a Chebyshev approximation.

---

### `lp_prox`

Evaluate the Lp proximal operator in a one-off call.

```python
from fast_lp_prox import lp_prox

x = lp_prox(
    y,
    p=1/3,
    lam=1.0,
    method="poly",
)
```

Typical use case:

- quick experiments
- small examples
- scripts where setup overhead is not important

Parameters:

| Parameter | Description |
|---|---|
| `y` | input PyTorch tensor |
| `p` | Lp exponent, with `0 < p < 1` |
| `lam` | regularization parameter |
| `method` | prox backend |

Returns:

| Return value | Description |
|---|---|
| tensor | Coordinatewise Lp prox output with the same shape as `y` |

---

## Supported prox methods

| Method | Description |
|---|---|
| `poly` | Chebyshev polynomial evaluator for the active branch |
| `newton8` | 8-step safeguarded Newton baseline |
| `newton12` | 12-step safeguarded Newton baseline |
| `bisect40` | 40-step bisection baseline |

The `poly` method is the main high-throughput evaluator. The Newton and bisection methods are included as iterative baselines and reference-style methods for benchmarking and validation.

---

## Constants

### `make_lp_constants` / constants helper

The constants module provides the threshold and normalization constants used by the Lp prox operator.

Typical constants include:

| Name | Description |
|---|---|
| `tau` | activation threshold in `y` |
| `u_star` | critical active-branch quantity |
| `rho` | normalization scale |
| `eps_tau` | threshold-related numerical tolerance |
| `z_tau` | normalized threshold coordinate |

The scalar prox has the threshold structure:

```math
\operatorname{prox}_{\lambda |\cdot|^p}(y) = 0
\quad \text{when } |y| \le \tau_{p,\lambda}.
```

For active entries, the nonzero output is computed by solving or approximating the active root equation.

---

## Coefficient utilities

The coefficient utilities are used to generate, save, load, and retrieve Chebyshev tables for `method="poly"`.

Main functionality:

| Functionality | Description |
|---|---|
| Generate coefficients | Build Chebyshev coefficients for a target exponent `p` |
| Save coefficients | Store coefficient tables as JSON |
| Load coefficients | Read coefficient tables from disk |
| Built-in tables | Load coefficient tables packaged under `src/fast_lp_prox/coeff_tables/` |

Current built-in target exponents:

```text
p = 1/3
p = 2/5
p = 3/4
```

Coefficient tables are stored under:

```text
src/fast_lp_prox/coeff_tables/
```

The recommended way to generate a new table is through the script interface:

```bash
python scripts/generate_coeffs.py \
  --p 1/3 \
  --target-error 2e-7 \
  --max-degree 128 \
  --n-fit 8192 \
  --n-val 80001 \
  --verbose
```

---

## Residual and objective utilities

The residual module provides utilities for checking stationarity and objective quality.

### `objective_value`

Compute the scalar/vector objective value associated with the Lp prox subproblem:

```math
\frac{1}{2}(x-y)^2 + \lambda |x|^p.
```

Typical use case:

- validation scripts
- objective-gap checks against a reference solver

---

### `prox_stationary_residual`

Compute the active-branch stationary residual:

```math
R(x;y) = |x| - |y| + \lambda p |x|^{p-1}.
```

Typical use case:

- checking whether a nonzero prox output approximately satisfies the active root equation
- residual validation against Newton or bisection baselines

---

### `residual_error_bound`

Compute a residual-based diagnostic bound for active-branch error.

Typical use case:

- validation and debugging
- checking whether polynomial approximation errors remain within expected tolerance

---

### `objective_gap_vs_reference`

Compare objective values against a reference solution.

Typical use case:

- validating `poly` against high-accuracy Newton reference
- checking whether small output differences affect the prox objective

---

## Proximal-gradient helper

### `make_pg_step`

Build a reusable proximal-gradient style update with a fixed Lp prox backend.

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

This helper is included as an application demo. It is not the main contribution of the package.

---

### `pg_step`

Run a one-off proximal-gradient style step.

Typical use case:

- simple examples
- quick experiments
- tests

For repeated iterations, prefer `make_pg_step(...)`.

---

## Benchmark utilities

The benchmark utilities provide helper functions for scripts such as:

```text
scripts/benchmark_prox.py
scripts/validate_accuracy.py
scripts/validate_residual.py
scripts/benchmark_lp_lasso_fb.py
```

Typical functionality includes:

| Utility type | Description |
|---|---|
| Device selection | Choose CPU or CUDA device |
| Test vector generation | Generate reproducible prox benchmark inputs |
| Timing | Measure repeated runs with warmup and synchronization |
| Summary statistics | Collect runtime, throughput, finite checks, and output statistics |

Most users should interact with benchmark utilities through the scripts rather than importing them directly.

---

## Recommended usage patterns

### Repeated high-throughput prox evaluation

```python
from fast_lp_prox import make_lp_prox

prox = make_lp_prox(p=1/3, lam=1.0, method="poly")

for y in batches:
    x = prox(y)
```

### Baseline comparison

```python
from fast_lp_prox import make_lp_prox

prox_poly = make_lp_prox(p=1/3, lam=1.0, method="poly")
prox_ref = make_lp_prox(p=1/3, lam=1.0, method="newton12")

x_poly = prox_poly(y)
x_ref = prox_ref(y)
```

### Validation workflow

Use scripts rather than manual API calls:

```bash
python scripts/validate_accuracy.py \
  --device cuda \
  --dtype float32 \
  --p 0.3333333333333333 \
  --n 1000000 \
  --sample-n 100000 \
  --methods poly,newton8,newton12,bisect40 \
  --output results_cuda/accuracy_p_1_3_cuda.csv
```

```bash
python scripts/validate_residual.py \
  --device cuda \
  --dtype float32 \
  --p 0.3333333333333333 \
  --n 1000000 \
  --sample-n 100000 \
  --methods poly,newton8,newton12,bisect40 \
  --output results_cuda/residual_p_1_3_cuda.csv
```

---

## Notes

- The package focuses on fast Lp prox evaluation, not on proposing a new optimization algorithm.
- Forward-Backward and proximal-gradient utilities are included as application demos.
- `method="poly"` is the main high-throughput evaluator.
- `newton8`, `newton12`, and `bisect40` are useful baselines for benchmarking and validation.
- Built-in coefficient tables currently target a small set of rational exponents.
- The project is still a research prototype; APIs may evolve.
