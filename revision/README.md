# Scalar ℓp Proximity Operator: Numerical Benchmarks

## Overview

This directory documents the numerical experiments for evaluating the accuracy and computational efficiency of scalar ℓp proximal operators, with 0 < p < 1.

The benchmark compares three evaluation strategies:

- **Chebyshev:** Offline polynomial approximation followed by online Clenshaw evaluation.
- **Newton:** A clipped Newton iteration with configurable iteration count.
- **Bisection:** A bracketed root solver with configurable iteration count.

## Experiment 1: Accuracy–Runtime Tradeoff

The goal is to compare different proximal evaluators under a common accuracy criterion.

For Chebyshev, we vary the polynomial degree D. For Newton and bisection, we vary the iteration count K.

All methods are evaluated on identical input batches against an independent double-precision reference solver.

The recorded metrics include maximum and mean proximal approximation errors, normalized error, stationarity residual, support consistency, and median online runtime.

Offline coefficient generation and reference computations are excluded from online timing.

## Reproducibility

Run the unit tests:

```bash
python -m pytest -q tests/test_revision_accuracy_runtime.py
```

Run a small CPU benchmark:

```bash
python scripts/benchmark_accuracy_runtime.py \
  --device cpu \
  --dtype float64 \
  --p 1/3 \
  --n 1000 \
  --cheb-degrees 4,8,12 \
  --newton-iters-grid 2,4,6 \
  --bisect-iters-grid 8,16,24 \
  --n-fit 512 \
  --n-val 501 \
  --warmup 1 \
  --repeats 2 \
  --output results/cpu_smoke.csv
```

Generate an accuracy–runtime plot:

```bash
python scripts/plot_accuracy_runtime.py \
  --input results/cpu_smoke.csv \
  --output figures/cpu_smoke.png
```

## Experimental Scope

The current benchmark focuses on scalar proximal accuracy and online evaluation cost. Larger-scale GPU benchmarks and end-to-end optimization experiments can be run using the corresponding experimental scripts.

The benchmark distinguishes offline preparation costs from online evaluation costs and records numerical accuracy alongside runtime.