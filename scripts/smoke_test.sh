#!/usr/bin/env bash
set -e

python -m pytest -q

python examples/basic_usage.py
python examples/compare_methods.py

python scripts/benchmark_prox.py \
  --device cpu \
  --p 0.3333333333333333 \
  --n 10000 \
  --repeats 3 \
  --warmup 1

python scripts/validate_accuracy.py \
  --device cpu \
  --p 0.3333333333333333 \
  --n 10000 \
  --sample-n 1000

python scripts/validate_residual.py \
  --device cpu \
  --p 0.3333333333333333 \
  --n 10000 \
  --sample-n 1000

python examples/lp_lasso_forward_backward_demo.py

echo "Smoke test passed."