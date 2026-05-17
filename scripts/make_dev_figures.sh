#!/usr/bin/env bash
set -e

python scripts/benchmark_prox.py \
  --device cpu \
  --p 0.3333333333333333 \
  --n 10000 \
  --repeats 5 \
  --warmup 2 \
  --output results/prox_benchmark_p_1_3_dev.csv

python scripts/validate_accuracy.py \
  --device cpu \
  --p 0.3333333333333333 \
  --n 10000 \
  --sample-n 2000 \
  --output results/accuracy_p_1_3_dev.csv

python scripts/validate_residual.py \
  --device cpu \
  --p 0.3333333333333333 \
  --n 10000 \
  --sample-n 2000 \
  --output results/residual_p_1_3_dev.csv

python scripts/plot_prox_curves.py \
  --p 1/3 \
  --output figures/prox_curve_p_1_3.png

python scripts/plot_prox_error.py \
  --p 1/3 \
  --output figures/prox_error_p_1_3.png

python scripts/plot_p_sweep_curves.py

python scripts/plot_benchmark_results.py \
  --input results/prox_benchmark_p_1_3_dev.csv \
  --output-dir figures

python scripts/plot_validation_results.py \
  --input results/accuracy_p_1_3_dev.csv \
  --kind accuracy \
  --output-dir figures

python scripts/plot_validation_results.py \
  --input results/residual_p_1_3_dev.csv \
  --kind residual \
  --output-dir figures

echo "Done. Figures saved to figures/"