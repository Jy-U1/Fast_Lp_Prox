#!/usr/bin/env bash
set -euo pipefail
# Run from cloned public Fast_Lp_Prox repo pinned to commit 7a1e32b.
# All CSVs are written under results/phase1_reproduced, never to the archive.
mkdir -p results/phase1_reproduced

# p = 1/3
python scripts/benchmark_accuracy_runtime.py \
  --device cuda \
  --dtype float32 \
  --p 1/3 \
  --lam 1.0 \
  --n 1000000 \
  --active-ratio 0.9 \
  --ymax-factor 10.0 \
  --seed 0 \
  --cheb-degrees 4,5,6,7,8,10,12 \
  --newton-iters-grid 2,3,4,5,6 \
  --bisect-iters-grid 12,16,20,24,28,32 \
  --n-fit 8192 \
  --n-val 80001 \
  --warmup 10 \
  --repeats 30 \
  --target-norm-error 1e-06 \
  --output results/phase1_reproduced/revision_phase1_A100_p0333_1M.csv

# p = 2/5
python scripts/benchmark_accuracy_runtime.py \
  --device cuda \
  --dtype float32 \
  --p 2/5 \
  --lam 1.0 \
  --n 1000000 \
  --active-ratio 0.9 \
  --ymax-factor 10.0 \
  --seed 0 \
  --cheb-degrees 4,5,6,7,8,10,12,16 \
  --newton-iters-grid 2,3,4,5,6 \
  --bisect-iters-grid 12,16,20,24,28,32 \
  --n-fit 8192 \
  --n-val 80001 \
  --warmup 10 \
  --repeats 30 \
  --target-norm-error 1e-06 \
  --output results/phase1_reproduced/revision_phase1_A100_p0400_1M.csv

# p = 3/4
python scripts/benchmark_accuracy_runtime.py \
  --device cuda \
  --dtype float32 \
  --p 3/4 \
  --lam 1.0 \
  --n 1000000 \
  --active-ratio 0.9 \
  --ymax-factor 10.0 \
  --seed 0 \
  --cheb-degrees 4,6,8,10,12,14,16,18,20,24 \
  --newton-iters-grid 2,3,4,5,6,8 \
  --bisect-iters-grid 12,16,20,24,28,32 \
  --n-fit 8192 \
  --n-val 80001 \
  --warmup 10 \
  --repeats 30 \
  --target-norm-error 1e-06 \
  --output results/phase1_reproduced/revision_phase1_A100_p0750_1M.csv

# p = 0.9
python scripts/benchmark_accuracy_runtime.py \
  --device cuda \
  --dtype float32 \
  --p 0.9 \
  --lam 1.0 \
  --n 1000000 \
  --active-ratio 0.9 \
  --ymax-factor 10.0 \
  --seed 0 \
  --cheb-degrees 12,16,20,24,28,32,40 \
  --newton-iters-grid 2,3,4,5,6,8 \
  --bisect-iters-grid 16,20,24,28,32 \
  --n-fit 8192 \
  --n-val 80001 \
  --warmup 10 \
  --repeats 30 \
  --target-norm-error 1e-06 \
  --output results/phase1_reproduced/revision_phase1_A100_p0900_1M.csv

# p = 0.99
python scripts/benchmark_accuracy_runtime.py \
  --device cuda \
  --dtype float32 \
  --p 0.99 \
  --lam 1.0 \
  --n 1000000 \
  --active-ratio 0.9 \
  --ymax-factor 10.0 \
  --seed 0 \
  --cheb-degrees 48,50,52,54,56,60,64,72,80 \
  --newton-iters-grid 3,4,5,6,8 \
  --bisect-iters-grid 16,20,24 \
  --n-fit 8192 \
  --n-val 80001 \
  --warmup 10 \
  --repeats 30 \
  --target-norm-error 1e-06 \
  --output results/phase1_reproduced/revision_phase1_A100_p0990_1M.csv

# p = 0.999
python scripts/benchmark_accuracy_runtime.py \
  --device cuda \
  --dtype float32 \
  --p 0.999 \
  --lam 1.0 \
  --n 1000000 \
  --active-ratio 0.9 \
  --ymax-factor 10.0 \
  --seed 0 \
  --cheb-degrees 104,108,112,116,120,124,128,144,160 \
  --newton-iters-grid 3,4,5,6,8 \
  --bisect-iters-grid 16,20,24,28 \
  --n-fit 8192 \
  --n-val 80001 \
  --warmup 10 \
  --repeats 30 \
  --target-norm-error 1e-06 \
  --output results/phase1_reproduced/revision_phase1_A100_p0999_1M.csv

