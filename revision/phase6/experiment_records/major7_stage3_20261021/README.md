# Major 7 Stage 3: verified same-CPU numerical comparison

## Inputs and primary findings
- User-uploaded input files: `major7_cpu_all_methods.csv` (54 rows), `major7_cpu_accuracy_matched.csv` (30 selected rows), `major7_cpu_metadata.json`.
- CPU: macOS 15.4.1 x86_64, Python 3.10.5, NumPy 1.26.4, SciPy 1.15.3, PyTorch 2.2.2, single PyTorch thread.
- `n=128`, `mu=1`, `seed=0`, 3 warmups, 9 timings; each family receives the same float64 input array at each exponent.
- Offline MB Gamma-contour-grid preparation and Chebyshev fitting are measured separately and excluded from online times.
- The MB family is the paper-sign, independent NumPy/SciPy reimplementation of the contour correction; no fallback, not formal floating-point certification, no independent MATLAB equivalence check. MATLAB default `R=0.9 u_star` range is relaxed to cover all active inputs.
- MB passed the sampled complex-error and imaginary-part criterion at p=1/3, 2/5, 3/4; no tested passing configuration at p=0.9, 0.99, 0.999. Nonpassing MB times must not be ranked as accuracy-matched.
- At passing exponents, MB takes 19.046, 18.941, 346.784 ms per 128-input call. The comparable Newton times are 0.120, 0.118, 0.149 ms; these are implementation- and batch-dependent results, not an inherent complexity theorem.
- All 54 all-methods rows have finite values and zero support mismatches. All 9 recorded timing samples per row were re-aggregated: the saved median was reproduced. Selected rows refer exactly to rows in the 54-row data set.

## Why objective gaps are provided as an optional follow-up
The Stage-3 CSV records output error, active stationarity residual, support, and timing, but **does not include proximal-objective gaps**. `verify_stage3_objective_gaps.py` reconstructs the same input arrays and independently computes these gaps, rejecting every run whose input SHA-256 differs from Stage-3 or whose recomputed output error does not reproduce Stage-3. Do not merge gaps from a mismatched RNG/NumPy environment. The original recorded environment was NumPy 1.26.4. (A separate newer-NumPy environment did not reproduce the exact input hash.)

Use from the repository root after extracting the experiment-only ZIP:

```bash
python -m revision.phase6.scripts.verify_stage3_objective_gaps \
  --all revision/phase6/experiment_records/major7_stage3_20261021/major7_cpu_all_methods.csv \
  --selected revision/phase6/experiment_records/major7_stage3_20261021/major7_cpu_accuracy_matched.csv \
  --metadata revision/phase6/experiment_records/major7_stage3_20261021/major7_cpu_metadata.json \
  --output-dir revision/phase6/experiment_records/major7_stage3_20261021/objective_gaps
```

Expect 54 + 30 passing replay records; do not claim this additional verification succeeded before actually running it. The computed objective is the real scalar objective `0.5*(x-y)**2+|x|**p` for lambda=1; MB's **real part** is the scalar candidate, while the full complex norm error and imaginary discrepancy are separately validated by Stage-3.

## GitHub privacy & reproducibility
Only experiment files in this folder are suitable for the public GitHub record. **Keep manuscript and reviewer response TeX exclusively in the private Overleaf project.** Before declaring the Phase-6 archive complete, also preserve the Stage-2 input grids and metadata (the Stage-3 experiment selected MB configurations from those grids) and Phase-6 code. Stage-3 metadata identifies the repository HEAD as `3044afd5845d4f87e8e58d4d5b9362678dbfa51a`; it does not by itself demonstrate that all untracked Phase-6 code was included in that commit.
