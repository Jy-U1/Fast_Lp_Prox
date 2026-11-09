# Phase 6 (Reviewer 2 Major 7): published MATLAB MB evaluator in Python

**Scope:** Reproduce the *actual residue-corrected Mellin--Barnes (MB)* algorithm
in the reference implementation, and distinguish it from Phase 5's
`ResidualSeriesBisection` backend, which has **no MB integral**.

**Original sources:**

- Paper v2: https://arxiv.org/html/2606.31951v2 (Sec. 5.2, Eqs. (20)--(28));
- Public code: https://github.com/lshen03/Lp-Proximity-Operator,
  especially `xplus_lp_hybrid.m`, `xplus_lp_series.m`, and `run_partB_demo.m`.
- Source inspection date: 2026-10-20. The original repository does not
  display a software license as of that check; the distributed Python code is
  an **independent implementation of the formulas**, not a republished copy
  of the original `.m` sources. Cite the original paper/repository.

## Scientific issues found in the released code

1. **Correction sign.** The paper defines
   `H(s) = - Gamma(-s) Gamma((2-p)s-1) / Gamma((1-p)s)` and **adds** the
   `H(s)*(-eps)^s` contour integral to the truncated series. The public MATLAB
   `mb_segment` instead computes the Gamma ratio **without the minus** while
   also adding it. `MBPort(mode="matlab")` reproduces this released formula;
   `MBPort(mode="paper")` follows the stated paper formula. Both are logged.

2. **Default contour-radius cap.** Source default `R=0.9*u_star`; however,
   at proximal activation `eps/u_star=q_p=p*2**(1-p)`, which is >0.9 at large p.
   Consequently the raw MATLAB default rejects valid near-threshold inputs.
   `--R-policy matlab-default` demonstrates this; `--R-policy full-active`
   explicitly raises `R` to `max(0.9,(1+q_p)/2)*u_star`, which covers the
   active domain without changing the mathematical contour formula.
   This is a *disclosed configuration change*, not the MATLAB default.

3. **Certificate strength.** Source estimates the Gamma envelope using
   maxima over finite `t` and `n` grids, and numerical integration uses
   fixed trapezoidal panels. Those finite maxima do **not** give verified
   global bounds; quadrature error is not certified. This agrees with the
   paper v2 warning that verified majorants and validated quadrature are
   required for machine-verifiable error certificates (Sec. 5.2 after Eq. 27).
   We label all such results *empirical*, not certified.

4. **Finite-contour complex values.** The truncated MB formula can leave a
   non-negligible imaginary component. `x_complex` preserves it; reporting
   tests include both the real-part error and full complex error. The
   benchmark refuses to count a raw MB candidate as accurate unless both
   full complex error and imaginary part satisfy the stated tolerances.

5. **Residual check versus raw MB.** `mode="paper", tol=None` measures only
   the paper MB candidate. `mode="paper", tol=...` checks bracket membership,
   active stationary residual and imaginary part, then bisects on failure.
   This last method is a *separate safeguarded implementation* and its runtime
   includes fallback. Neither uses interval arithmetic.

## Run locally (from existing Fast_Lp_Prox root)

```bash
python -m pip install numpy scipy pytest mpmath
python -m pytest -q tests/test_phase6_mb_port.py
python -m revision.phase6.scripts.benchmark_mb_cpu \
  --p 0.3333333333333333,0.4,0.75,0.9,0.99,0.999 \
  --lam 1 --n 128 --N 3 --T 60 --nt 4000 \
  --include-phase5 yes --warmup 2 --repeats 5 \
  --R-policy full-active \
  --output-dir results/reviewer2_major7_mb_cpu
```

- `--include-phase5 yes` additionally evaluates the pre-existing **project's**
  Chebyshev, Newton and Bisection PyTorch CPU float64 implementations on the
  **identical input tensor**, choosing the previously tested Chebyshev degrees
  `6,6,14,24,54,128` and Newton iteration budgets `3,3,4,4,5,4`.
- Use `--include-phase5 no` to run this package independently of the rest
  of FastLp (only NumPy/SciPy required, plus pytest/mpmath for tests).
- For the public MATLAB default cap, switch to `--R-policy matlab-default`.
  Rows rejected by the original domain test are flagged `REJECTED`, not
  misreported as numerical inaccuracy.
- To try the original MATLAB's *sampled* automatic `N,T` search (slower),
  add `--matlab-autoselect --selection-eta 1e-8`.
  Its empirical tail indicators are **not proofs**.
- To explore quadrature convergence, repeat on *the same inputs* with
  different `--N`, `--T`, `--nt`; avoid varying `nt` independently of `T`
  without checking the integration step size `2T/nt`.

### Recommended workflow

1. Small functionality test: `p=2/3`, `n=64`, `N=3`, `T=60`, `nt=4000`.
2. Six-exponent CPU **correctness** sweep (the default). The pilot `N=3,T=60`
   is *intentionally coarse*; it cannot justify speed-vs-accuracy claims.
3. Run a small accuracy-matched contour grid, e.g. `N in {3,6,12}` and
   `T in {60,240,960}`, choosing `nt` so `2T/nt <= 0.03` if feasible.
4. Keep candidate-only and safeguarded/fallback timing rows separate.
5. *Only after validating the original comparator* prepare the revised
   manuscript and response. Do not mix these CPU timings with archived A100
   PyTorch float32 timings (different hardware, precision, batch sizes).

## Files and metrics

`revision/phase6/mb_port.py` is the independent algorithm module.
`revision/phase6/scripts/benchmark_mb_cpu.py` produces:

- `mb_major7_cpu_results.csv`: method, `N,T,nt,R`, raw times, preparation time,
  complex and real normalized errors, imaginary part, support, residual,
  objective gap, fallback count, and pass/fail;
- `mb_major7_metadata.json`: experiment parameters, Git commit, version info,
  provenance, input SHA-256; the shared-input generator matches Phase 5;
- `tests/test_phase6_mb_port.py`: numerical and regression tests, including
  a strict test that the MATLAB sign differs from the paper sign.

**Colab:** open `notebooks/Phase6_MB_CPU_Colab.ipynb`, select a **CPU** runtime,
then upload the companion ZIP package when requested. An A100 provides no
benefit for SciPy CPU complex log-Gamma in this port.

## Important limitations

- The project has not run the actual `.m` file in MATLAB/Octave here. The two
  formulas were matched to the **public source text**, then tested against
  an independent float64 bisection numerical reference and mpmath roots.
  Source-to-Python numerical equality is not yet experimentally confirmed.
- The preprint's published MATLAB has no reproducible software environment
  manifest. We record the Python SciPy version and the code used instead.
- Floating-point `tol` and sampled Gamma bounds do not imply a rigorous
  certification; claiming a formally certified method requires interval
  implementation and proof of quadrature error bounds.
- This experiment addresses **Major 7** only, not the full inexact-FB
  convergence issue of Major 6.
