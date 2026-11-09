# Phase 6: reproducible Mellin--Barnes CPU comparator

This directory contains **public experimental materials only**. The paper,
reviewer response, journal template, bibliography and private Overleaf content
are deliberately excluded.

## Core sources and provenance

- Original source repository: https://github.com/lshen03/Lp-Proximity-Operator
- Paper: https://arxiv.org/abs/2606.31951 (v2, Mellin--Barnes section)
- Independent Python implementation: `revision/phase6/mb_port.py`
- CPU pilot: `revision/phase6/scripts/benchmark_mb_cpu.py`
- Stage 2 grid: `revision/phase6/scripts/sweep_mb_accuracy.py`
- Stage 3 same-CPU comparison: `revision/phase6/scripts/finalize_mb_comparison.py`
- A notebook for the initial CPU pilot: `revision/phase6/notebooks/Phase6_MB_CPU_Colab.ipynb`

The Python implementation was derived from the formulas and behavior of the
public MATLAB source. **It has not been established as numerically identical to
the original MATLAB program.** The paper-consistent MB correction sign is
kept distinct from the published MATLAB sign. Source's R=0.9*u_star default
is enlarged when explicitly testing the full active domain. The contour
integral uses nonvalidated SciPy trapezoidal quadrature; its numerical output
must not be described as a *certified* floating-point bound. A separate
MB-plus-bisection variant exists in the pilot and is never counted as raw MB.

## Records

- `major7_stage2_20261021/standard/` contains 48 original N/T configurations
  from six exponents.
- `major7_stage2_20261021/longT/` contains 12 additional configurations.
- `major7_stage3_20261021/` contains the 54-row all-method table and 30-row
  selected table, environment and input SHA-256 metadata, with a digest list.

Stage 3 reports the same CPU, same dtype (`float64`), same inputs, n=128,
3 warmups, 9 timing repeats, and p in {1/3, 2/5, 3/4, 0.9, 0.99, 0.999}.
It measures **online** runtime separately from MB grid preparation and
Chebyshev fitting. Computational conclusions are specific to the recorded
NumPy/SciPy and PyTorch implementations; these CPU times should NOT be directly
compared with Phase 5's A100 float32 n=1,000,000 numbers.

Stage-3's `git_commit` in the recorded metadata describes the local HEAD *at
measurement time*. The separately committed Phase-6 source may have been
untracked at that moment. The public code commit and its verification make
all source files available for inspection, but this is not proof of a fully
clean working tree at measurement time.

## Final verification, after archiving the Stage-2 raw files

From the repository root:

```bash
python -m revision.phase6.scripts.verify_phase6_archive
```

Expected:

```
Stage 3: 54 raw rows, 30 selected, 5 SHA-256 files, 0 support mismatches: PASS
Stage 2: 48+12 grid rows, 6+4 selected rows, 7 SHA-256 files, linked to Stage-3: PASS
PASS: Phase-6 public experimental archive is independently restorable
```

If Stage-2 files are missing, use
`python -m revision.phase6.scripts.archive_stage2_records` on the machine
holding the originally measured `results/reviewer2_major7_mb_stage2` and
`results/reviewer2_major7_mb_stage2_longT` directories. The helper checks
input SHA-256 against Stage 3 before copying files. Do **not** synthesize the
original measurements from the printed console log.

## Optional stage-3 objective-gap recheck

Use `python -m revision.phase6.scripts.verify_stage3_objective_gaps` with
arguments in the Stage-3 README. This requires the original compatible
NumPy RNG environment; it rejects mismatched input SHA-256 rather than
pretending to recreate the original records. The Stage-3 timing/error table
already stands on its own without this additional diagnostic.
