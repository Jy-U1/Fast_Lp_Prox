# Phase 2B — Interleaved Benchmark Reliability

This add-on does **not** modify Phase 1 or Phase 2A. It compares the *same eager PyTorch online evaluators* used in Phase 1 with a new **randomized-block, multi-session GPU timing protocol**. It is public experimental code: no manuscript draft or reviewer correspondence is included.

## Question and protocol

We test whether the small Phase 1 timing differences between degree-truncated Chebyshev and Newton survive a more controlled protocol. The main accuracy-matched competitors from Phase 1 are:

| p | Chebyshev | Newton | Additional comparison |
|---|---|---|---|
| 1/3 | D=6 | K=3 | — |
| 2/5 | D=6 | K=3 | — |
| 3/4 | D=12 | K=4 | D=14, with more accuracy margin |

For each `(p, input size, session)`, candidates share an identical input tensor with the same active fraction/distribution as in Phase 1. All coefficients, scalars and CUDA tensors are loaded before timing. After warming each candidate, every block runs each candidate **exactly once in randomized, position-balanced order**; multiple blocks comprise a session. Both CUDA-event stream duration and synchronized host wall-clock duration are kept **per invocation**, along with block order and the input seed. The main GPU summaries use the **paired Chebyshev/Newton event-time ratio within each block**, not an independently calculated ratio of unrelated medians.

Sessions are repeated measurement groups **within one Colab runtime**; this does not replace experiments on separately restarted machines/runtimes. Reported bootstrap confidence bands resample sessions and within-session blocks. With few sessions these bands are *descriptive* and should not be interpreted as a hardware-independent statistical guarantee. Record the GPU model (e.g., T4 vs A100); do **not** mix devices in a single comparison or assert a particular model without inspecting `metadata.json`.

**Accuracy gate:** On a shared subset of each timed input tensor, both candidate outputs are compared to a separate float64 bracketed-bisection reference. The raw accuracy checks include maximum normalized output error and support mismatches. A group can only be called *accuracy-matched on its checked subset* if **both candidates pass**. The subset is not a uniform-domain error certificate. Phase 1 has separately archived full-million-input accuracy validation. In particular, p=3/4 D=12 is near the 1e-6 error boundary; results may fail for some sessions: keep such failures, do not hide or average them away.

CUDA event timing does **not** isolate pure arithmetic cost: with eager PyTorch, the event interval can include host submission gaps and scheduling effects. Wall time additionally includes Python/event work and explicit synchronization. Input-size scaling helps diagnose overhead dominance, but cannot determine its cause without lower-level kernel profiling. No CUDA Graph, kernel fusion, torch.compile, or clock controls are used.

## Install and CPU smoke

From the repository root with editable package installed:

```bash
python -m pytest -q tests/test_phase2b_benchmark.py
python revision/phase2/benchmarks/run_benchmark_reliability.py \
  --device cpu --p-grid 1/3 --sizes 512,2048 \
  --sessions 2 --blocks 4 --warmup 1 \
  --fit-n 256 --validation-n 501 --accuracy-n 512 \
  --output-dir results/phase2/reliability_smoke
python revision/phase2/benchmarks/rebuild_benchmark_artifacts.py \
  --root results/phase2/reliability_smoke --bootstrap-resamples 100
```

CPU smoke verifies operation and output generation, **not GPU timing claims**. Outputs are never overwritten by default; use a new directory or explicitly `--overwrite`.

## Formal GPU run in Colab

```bash
python revision/phase2/benchmarks/run_benchmark_reliability.py \
  --device cuda --dtype float32 \
  --p-grid 1/3,2/5,3/4 \
  --sizes 10000,100000,1000000,10000000 \
  --sessions 6 --blocks 30 --warmup 10 \
  --accuracy-n 8192 --target-norm-error 1e-6 \
  --fit-n 8192 --validation-n 80001 \
  --output-dir results/phase2/benchmark_reliability/formal
python revision/phase2/benchmarks/rebuild_benchmark_artifacts.py \
  --root results/phase2/benchmark_reliability/formal
```

The notebook `revision/phase2/notebooks/Phase2B_Benchmark_Reliability.ipynb` provides CPU smoke, opt-in full CUDA run, review and ZIP export. It clones the current `revision/major-experiments` branch on Colab; metadata captures the exact SHA. Do not overwrite a formal experiment if an earlier one exists—use a distinct directory/session or archive first.

## Outputs

- `timings_raw.csv`: one row per online invocation, both timers, randomized order and session/seed.
- `accuracy_checks.csv`: accuracy checks for every `(p,n,session,candidate)`, independently of timing.
- `metadata.json`: machine, code revision, seeds and numerical configuration.
- `paired_blocks.csv`: within-block paired ratios and pass/fail gate.
- `paired_summary.csv`: median paired ratio and empirical two-level bootstrap interval.
- `per_session_summary.csv`: session-level variation.
- `figures/paired_ratio_vs_size.png/.pdf` and `SUMMARY.md`.
- `manifest.json`: SHA-256 checksums, for archival verification.

**No reviewer response or private notes should be added to this public repository.** Keep ZIP backups in your local `Revision/Archives/`, not tracked in Git.
