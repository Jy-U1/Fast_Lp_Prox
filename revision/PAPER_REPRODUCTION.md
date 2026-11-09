# Paper reproduction: revised Fast-Lp-Prox experiments

This is a **small research-code reproduction index**, not a new software framework.
The algorithms remain in `src/fast_lp_prox` and the existing `revision/phase1`–`phase5` directories.

## What is already public?

| Manuscript result | Public source | Reproduction |
|---|---|---|
| Accuracy-matched scalar comparisons (Table 3) | `revision/phase1/tables/formal_best_comparison.csv` and Phase-1 raw inputs | Phase-1 README and scripts |
| Expanded scalar / near-threshold tests | `revision/experiment_records/phase4_phase5_20261020/raw/*Phase5*.zip`; machine-readable `summaries/phase5_*.csv` | Phase-5 pinned notebook, benchmark and archive verifier |
| Lasso FB timing (Table 9) | `.../summaries/phase4_lasso_all_methods_verified.csv` | `python -m revision.paper_reproduction` |
| Lasso recovery and offline cost (Table 10) | The same Phase-4 summary and `phase4_lasso_six_case_summary.csv` | `python -m revision.paper_reproduction` |
| Lasso iteration curves | Phase-4 original ZIP (per-run trajectories) | Phase-4 pinned notebook and diagnostic scripts |

The Phase-4/5 record (including original ZIPs, notebooks, checksums and a
standard-library verifier) lives at
`revision/experiment_records/phase4_phase5_20261020/`.
**Do not confuse the old Phase-3 README's default tolerance `1e-4` with
the final Phase-4 campaign's independently checked `1e-6`.**

## Verify the public data without a GPU

From the repository root:

```bash
python revision/experiment_records/phase4_phase5_20261020/verify_experiment_record.py
python -m revision.paper_reproduction
python -m revision.paper_reproduction --output-dir results/paper_tables_new
python -m pytest -q tests/test_paper_reproduction.py
```

The verifier checks SHA-256, ZIP integrity, Phase-4 stop/replay counts and Phase-5
accuracy/support counts. The read-only table reconstruction checks all six Phase-4
cases and 18 method summaries and writes two CSVs **only in a new output directory**.
It is a reformatting of frozen measured data; it is *not* a new experimental run
and does not establish numerical accuracy independently of the archived records.

Table 9 uses the mean of per-seed loop replay medians (no checks), the mean
of per-seed checked-stop wall-time medians, and separately instrumented
gradient/prox stage medians. Instrumented stages are **not** components of the
uninstrumented loop timing and need not sum to it.

Table 10 uses the Chebyshev reconstruction **at the first independently
verified stopping checkpoint**, not the final (500-iteration) diagnostic
iterate. The archive reports active *fractions*; the paper prints percentages.
The amortization count excludes an advantage when the Chebyshev evaluator
is not faster. The numerical tables are empirical A100 results, not floating-
point certification or general convergence proofs.

## What is intentionally not included

Development logs, failed exploratory runs, and repetitive diagnostic plots
are not required for the paper. The public record retains the tested grids,
raw source data and reference checks needed to inspect the reported results.
The old repository-root fixed-iteration performance snapshot is historical
and is **not** the revised manuscript's accuracy-matched comparison.

The signed mean objective difference from original Phase 1 and the
positive-part mean objective gap used in Phase 5 have **different
mathematical definitions**. They must not be silently merged or compared
as if they were the same statistic.

GPU benchmarking should be rerun only when new evidence is needed; numerical
timing across different hardware/software environments will not be
bitwise reproducible.
