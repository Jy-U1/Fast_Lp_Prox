# Phase-4 launcher hotfix (2026-10-20)

The first Phase-4 launcher wrote `phase4_launch.json` into its case output
folder before invoking the Phase-3 driver with `--output-dir` set to that
same folder. The Phase-3 driver intentionally refuses nonempty output folders.
This made the smoke test fail with `FileExistsError` before its experiment.

The revised launcher keeps case-level Phase-4 logs and provenance in the parent
folder, while Phase-3 results live inside a fresh `phase3_results/` child.
The test suite contains a stub Phase-3 driver enforcing the real no-overwrite
rule; this catches the regression without requiring CUDA.

Existing failed run artifacts must be preserved. Rerun the smoke test using a
**new** output root, not `--overwrite`:

```
python -m pytest -q tests/test_phase4_reporting.py tests/test_phase3_lasso.py
python revision/phase4/scripts/run_reviewer2_lasso_matrix.py --mode smoke --execute --output-root results/reviewer2_phase4_lasso_retry1
```

If any other failure occurs, inspect:
`results/reviewer2_phase4_lasso_retry1/p03333_m80_n160/phase4_stderr.log`.

This hotfix does not affect the archived Phase-1 scalar data or Phase-3 runs;
it does not validate GPU timing or scientific outcomes.
