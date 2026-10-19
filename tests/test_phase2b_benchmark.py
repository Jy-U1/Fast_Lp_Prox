"""Phase 2B data protocol and analysis tests; no CUDA required."""
from __future__ import annotations

from argparse import Namespace
import random
import numpy as np
import pandas as pd
import pytest
import torch

from revision.phase2.benchmarks.run_benchmark_reliability import (
    CANDIDATE_GRID, configuration, measure_invocation, parse_p_labels,
    parse_positive_sizes, randomized_blocks, run,
)
from revision.phase2.benchmarks.rebuild_benchmark_artifacts import (
    _pairings, clustered_median_ci, rebuild, summarize_pairs,
)


def test_size_and_p_grid_validation():
    assert parse_positive_sizes("2048,512") == [512, 2048]
    assert parse_p_labels("1/3,3/4") == ["1/3", "3/4"]
    with pytest.raises(ValueError):
        parse_positive_sizes("1,1")
    with pytest.raises(ValueError):
        parse_positive_sizes("1,-2")
    with pytest.raises(ValueError):
        parse_p_labels("1/3,0.99")


def test_randomized_blocks_are_paired_and_reproducible():
    labels = ["chebyshev_D6", "newton_K3", "bisection_K20"]
    a = randomized_blocks(labels, 25, random.Random(7))
    b = randomized_blocks(labels, 25, random.Random(7))
    assert a == b
    assert all(set(x) == set(labels) and len(x) == len(labels) for x in a)
    assert any(x != labels for x in a)
    assert len({tuple(x) for x in a}) > 1
    # Complete cycles enforce balanced candidate positions.
    from collections import Counter
    counts = Counter((label, i) for block in a[:24] for i, label in enumerate(block))
    assert set(counts.values()) == {8}


def test_cpu_timing_records_wall_not_cuda():
    y = torch.linspace(1, 3, 64)
    s = measure_invocation(lambda t: t*t, y)
    assert s["wall_ms"] > 0
    assert np.isnan(s["cuda_event_ms"])


def _synthetic_tables(fail: bool = False):
    rows, quality = [], []
    for session in range(3):
        for candidate in ("chebyshev_D6", "newton_K3"):
            quality.append({"p_label":"1/3", "n":128, "session":session,
                            "configuration":candidate,
                            "accuracy_pass": not (fail and candidate=="newton_K3")})
        for block in range(5):
            for i, (candidate, ms) in enumerate((("chebyshev_D6", 0.75), ("newton_K3", 1.0))):
                rows.append({"p_label":"1/3", "n":128, "session":session,
                             "block":block, "configuration":candidate,
                             "cuda_event_ms":float("nan"), "wall_ms":ms,
                             "order_in_block":(i+block)%2})
    return pd.DataFrame(rows), pd.DataFrame(quality)


def test_paired_ratios_and_clustered_bootstrap():
    timing, quality = _synthetic_tables()
    paired = _pairings(timing, quality)
    assert len(paired) == 15
    assert np.allclose(paired["ratio_wall_cheb_over_newton"], .75)
    lo, hi = clustered_median_ci(paired, "ratio_wall_cheb_over_newton", resamples=100, seed=1)
    assert abs(lo - .75) < 1e-12 and abs(hi - .75) < 1e-12
    summary, session = summarize_pairs(paired, resamples=100)
    assert summary.iloc[0]["all_accuracy_checks_pass"]
    assert summary.iloc[0]["n_sessions"] == 3
    assert len(session) == 3


def test_failed_accuracy_gate_remains_visible():
    timing, quality = _synthetic_tables(fail=True)
    paired = _pairings(timing, quality)
    summary, _ = summarize_pairs(paired, resamples=20)
    assert not summary.iloc[0]["all_accuracy_checks_pass"]
    assert not paired["accuracy_matched_in_checked_subset"].any()


def test_small_cpu_pipeline(tmp_path):
    out = tmp_path / "phase2b"
    args = Namespace(device="cpu", dtype="float32", p_grid="1/3", sizes="64,128",
        sessions=2, blocks=2, warmup=0, accuracy_n=64,
        target_norm_error=1e-6, lam=1.0, active_ratio=.9, ymax_factor=10.0,
        fit_n=64, validation_n=129, seed=137, output_dir=str(out), overwrite=False)
    metadata = run(args)
    assert metadata["n_raw_timing_rows"] == 2*2*2*2
    assert metadata["n_accuracy_rows"] == 2*2*2
    result = rebuild(out, bootstrap_resamples=20)
    assert len(result) == 2
    assert (out / "paired_blocks.csv").exists()
    assert (out / "figures/paired_ratio_vs_size.png").exists()
    with pytest.raises(FileExistsError):
        run(args)
