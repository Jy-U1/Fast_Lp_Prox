"""Phase-4 additional Lasso diagnostics: CPU-only unit and isolated integration tests."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

SCRIPT = (Path(__file__).resolve().parents[1] / "revision/phase4/scripts" /
          "diagnose_lasso_outputs.py")
spec = importlib.util.spec_from_file_location("phase4_diagnostics", SCRIPT)
diag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diag
assert spec.loader is not None
spec.loader.exec_module(diag)


def test_regenerate_truth_matches_phase3_random_sequence():
    m, n, support, seed = 6, 9, 3, 77
    x, loc = diag.regenerate_truth(seed=seed, m=m, n=n, support=support)
    rng = np.random.default_rng(seed)
    _ = rng.standard_normal((m, n)).astype(np.float32) / np.float32(np.sqrt(m))
    truth_loc = np.sort(rng.choice(n, size=support, replace=False))
    truth_values = rng.standard_normal(support)
    assert np.array_equal(loc, truth_loc)
    assert np.allclose(x[loc], truth_values, atol=0, rtol=0)
    assert np.count_nonzero(x) == support


def test_truth_metrics_exact_and_zero_iterate():
    truth = np.array([3.0, 0.0, -4.0, 0.0])
    perfect = diag.reconstruction_stats(truth, truth)
    assert perfect["recovery_l2_relative"] == 0
    assert perfect["support_precision"] == perfect["support_recall"] == 1.0
    assert perfect["support_jaccard_truth"] == 1.0
    zero = diag.reconstruction_stats(np.zeros_like(truth), truth)
    assert zero["recovery_l2_relative"] == 1.0
    assert zero["active_coordinate_fraction"] == 0.0
    assert zero["support_recall"] == 0.0
    noisy = diag.reconstruction_stats(np.array([3.0, 2.0, -4.0, 0.0]), truth)
    assert noisy["support_true_positive"] == 2
    assert noisy["support_false_positive"] == 1
    assert noisy["support_precision"] == pytest.approx(2 / 3)
    assert noisy["support_recall"] == 1.0


def toy_problem():
    A = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], dtype=torch.float32)
    b = torch.tensor([1.0, -2.0], dtype=torch.float32)
    return SimpleNamespace(A=A, b=b, Ad=A.double(), bd=b.double(),
                           n=3, alpha=0.25, lambda_reg=0.05, p=0.5)


def toy_objective(x, problem):
    y = x.double()
    return (0.5 * torch.sum((problem.Ad @ y - problem.bd)**2)
            + problem.lambda_reg * torch.sum(torch.abs(y)**problem.p))


def test_prox_input_active_fraction_differs_from_solution_support():
    prob = toy_problem()
    x = torch.zeros(3)
    frac = diag.proximal_input_active_fraction(x, prob, tau=0.1)
    assert frac == pytest.approx(2/3)
    assert diag.reconstruction_stats(x.numpy(), np.array([1.0, 0, 0]))["active_coordinate_fraction"] == 0


def test_recovery_trace_and_stage_timing_cpu():
    prob = toy_problem()
    proximal = lambda x: torch.where(torch.abs(x) > 0.2, x, torch.zeros_like(x))
    truth = np.array([1.0, 0.0, 0.0])
    trace = diag.replay_recovery_trajectory(prob, proximal, xtrue=truth,
                                            steps=4, checkpoint_every=2,
                                            tau=0.2, objective=toy_objective)
    assert [r["iteration"] for r in trace] == [0, 2, 4]
    assert trace[0]["recovery_l2_relative"] == 1.0
    assert all(0 <= x["active_coordinate_fraction"] <= 1 for x in trace)
    assert all(np.isfinite(x["objective"]) for x in trace)
    times = diag.instrumented_stage_replays(prob, proximal, steps=4, repeats=2, warmup=1)
    assert len(times) == 2
    assert all(t["gradient_forward_ms"] >= 0 and t["prox_ms"] >= 0 for t in times)
    assert all(t["instrumented_total_ms"] > 0 for t in times)
    assert all(t["prox_fraction"] + t["gradient_forward_fraction"] == pytest.approx(1) for t in times)
    assert all(t["timing_semantics"] == "cpu_per_step_wall_diagnostic" for t in times)


def test_validation_checks_objective_and_true_support():
    final = {"objective": 0.3, "nonzero_count": 2, "support_jaccard_truth": 0.5}
    row = pd.Series({"final_objective": 0.30000001, "final_nonzero_count": 2,
                     "support_jaccard_vs_truth": 0.5})
    assert diag.verify_terminal(final, row, rtol=1e-4, atol=1e-5)["phase3_replay_check_pass"]
    changed = row.copy()
    changed["final_nonzero_count"] = 3
    assert not diag.verify_terminal(final, changed, rtol=1e-4, atol=1e-5)["phase3_replay_check_pass"]
    changed = row.copy()
    changed["final_objective"] = 0.4
    assert not diag.verify_terminal(final, changed, rtol=1e-4, atol=1e-5)["phase3_replay_check_pass"]


def test_stage_timing_rejects_bad_parameters():
    prob = toy_problem()
    with pytest.raises(ValueError):
        diag.instrumented_stage_replays(prob, lambda x: x, steps=10, repeats=0, warmup=1)
    with pytest.raises(ValueError):
        diag.replay_recovery_trajectory(prob, lambda x: x, xtrue=np.ones(3),
                                       steps=-1, checkpoint_every=1, tau=0.1, objective=toy_objective)


def test_end_to_end_standalone_postprocessor_with_mock_phase3(tmp_path, monkeypatch):
    # Stand-ins reproduce the SAME imported API. This is a packaging/integrity
    # test, not a substitute for the real repository integration run.
    for fullname in ["revision", "revision.phase3", "revision.phase3.scripts",
                     "fast_lp_prox"]:
        pkg = types.ModuleType(fullname)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, fullname, pkg)
    phase3_name = "revision.phase3.scripts.run_lasso_end_to_end"
    fake_phase3 = types.ModuleType(phase3_name)
    def make_problem(**kwargs):
        p = toy_problem()
        _, supp = diag.regenerate_truth(seed=kwargs["seed"], m=2, n=3, support=1)
        p.xtrue_support = supp
        return p
    def parse_candidates(value):
        assert value == "newton:1"
        return [SimpleNamespace(family="newton", work=1, label="newton_K1")]
    fake_phase3.make_problem = make_problem
    fake_phase3.parse_candidates = parse_candidates
    fake_phase3.objective_tensor = toy_objective
    fake_phase3.git_commit = lambda: "mock-commit"
    monkeypatch.setitem(sys.modules, phase3_name, fake_phase3)

    fake_bench = types.ModuleType("fast_lp_prox.revision_bench")
    fake_bench.fit_degree_family = lambda *a, **k: None
    fake_bench.make_preloaded_prox = lambda **kwargs: (lambda z: z)
    monkeypatch.setitem(sys.modules, "fast_lp_prox.revision_bench", fake_bench)
    fake_const = types.ModuleType("fast_lp_prox.constants")
    fake_const.lp_constants = lambda *args, **kwargs: SimpleNamespace(tau=0.05)
    monkeypatch.setitem(sys.modules, "fast_lp_prox.constants", fake_const)

    parent = tmp_path / "trial"
    root = parent / "phase3_results"
    root.mkdir(parents=True)
    prob = make_problem(seed=0)
    truth, _ = diag.regenerate_truth(seed=0, m=2, n=3, support=1)
    trace = diag.replay_recovery_trajectory(prob, lambda z: z, xtrue=truth,
                                            steps=4, checkpoint_every=2, tau=0.05,
                                            objective=toy_objective)
    final = trace[-1]
    info = {"device": "cpu", "m": 2, "n": 3, "s": 1, "sigma": 0.01,
            "p": 0.5, "lambda_reg": 0.05, "seeds": [0],
            "candidates": ["newton_K1"], "max_iterations": 4,
            "verify_every": 2, "fit_sampled_errors": {}, "code_commit": "mock-hash"}
    (root / "metadata.json").write_text(json.dumps(info))
    (parent / "phase4_launch.json").write_text(json.dumps({"cmd": ["--dtype", "float32", "--candidates", "newton:1"]}))
    pd.DataFrame([{"seed": 0, "method": "newton_K1", "final_objective": final["objective"],
                  "final_nonzero_count": final["nonzero_count"],
                  "support_jaccard_vs_truth": final["support_jaccard_truth"],
                  "alpha": prob.alpha, "time_budget_iterations": 4,
                  "verified_stop_hit": 0, "final_verified_displacement": 0.11,
                  "replay_budget_median_ms": 0.2}]).to_csv(root / "run_summary.csv", index=False)
    (root / "trajectories.csv").write_text("iteration,objective\n0,1.0\n")
    (root / "reference_FB_trajectories.csv").write_text("iteration,objective\n0,1.0\n")
    args = argparse.Namespace(input_dir=root, output_dir=None, device="cpu", dtype=None,
                              stage_repeats=2, stage_warmup=1, power_iterations=100,
                              n_fit=128, n_validation=101, objective_rtol=1e-4,
                              objective_atol=1e-5)
    dest = diag.run(args)
    assert (dest / "diagnostics_summary.csv").is_file()
    summary = pd.read_csv(dest / "diagnostics_summary.csv")
    assert len(summary) == 1
    assert bool(summary["phase3_replay_check_pass"].iat[0])
    assert summary["recovery_l2_relative"].iat[0] > 0
    assert len(pd.read_csv(dest / "stage_timings_raw.csv")) == 2
    provenance = json.loads((dest / "diagnostics_metadata.json").read_text())
    assert provenance["original_phase3_commit"] == "mock-hash"
    with pytest.raises(FileExistsError):
        diag.run(args)
