#!/usr/bin/env python3
"""Non-destructive Phase-4 Lasso diagnostics for an existing Phase-3 run.

Reconstruct the planted signal from the EXACT Phase-3 NumPy RNG sequence,
replay candidate iterations to check agreement with frozen Phase-3 results,
and write separate (1) recovery/active-coordinate traces and (2) instrumented
forward-gradient versus proximal stage timings.

No Phase-3 source files, original outputs, or existing timing measurements are
modified. Stage timing is *instrumented diagnostics*, not the original
whole-loop benchmark; no convergence claim is made when the stop test fails.

Use from the repository root:
  python -m revision.phase4.scripts.diagnose_lasso_outputs \
      --input-dir results/reviewer2_phase4_lasso_retry1/p03333_m80_n160/phase3_results \
      --device cpu --stage-repeats 2
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch


def regenerate_truth(*, seed: int, m: int, n: int, support: int) -> tuple[np.ndarray, np.ndarray]:
    """Replay Phase-3 make_problem RNG exactly, without changing Phase-3 code.

    Its draw order is A (standard_normal m*n), support (rng.choice),
    then coefficients (standard_normal support). The noise draw follows these.
    A is discarded here; make_problem separately creates the rounded A and b.
    """
    if not (m >= 1 and n >= 2 and 1 <= support <= n):
        raise ValueError("invalid dimensions/support")
    rng = np.random.default_rng(int(seed))
    _ = rng.standard_normal((m, n)).astype(np.float32)
    indices = np.sort(rng.choice(n, size=support, replace=False))
    xtrue = np.zeros(n, dtype=np.float64)
    xtrue[indices] = rng.standard_normal(support)
    return xtrue, indices


def reconstruction_stats(x: np.ndarray, xtrue: np.ndarray) -> dict[str, float | int]:
    """Truth metrics with exact nonzero interpretation matching Phase-3."""
    x = np.asarray(x, dtype=np.float64)
    xtrue = np.asarray(xtrue, dtype=np.float64)
    if x.ndim != 1 or x.shape != xtrue.shape or not np.isfinite(x).all():
        raise ValueError("x and xtrue must be matching finite 1D arrays")
    err = float(np.linalg.norm(x - xtrue))
    norm = float(np.linalg.norm(xtrue))
    predicted = x != 0.0
    actual = xtrue != 0.0
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    precision = tp / (tp + fp) if tp + fp else (1.0 if not actual.any() else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = tp + fp + fn
    return {
        "recovery_l2_absolute": err,
        "recovery_l2_relative": err / norm if norm else (0.0 if err == 0 else float("inf")),
        "nonzero_count": int(np.count_nonzero(predicted)),
        "active_coordinate_fraction": float(np.mean(predicted)),
        "truth_nonzero_count": int(np.count_nonzero(actual)),
        "support_true_positive": tp,
        "support_false_positive": fp,
        "support_false_negative": fn,
        "support_precision": float(precision),
        "support_recall": float(recall),
        "support_f1": float(f1),
        "support_jaccard_truth": float(tp / union if union else 1.0),
    }


def proximal_input_active_fraction(x: torch.Tensor, problem, tau: float) -> float:
    """Fraction of pre-prox input with |z_i|>tau, distinct from support(x)."""
    with torch.no_grad():
        z = x - problem.alpha * (problem.A.T @ (problem.A @ x - problem.b))
        return float((torch.abs(z) > tau).to(torch.float64).mean().item())


def replay_recovery_trajectory(problem, prox: Callable, *, xtrue: np.ndarray,
                               steps: int, checkpoint_every: int, tau: float,
                               objective: Callable[[torch.Tensor, object], torch.Tensor]
                               ) -> list[dict]:
    if steps < 0 or checkpoint_every < 1:
        raise ValueError("invalid steps/checkpoint interval")
    x = torch.zeros(problem.n, dtype=problem.A.dtype, device=problem.A.device)
    rows: list[dict] = []
    def capture(k: int) -> None:
        x_np = x.detach().to(dtype=torch.float64).cpu().numpy()
        stat = reconstruction_stats(x_np, xtrue)
        rows.append({"iteration": k,
                     "objective": float(objective(x, problem).item()),
                     "prox_input_active_fraction": proximal_input_active_fraction(x, problem, tau),
                     **stat})
    with torch.no_grad():
        capture(0)
        for k in range(1, steps + 1):
            z = x - problem.alpha * (problem.A.T @ (problem.A @ x - problem.b))
            x = prox(z)
            if k % checkpoint_every == 0 or k == steps:
                capture(k)
    return rows


def instrumented_stage_replays(problem, prox: Callable, *, steps: int,
                               repeats: int, warmup: int) -> list[dict]:
    """Separate forward gradient/update and proximal stage measurements.

    For CUDA, events bracket stages within the SAME FB trajectory, with only
    one device synchronization at the end of each replay. Events perturb
    execution; results are diagnostic and must NOT replace Phase-3 replay.
    CPU uses perf_counter around stages and has Python-timer overhead.
    """
    if steps < 0 or repeats < 1 or warmup < 0:
        raise ValueError("invalid stage replay parameters")
    device = problem.A.device
    cuda = device.type == "cuda"
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA timing are supported")
    def one_step(x: torch.Tensor) -> torch.Tensor:
        z = x - problem.alpha * (problem.A.T @ (problem.A @ x - problem.b))
        return prox(z)
    rows = []
    with torch.no_grad():
        for _ in range(warmup):
            x = torch.zeros(problem.n, device=device, dtype=problem.A.dtype)
            for _ in range(steps):
                x = one_step(x)
        if cuda:
            torch.cuda.synchronize(device)
        for rep in range(repeats):
            x = torch.zeros(problem.n, device=device, dtype=problem.A.dtype)
            gradient_total = 0.0
            prox_total = 0.0
            events = []
            for _ in range(steps):
                if cuda:
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e2 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                else:
                    t0 = time.perf_counter()
                z = x - problem.alpha * (problem.A.T @ (problem.A @ x - problem.b))
                if cuda:
                    e1.record()
                else:
                    t1 = time.perf_counter()
                x = prox(z)
                if cuda:
                    e2.record()
                    events.append((e0, e1, e2))
                else:
                    t2 = time.perf_counter()
                    gradient_total += 1000.0 * (t1 - t0)
                    prox_total += 1000.0 * (t2 - t1)
            if cuda:
                torch.cuda.synchronize(device)
                gradient_total = sum(t0.elapsed_time(t1) for t0, t1, _ in events)
                prox_total = sum(t1.elapsed_time(t2) for _, t1, t2 in events)
            total = gradient_total + prox_total
            rows.append({"repeat": rep, "steps": steps,
                         "gradient_forward_ms": float(gradient_total),
                         "prox_ms": float(prox_total),
                         "instrumented_total_ms": float(total),
                         "gradient_forward_fraction": float(gradient_total / total) if total else float("nan"),
                         "prox_fraction": float(prox_total / total) if total else float("nan"),
                         "timing_semantics": "cuda_events_no_per_step_sync" if cuda else "cpu_per_step_wall_diagnostic"})
    return rows


def verify_terminal(recomputed: dict, recorded: pd.Series, *, rtol: float,
                    atol: float) -> dict[str, float | bool]:
    """Fail closed if replay differs materially from original Phase-3 output."""
    want_obj = float(recorded["final_objective"])
    got_obj = float(recomputed["objective"])
    matches_obj = bool(np.isclose(got_obj, want_obj, rtol=rtol, atol=atol))
    want_nnz = int(recorded["final_nonzero_count"])
    matches_nnz = int(recomputed["nonzero_count"]) == want_nnz
    want_j = float(recorded["support_jaccard_vs_truth"])
    matches_jaccard = bool(np.isclose(float(recomputed["support_jaccard_truth"]),
                                       want_j, rtol=0.0, atol=1e-12))
    return {"phase3_objective_abs_diff": abs(got_obj - want_obj),
            "phase3_objective_check_pass": matches_obj,
            "phase3_support_count_check_pass": matches_nnz,
            "phase3_truth_jaccard_check_pass": matches_jaccard,
            "phase3_replay_check_pass": matches_obj and matches_nnz and matches_jaccard}


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def _arg_from_cmd(cmd: list[str], key: str) -> str | None:
    try:
        i = cmd.index(key)
    except ValueError:
        return None
    return cmd[i + 1] if i + 1 < len(cmd) else None


def _load_config(source: Path, args: argparse.Namespace) -> tuple[dict, pd.DataFrame, str, str]:
    info = json.loads((source / "metadata.json").read_text())
    summary = pd.read_csv(source / "run_summary.csv", float_precision="round_trip")
    launch_path = source.parent / "phase4_launch.json"
    launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
    cmd = launch.get("cmd", [])
    recorded_dtype = _arg_from_cmd(cmd, "--dtype")
    if recorded_dtype and args.dtype and recorded_dtype != args.dtype:
        raise ValueError(f"Requested dtype={args.dtype} differs from recorded {recorded_dtype}")
    dtype_name = args.dtype or recorded_dtype
    if dtype_name is None:
        raise ValueError("Phase-3 metadata does not record dtype; pass --dtype float32/float64")
    recorded_device = str(info["device"])
    if args.device and args.device != recorded_device:
        raise ValueError(f"Requested device={args.device} differs from recorded {recorded_device}; "
                         "diagnose on the original device")
    device_name = args.device or recorded_device
    if device_name not in ("cpu", "cuda"):
        raise ValueError("Diagnostic staging is supported only on CPU or CUDA")
    for needed in ("seed", "method", "final_objective", "final_nonzero_count",
                   "support_jaccard_vs_truth", "time_budget_iterations"):
        if needed not in summary.columns:
            raise ValueError(f"Missing Phase-3 run_summary column: {needed}")
    return info, summary, dtype_name, device_name


def run(args: argparse.Namespace) -> Path:
    # Import Phase-3 only when running the actual local project; the pure tests
    # of this module work without monkey-patching its global imports.
    from revision.phase3.scripts.run_lasso_end_to_end import (
        make_problem, parse_candidates, objective_tensor, git_commit
    )
    from fast_lp_prox.revision_bench import fit_degree_family, make_preloaded_prox
    from fast_lp_prox.constants import lp_constants
    source = args.input_dir.resolve()
    required = ("metadata.json", "run_summary.csv", "trajectories.csv",
                "reference_FB_trajectories.csv")
    if not source.is_dir() or any(not (source / x).is_file() for x in required):
        raise FileNotFoundError(f"Not a complete Phase-3 output directory: {source}")
    output = args.output_dir.resolve() if args.output_dir else source.parent / "phase4_diagnostics"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase-4 output: {output}")
    info, old_summary, dtype_name, device_name = _load_config(source, args)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Original run used CUDA; run this diagnostic on CUDA, not Mac CPU")
    device = torch.device(device_name)
    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_name]
    p = float(info["p"])
    lam = float(info["lambda_reg"])
    m, n, support = int(info["m"]), int(info["n"]), int(info["s"])
    iters = int(info["max_iterations"])
    verify_every = int(info["verify_every"])
    cand_text = _arg_from_cmd(json.loads((source.parent / "phase4_launch.json").read_text()).get("cmd", []), "--candidates") if (source.parent / "phase4_launch.json").exists() else None
    if cand_text is None:
        # Archived Phase-3 metadata records candidate labels but not family:work.
        cand_text = ",".join(f"{s.split('_')[0]}:{s.split('_')[1][1:]}" for s in info["candidates"])
    candidates = parse_candidates(cand_text)
    recorded_labels = list(info["candidates"])
    if {c.label for c in candidates} != set(recorded_labels):
        raise ValueError("Recorded candidates disagree with launch command")
    needed_pairs = {(int(s), c.label) for s in info["seeds"] for c in candidates}
    present_pairs = {(int(r.seed), str(r.method)) for r in old_summary.itertuples()}
    if needed_pairs != present_pairs or len(old_summary) != len(needed_pairs):
        raise ValueError("Phase-3 summary is incomplete or contains duplicate seed/method pairs")
    degrees = sorted({c.work for c in candidates if c.family == "chebyshev"})
    fitted = fit_degree_family(p, degrees, n_fit=args.n_fit,
                               n_validation=args.n_validation) if degrees else None
    if fitted is not None:
        for k, v in info.get("fit_sampled_errors", {}).items():
            actual = fitted.sampled_errors.get(int(k))
            if actual is None or not np.isclose(actual, float(v), rtol=1e-6, atol=1e-10):
                raise ValueError("Chebyshev fit does not match archived sampled errors; "
                                 "check --n-fit and --n-validation")
    recovery_rows: list[dict] = []
    timing_rows: list[dict] = []
    summaries: list[dict] = []
    for seed in info["seeds"]:
        seed = int(seed)
        problem = make_problem(m=m, n=n, support=support,
                               sigma=float(info["sigma"]), lam=lam, p=p,
                               seed=seed, device=device, dtype=dtype,
                               power_iterations=args.power_iterations)
        xtrue, truth_support = regenerate_truth(seed=seed, m=m, n=n, support=support)
        if not np.array_equal(truth_support, problem.xtrue_support):
            raise AssertionError("Recreated planted support does not match Phase-3 RNG")
        tau_prox = float(lp_constants(p, lam=problem.alpha * problem.lambda_reg).tau)
        for candidate in candidates:
            subset = old_summary[(old_summary["seed"] == seed) &
                                 (old_summary["method"] == candidate.label)]
            record = subset.iloc[0]
            if not np.isclose(float(record["alpha"]), problem.alpha, rtol=1e-5, atol=1e-8):
                raise ValueError("Reconstructed alpha differs from Phase-3; check power iterations/device")
            coeff = (fitted.coefficients[:candidate.work + 1] if candidate.family == "chebyshev" else None)
            prox = make_preloaded_prox(p=p, lam=problem.alpha * lam, family=candidate.family,
                                      work=candidate.work, coefficients=coeff,
                                      dtype=dtype, device=device)
            traj = replay_recovery_trajectory(
                problem, prox, xtrue=xtrue, steps=iters,
                checkpoint_every=verify_every, tau=tau_prox, objective=objective_tensor)
            for row in traj:
                recovery_rows.append({"seed": seed, "p": p, "method": candidate.label,
                                      "m": m, "n": n, **row})
            terminal = traj[-1]
            comparison = verify_terminal(terminal, record, rtol=args.objective_rtol,
                                         atol=args.objective_atol)
            if not comparison["phase3_replay_check_pass"]:
                raise RuntimeError(f"Phase-4 terminal replay disagrees with Phase-3: "
                                   f"seed={seed}, method={candidate.label}, {comparison}")
            budget = int(record["time_budget_iterations"])
            if not 0 <= budget <= iters:
                raise ValueError("Archived replay budget is outside the permitted range")
            stage_rows = instrumented_stage_replays(problem, prox, steps=budget,
                                                    repeats=args.stage_repeats,
                                                    warmup=args.stage_warmup)
            for row in stage_rows:
                timing_rows.append({"seed": seed, "p": p, "method": candidate.label,
                                    "m": m, "n": n, **row})
            grad_med = float(np.median([r["gradient_forward_ms"] for r in stage_rows]))
            prox_med = float(np.median([r["prox_ms"] for r in stage_rows]))
            summaries.append({"seed": seed, "p": p, "method": candidate.label,
                              "m": m, "n": n, "budget_iterations": budget,
                              "phase3_stop_hit": int(record["verified_stop_hit"]),
                              "phase3_final_displacement": float(record["final_verified_displacement"]),
                              **terminal, **comparison,
                              "instrumented_gradient_forward_median_ms": grad_med,
                              "instrumented_prox_median_ms": prox_med,
                              "instrumented_prox_time_fraction": prox_med / (grad_med + prox_med) if grad_med + prox_med else float("nan"),
                              "phase3_replay_budget_median_ms": float(record["replay_budget_median_ms"])})
        print(f"Recomputed diagnostics for seed={seed}, p={p}, m={m}, n={n}", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(recovery_rows).to_csv(output / "recovery_trajectories.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(output / "stage_timings_raw.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "diagnostics_summary.csv", index=False)
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "original_phase3_directory": str(source),
        "original_phase3_commit": info.get("code_commit"),
        "current_git_commit": git_commit(),
        "original_hashes_sha256": {name: hash_file(source / name) for name in required},
        "p": p, "m": m, "n": n, "support": support,
        "seeds": info["seeds"], "candidates": [c.label for c in candidates],
        "device": device_name, "dtype": dtype_name,
        "fit_n": args.n_fit, "fit_validation_n": args.n_validation,
        "stage_repeats": args.stage_repeats, "stage_warmup": args.stage_warmup,
        "diagnostic_checkpoint_every": verify_every,
        "recovery_semantics": "relative l2 recovery = ||x_k - xtrue||_2 / ||xtrue||_2; planted xtrue reconstructed via exact Phase-3 NumPy RNG draw order",
        "active_fraction_semantics": "active_coordinate_fraction = #(x!=0)/n; prox_input_active_fraction = #(|x-alpha grad f(x)|>tau_{p,alpha*lambda})/n",
        "timing_semantics": "separate instrumented gradient+forward and prox stage time, not the Phase-3 end-to-end replay or verified-stop wall time",
        "verification_semantics": "final objective, final nonzero count, and true-support Jaccard must match archived Phase-3 results before any new results are written",
    }
    (output / "diagnostics_metadata.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved new diagnostics only to {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="existing directory containing Phase-3 CSVs and metadata.json")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="new output directory; default: sibling phase4_diagnostics")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--dtype", choices=["float32", "float64"], default=None)
    ap.add_argument("--stage-repeats", type=int, default=3)
    ap.add_argument("--stage-warmup", type=int, default=1)
    ap.add_argument("--power-iterations", type=int, default=100,
                    help="must equal Phase-3 --power-iterations; default 100")
    ap.add_argument("--n-fit", type=int, default=8192)
    ap.add_argument("--n-validation", type=int, default=80001)
    ap.add_argument("--objective-rtol", type=float, default=1e-4)
    ap.add_argument("--objective-atol", type=float, default=1e-5)
    args = ap.parse_args(argv)
    if args.stage_repeats < 1 or args.stage_warmup < 0:
        ap.error("invalid number of timing repetitions/warmups")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
