"""Small tests for non-destructive Phase-4 utilities. No CUDA required."""
from __future__ import annotations

import csv
import importlib.util
import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _import(filename, name):
    path = ROOT / "revision/phase4/scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


summ = _import("summarize_scalar_backends.py", "phase4_scalar")
lasso = _import("run_reviewer2_lasso_matrix.py", "phase4_lasso")


def _fixture():
    fields = ["p", "family", "configuration", "max_abs_error", "max_norm_error",
              "max_active_stationarity_residual", "mean_prox_objective_gap",
              "support_mismatch_count", "median_ms", "n_accuracy", "git_commit",
              "all_finite", "meets_accuracy_target", "accuracy_target",
              "device", "dtype", "seed", "active_ratio_realized", "reference_method",
              "accuracy_same_inputs_as_timing"]
    rows = []
    for method in summ.FAMILIES:
        for idx, (error, t) in enumerate([(2e-6, 0.1), (8e-7, 0.3), (1e-7, 0.5)]):
            rows.append(dict(p=str(1 / 3), family=method,
                             configuration=f"{method}_{idx}", max_abs_error=str(error),
                             max_norm_error=str(error), max_active_stationarity_residual=str(error),
                             mean_prox_objective_gap="2e-12", support_mismatch_count="0",
                             median_ms=str(t), n_accuracy="100", git_commit="abc",
                             all_finite="True", meets_accuracy_target=str(error <= 1e-6),
                             accuracy_target="1e-6", device="cuda", dtype="float32",
                             seed="0", active_ratio_realized="0.9",
                             reference_method="bisection_75", accuracy_same_inputs_as_timing="True"))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def test_fastest_accuracy_matched_selected():
    chosen = summ.select_records([("fake.csv", _fixture())])
    assert len(chosen) == 3
    assert {r["family"] for r in chosen} == set(summ.FAMILIES)
    assert all(r["configuration"].endswith("_1") for r in chosen)
    assert all(float(r["max_norm_error"]) <= 1e-6 for r in chosen)


def test_selected_output_is_not_overwritten(tmp_path):
    data = summ.select_records([("fake.csv", _fixture())])
    summ.write_outputs(tmp_path / "new", data)
    assert (tmp_path / "new/selected_scalar_backend_metrics.csv").is_file()
    assert (tmp_path / "new/selected_scalar_backend_metrics.tex").is_file()
    assert (tmp_path / "new/data_provenance.json").is_file()
    with pytest.raises(FileExistsError):
        summ.write_outputs(tmp_path / "new", data)


def test_inconsistent_inputs_are_rejected():
    payload = _fixture().decode().replace(",0.9,bisection_75,True", ",0.8,bisection_75,True", 1)
    with pytest.raises(ValueError):
        summ.select_records([("fake.csv", payload.encode())])


def test_float_near_one_grid_is_separate_and_multiple_sizes():
    exps = lasso.generate_grid("formal")
    assert len(exps) == 6
    assert {e.p_label for e in exps} == {"1/3", "3/4", "0.9"}
    assert {(e.m, e.n) for e in exps} == {(1000, 2000), (2000, 4000)}
    assert all(e.support / e.n == 0.05 for e in exps)
    assert all("bisection:20" in e.candidates for e in exps)


def test_smoke_experiment_runs_cpu_without_cuda_default(tmp_path):
    import argparse
    e = lasso.generate_grid("smoke")[0]
    flags = argparse.Namespace(mode="smoke", device=None)
    cmd = lasso.make_cmd(e, flags, tmp_path / "trial")
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert cmd[cmd.index("--seeds") + 1] == "0"
    assert cmd[cmd.index("--max-iter") + 1] == "20"
    assert Path(cmd[cmd.index("--output-dir") + 1]) == tmp_path / "trial" / "phase3_results"


def test_tex_records_sampled_not_certified():
    data = summ.select_records([("fake.csv", _fixture())])
    text = summ.render_latex(data)
    assert "sampled" in text
    assert "certified" in text
    assert "chebyshev" in text


def test_wrapper_and_phase3_have_separate_output_directories(tmp_path):
    """Regression: Phase-3 rejects nonempty output, Phase-4 writes launch metadata."""
    # Tiny Phase-3 stand-in that enforces the same no-overwrite policy as the
    # actual driver, without requiring torch/GPU or fitting Chebyshev tables.
    pkg = tmp_path / "revision" / "phase3" / "scripts"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "revision", tmp_path / "revision" / "phase3", pkg):
        (d / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "run_lasso_end_to_end.py").write_text("import argparse\nfrom pathlib import Path\np = argparse.ArgumentParser(add_help=False)\np.add_argument('--output-dir', required=True)\na, _ = p.parse_known_args()\nout = Path(a.output_dir)\nif out.exists() and any(out.iterdir()):\n    raise FileExistsError(f'{out} exists and is nonempty')\nout.mkdir(parents=True, exist_ok=True)\n(out / 'run_summary.csv').write_text('method,verified\\nmock,1\\n')\n", encoding="utf-8")
    args = ["--mode", "smoke", "--repo-root", str(tmp_path),
            "--output-root", "results/reviewer2_phase4_lasso", "--execute"]
    assert lasso.cli(args) == 0
    parent = tmp_path / "results" / "reviewer2_phase4_lasso" / "p03333_m80_n160"
    assert (parent / "phase4_launch.json").is_file()
    assert (parent / "phase4_stdout.log").is_file()
    assert (parent / "phase3_results" / "run_summary.csv").is_file()
    # Existing attempts are never overwritten by a repeated launch.
    assert lasso.cli(args) == 2
