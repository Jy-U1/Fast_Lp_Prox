#!/usr/bin/env python3
"""Non-destructive expansion of existing Phase-3 Lasso tests for Reviewer 2.

Default: dry run (print commands only). Use --execute to run and archive logs.
This invokes the existing Phase-3 driver; it does not patch its internals.

Smoke (Mac CPU):
  python revision/phase4/scripts/run_reviewer2_lasso_matrix.py --mode smoke
  python revision/phase4/scripts/run_reviewer2_lasso_matrix.py --mode smoke --execute
Formal (A100):
  python revision/phase4/scripts/run_reviewer2_lasso_matrix.py --mode formal --device cuda
  python revision/phase4/scripts/run_reviewer2_lasso_matrix.py --mode formal --device cuda --execute
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


@dataclass(frozen=True)
class Experiment:
    p: float
    p_label: str
    m: int
    n: int
    support: int
    candidates: str

    @property
    def tag(self) -> str:
        return f"p{self.p:.4f}".replace(".", "") + f"_m{self.m}_n{self.n}"


P_GRIDS = [
    (1 / 3, "1/3", "chebyshev:6,newton:3,bisection:20"),
    (3 / 4, "3/4", "chebyshev:12,newton:4,bisection:20"),
    (0.9, "0.9", "chebyshev:24,newton:4,bisection:20"),
]


def generate_grid(mode: str) -> list[Experiment]:
    if mode == "smoke":
        return [Experiment(1 / 3, "1/3", 80, 160, 8,
                           "chebyshev:6,newton:3,bisection:20")]
    if mode == "formal":
        return [Experiment(p, label, m, n, support, cands)
                for p, label, cands in P_GRIDS
                for m, n, support in ((1000, 2000, 100), (2000, 4000, 200))]
    raise ValueError(f"invalid mode: {mode}")


def make_cmd(exp: Experiment, args: argparse.Namespace, output: Path) -> list[str]:
    return [
        sys.executable, "-m", "revision.phase3.scripts.run_lasso_end_to_end",
        "--device", args.device or ("cpu" if args.mode == "smoke" else "cuda"),
        "--dtype", "float32", "--p", str(exp.p), "--p-label", exp.p_label,
        "--m", str(exp.m), "--n", str(exp.n), "--support", str(exp.support),
        "--lambda-reg", "0.05", "--seeds", "0" if args.mode == "smoke" else "0,1,2",
        "--max-iter", "20" if args.mode == "smoke" else "500",
        "--verify-every", "10", "--tol", "1e-4" if args.mode == "smoke" else "1e-6",
        # Phase-4 metadata/logs live in ``output``. The Phase-3 driver rejects
        # nonempty output directories, so place its results in a fresh child.
        # Do not use Phase-3's --overwrite flag: existing runs are immutable.
        "--candidates", exp.candidates,
        "--output-dir", str(output / "phase3_results"),
    ]


def git_info(root: Path) -> dict[str, str]:
    def read(*git_args: str) -> str:
        r = subprocess.run(["git", *git_args], cwd=root, text=True,
                           capture_output=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else f"UNKNOWN: {r.stderr.strip()}"
    return {"sha": read("rev-parse", "HEAD"),
            "branch": read("branch", "--show-current"),
            "status": read("status", "--porcelain")}


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--output-root", type=Path, default=Path("results/reviewer2_phase4_lasso"))
    parser.add_argument("--only-p", choices=["1/3", "3/4", "0.9"], default=None)
    parser.add_argument("--only-size", choices=["1000x2000", "2000x4000"], default=None)
    parser.add_argument("--execute", action="store_true", help="actually invoke Phase-3 experiments")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="permit formal runs on uncommitted code (not recommended)")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    existing = root / "revision/phase3/scripts/run_lasso_end_to_end.py"
    if not existing.is_file():
        print(f"ERROR: missing Phase-3 driver: {existing}", file=sys.stderr)
        return 2
    base = args.output_root if args.output_root.is_absolute() else root / args.output_root
    grid = generate_grid(args.mode)
    if args.only_p is not None:
        grid = [e for e in grid if e.p_label == args.only_p]
    if args.only_size is not None:
        grid = [e for e in grid if f"{e.m}x{e.n}" == args.only_size]
    if not grid:
        print("ERROR: filters select no experiments", file=sys.stderr)
        return 2
    metadata = git_info(root)
    print(f"Git branch: {metadata['branch']}, SHA: {metadata['sha']}")
    if metadata["status"]:
        print("WARNING: working tree is not clean; timings will not be pinned to a clean commit.")
    if args.mode == "formal" and args.execute:
        if args.device not in (None, "cuda"):
            print("ERROR: formal timing campaign requires --device cuda / A100", file=sys.stderr)
            return 2
        if metadata["status"] and not args.allow_dirty:
            print("ERROR: refuse formal campaign on dirty tree; commit or use --allow-dirty", file=sys.stderr)
            return 2
    print(f"Campaign: {args.mode}; {len(grid)} independent case(s); execute={args.execute}")
    for exp in grid:
        output = base / exp.tag
        cmd = make_cmd(exp, args, output)
        print("\n" + shlex.join(cmd))
        if not args.execute:
            continue
        if output.exists():
            print(f"ERROR: refusing to overwrite existing experiment directory {output}", file=sys.stderr)
            return 2
        output.mkdir(parents=True)
        start = datetime.now(timezone.utc).isoformat()
        record = {"experiment": asdict(exp), "mode": args.mode,
                  "cmd": cmd, "started_utc": start, "git": metadata,
                  "python_executable": sys.executable, "device": args.device or ("cpu" if args.mode == "smoke" else "cuda")}
        (output / "phase4_launch.json").write_text(json.dumps(record, indent=2) + "\n")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.run(cmd, cwd=root, capture_output=True, text=True, env=env)
        (output / "phase4_stdout.log").write_text(process.stdout)
        (output / "phase4_stderr.log").write_text(process.stderr)
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["return_code"] = process.returncode
        (output / "phase4_launch.json").write_text(json.dumps(record, indent=2) + "\n")
        if process.returncode != 0:
            print(f"ERROR: experiment failed with code {process.returncode}; see {output}/phase4_stderr.log", file=sys.stderr)
            # A short traceback preview makes remote debugging possible without
            # changing the complete archived subprocess stderr log.
            tail = process.stderr.strip().splitlines()[-12:]
            if tail:
                print("--- Last stderr lines ---", file=sys.stderr)
                print("\n".join(tail), file=sys.stderr)
            return process.returncode
        print(f"Completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
