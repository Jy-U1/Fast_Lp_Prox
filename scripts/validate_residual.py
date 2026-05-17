from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from fast_lp_prox import (
    active_mask,
    lp_constants,
    make_lp_prox,
    objective_gap_vs_reference,
    prox_stationary_residual,
    residual_error_bound,
)
from fast_lp_prox.benchmark_utils import get_device, make_test_vector


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()

    if name in {"float32", "fp32"}:
        return torch.float32

    if name in {"float64", "double", "fp64"}:
        return torch.float64

    raise ValueError(f"Unsupported dtype: {name!r}")


def parse_methods(methods: str) -> list[str]:
    return [m.strip() for m in methods.split(",") if m.strip()]


def run_residual_validation(args: argparse.Namespace) -> pd.DataFrame:
    device = get_device(args.device)
    dtype = parse_dtype(args.dtype)

    print("=" * 80)
    print("fast-lp-prox: residual / objective validation")
    print("=" * 80)
    print(f"device       : {device}")
    print(f"dtype        : {dtype}")
    print(f"p            : {args.p}")
    print(f"lambda       : {args.lam}")
    print(f"n            : {args.n}")
    print(f"sample_n     : {args.sample_n}")
    print(f"active_ratio : {args.active_ratio}")
    print(f"methods      : {args.methods}")
    print("=" * 80)

    const = lp_constants(args.p, args.lam)
    print("constants:")
    print(const)

    y = make_test_vector(
        n=args.n,
        tau=const.tau,
        active_ratio=args.active_ratio,
        ymax_factor=args.ymax_factor,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )

    sample_n = min(args.sample_n, args.n)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 12345)

    idx = torch.randint(
        low=0,
        high=args.n,
        size=(sample_n,),
        device=device,
        generator=generator,
    )

    y_sample = y[idx]
    active = active_mask(y_sample, p=args.p, lam=args.lam)

    print(f"active entries in sample: {int(active.sum().item())} / {sample_n}")

    print("\nBuilding float64 Newton reference...")

    prox_ref = make_lp_prox(
        p=args.p,
        lam=args.lam,
        method="newton12",
        dtype=torch.float64,
        newton_iters=args.ref_newton_iters,
    )

    y_ref = y_sample.to(torch.float64)
    x_ref = prox_ref(y_ref)

    rows = []

    for method in parse_methods(args.methods):
        print("\n" + "-" * 80)
        print(f"Method: {method}")
        print("-" * 80)

        prox = make_lp_prox(
            p=args.p,
            lam=args.lam,
            method=method,
            dtype=dtype,
            target_error=args.target_error,
            max_degree=args.max_degree,
            n_fit=args.n_fit,
            n_val=args.n_val,
        )

        x = prox(y_sample)
        x64 = x.to(torch.float64)

        residual = prox_stationary_residual(
            y_ref,
            x64,
            p=args.p,
            lam=args.lam,
        )

        bound = residual_error_bound(
            y_ref,
            x64,
            p=args.p,
            lam=args.lam,
        )

        obj_gap = objective_gap_vs_reference(
            y_ref,
            x64,
            x_ref,
            p=args.p,
            lam=args.lam,
        )

        abs_residual = torch.abs(residual[active])
        active_bound = bound[active]
        active_obj_gap = obj_gap[active]

        if active.sum().item() > 0:
            max_abs_residual = float(abs_residual.max().item())
            mean_abs_residual = float(abs_residual.mean().item())
            max_bound = float(active_bound.max().item())
            mean_bound = float(active_bound.mean().item())
            max_abs_obj_gap = float(torch.abs(active_obj_gap).max().item())
            mean_abs_obj_gap = float(torch.abs(active_obj_gap).mean().item())
        else:
            max_abs_residual = 0.0
            mean_abs_residual = 0.0
            max_bound = 0.0
            mean_bound = 0.0
            max_abs_obj_gap = 0.0
            mean_abs_obj_gap = 0.0

        row = {
            "p": args.p,
            "lambda": args.lam,
            "n": args.n,
            "sample_n": sample_n,
            "active_sample_n": int(active.sum().item()),
            "active_ratio": args.active_ratio,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "method": method,
            "max_abs_residual": max_abs_residual,
            "mean_abs_residual": mean_abs_residual,
            "max_residual_bound": max_bound,
            "mean_residual_bound": mean_bound,
            "max_abs_obj_gap_vs_ref": max_abs_obj_gap,
            "mean_abs_obj_gap_vs_ref": mean_abs_obj_gap,
        }

        rows.append(row)

        print(
            f"{method:10s}: "
            f"max_res={max_abs_residual:.3e}, "
            f"mean_res={mean_abs_residual:.3e}, "
            f"max_bound={max_bound:.3e}, "
            f"mean_bound={mean_bound:.3e}, "
            f"max_obj_gap={max_abs_obj_gap:.3e}"
        )

    df = pd.DataFrame(rows)

    print("\nResidual validation summary:")
    print(df.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved residual validation results to: {output_path}")

    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate stationary residuals and objective gaps for fast-lp-prox."
    )

    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--sample-n", type=int, default=10_000)

    parser.add_argument("--active-ratio", type=float, default=0.9)
    parser.add_argument("--ymax-factor", type=float, default=10.0)

    parser.add_argument(
        "--methods",
        type=str,
        default="poly,newton8,newton12,bisect40",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
    )

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--target-error", type=float, default=2e-7)
    parser.add_argument("--max-degree", type=int, default=128)
    parser.add_argument("--n-fit", type=int, default=8192)
    parser.add_argument("--n-val", type=int, default=80001)

    parser.add_argument("--ref-newton-iters", type=int, default=50)

    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional CSV output path.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_residual_validation(args)


if __name__ == "__main__":
    main()