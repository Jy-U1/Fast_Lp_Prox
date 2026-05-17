from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from fast_lp_prox import lp_constants, make_lp_prox
from fast_lp_prox.benchmark_utils import (
    get_device,
    make_test_vector,
    tensor_summary,
    time_function,
)


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()

    if name in {"float32", "fp32"}:
        return torch.float32

    if name in {"float64", "double", "fp64"}:
        return torch.float64

    raise ValueError(f"Unsupported dtype: {name!r}")


def parse_methods(methods: str) -> list[str]:
    return [m.strip() for m in methods.split(",") if m.strip()]


def run_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    device = get_device(args.device)
    dtype = parse_dtype(args.dtype)

    print("=" * 80)
    print("fast-lp-prox: prox benchmark")
    print("=" * 80)
    print(f"device       : {device}")
    print(f"dtype        : {dtype}")
    print(f"p            : {args.p}")
    print(f"lambda       : {args.lam}")
    print(f"n            : {args.n}")
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

        stats, out = time_function(
            prox,
            y,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        out_stats = tensor_summary(out)
        million_elems_per_sec = args.n / (stats["median_ms"] * 1e-3) / 1e6

        row = {
            "p": args.p,
            "lambda": args.lam,
            "n": args.n,
            "active_ratio": args.active_ratio,
            "ymax_factor": args.ymax_factor,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "method": method,
            "median_ms": stats["median_ms"],
            "mean_ms": stats["mean_ms"],
            "std_ms": stats["std_ms"],
            "min_ms": stats["min_ms"],
            "max_ms": stats["max_ms"],
            "million_elems_per_sec": million_elems_per_sec,
            "checksum": stats["checksum"],
            "all_finite": out_stats["all_finite"],
            "out_min": out_stats["min"],
            "out_max": out_stats["max"],
            "out_mean": out_stats["mean"],
            "out_std": out_stats["std"],
        }

        rows.append(row)

        print(
            f"{method:10s}: "
            f"median={stats['median_ms']:.4f} ms, "
            f"throughput={million_elems_per_sec:.2f} M elems/s, "
            f"finite={out_stats['all_finite']}"
        )

    df = pd.DataFrame(rows)

    print("\nBenchmark summary:")
    print(df.to_string(index=False))

    if "poly" in df["method"].values:
        poly_ms = float(df.loc[df["method"] == "poly", "median_ms"].iloc[0])

        speedup_rows = []
        for _, row in df.iterrows():
            if row["method"] == "poly":
                continue
            speedup_rows.append(
                {
                    "baseline": row["method"],
                    "speedup_baseline_vs_poly": float(row["median_ms"]) / poly_ms,
                }
            )

        if speedup_rows:
            speedup_df = pd.DataFrame(speedup_rows)
            print("\nSpeedup relative to poly:")
            print(speedup_df.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved benchmark results to: {output_path}")

    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark fast-lp-prox proximal evaluators."
    )

    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--active-ratio", type=float, default=0.9)
    parser.add_argument("--ymax-factor", type=float, default=10.0)

    parser.add_argument(
        "--methods",
        type=str,
        default="poly,newton8,newton12,bisect40",
        help="Comma-separated methods.",
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

    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    # Chebyshev coefficient generation settings.
    # These defaults are development-friendly.
    # For final GPU benchmarks, use tighter target / larger n_fit / n_val.
    parser.add_argument("--target-error", type=float, default=1e-4)
    parser.add_argument("--max-degree", type=int, default=32)
    parser.add_argument("--n-fit", type=int, default=1024)
    parser.add_argument("--n-val", type=int, default=2001)

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
    run_benchmark(args)


if __name__ == "__main__":
    main()