from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from fast_lp_prox import make_lp_prox
from fast_lp_prox.benchmark_utils import get_device, synchronize_device


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()

    if name in {"float32", "fp32"}:
        return torch.float32

    if name in {"float64", "double", "fp64"}:
        return torch.float64

    raise ValueError(f"Unsupported dtype: {name!r}")


def parse_p(value: str) -> float:
    value = value.strip()

    if "/" in value:
        num, den = value.split("/")
        return float(num) / float(den)

    return float(value)


def parse_methods(methods: str) -> list[str]:
    return [m.strip() for m in methods.split(",") if m.strip()]


def make_synthetic_problem(
    m: int,
    n: int,
    sparsity: int,
    noise_std: float,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    A = torch.randn(m, n, generator=generator, dtype=dtype, device=device) / (m**0.5)

    x_true = torch.zeros(n, dtype=dtype, device=device)
    support = torch.randperm(n, generator=generator, device=device)[:sparsity]
    x_true[support] = torch.randn(sparsity, generator=generator, dtype=dtype, device=device)

    noise = noise_std * torch.randn(m, generator=generator, dtype=dtype, device=device)
    b = A @ x_true + noise

    return A, b, x_true


def spectral_norm_squared(A: torch.Tensor, iters: int = 80) -> float:
    n = A.shape[1]
    v = torch.randn(n, dtype=A.dtype, device=A.device)
    v = v / torch.linalg.norm(v)

    for _ in range(iters):
        v = A.T @ (A @ v)
        v = v / torch.linalg.norm(v)

    Av = A @ v
    return float(torch.dot(Av, Av).item())


def objective_parts(
    A: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    p: float,
    lam: float,
) -> tuple[float, float, float]:
    residual = A @ x - b
    data_term = 0.5 * torch.dot(residual, residual)
    reg_term = lam * torch.sum(torch.pow(torch.abs(x), p))
    total = data_term + reg_term

    return (
        float(total.item()),
        float(data_term.item()),
        float(reg_term.item()),
    )


def run_fb_once(
    A: torch.Tensor,
    b: torch.Tensor,
    p: float,
    lam: float,
    alpha: float,
    method: str,
    num_iters: int,
    target_error: float,
    max_degree: int,
    n_fit: int,
    n_val: int,
):
    prox = make_lp_prox(
        p=p,
        lam=alpha * lam,
        method=method,
        dtype=A.dtype,
        target_error=target_error,
        max_degree=max_degree,
        n_fit=n_fit,
        n_val=n_val,
    )

    x = torch.zeros(A.shape[1], dtype=A.dtype, device=A.device)

    for _ in range(num_iters):
        grad = A.T @ (A @ x - b)
        y = x - alpha * grad
        x = prox(y)

    return x


def time_fb_method(
    A: torch.Tensor,
    b: torch.Tensor,
    x_true: torch.Tensor,
    p: float,
    lam: float,
    alpha: float,
    method: str,
    num_iters: int,
    warmup: int,
    repeats: int,
    target_error: float,
    max_degree: int,
    n_fit: int,
    n_val: int,
):
    device = A.device

    # Build once and warm up.
    for _ in range(warmup):
        _ = run_fb_once(
            A=A,
            b=b,
            p=p,
            lam=lam,
            alpha=alpha,
            method=method,
            num_iters=max(1, min(3, num_iters)),
            target_error=target_error,
            max_degree=max_degree,
            n_fit=n_fit,
            n_val=n_val,
        )

    synchronize_device(device)

    times_ms = []
    final_x = None

    for _ in range(repeats):
        synchronize_device(device)
        t0 = time.perf_counter()

        final_x = run_fb_once(
            A=A,
            b=b,
            p=p,
            lam=lam,
            alpha=alpha,
            method=method,
            num_iters=num_iters,
            target_error=target_error,
            max_degree=max_degree,
            n_fit=n_fit,
            n_val=n_val,
        )

        synchronize_device(device)
        t1 = time.perf_counter()

        times_ms.append((t1 - t0) * 1000.0)

    total_obj, data_term, reg_term = objective_parts(A, b, final_x, p=p, lam=lam)

    rel_err = torch.linalg.norm(final_x - x_true) / torch.maximum(
        torch.tensor(1.0, dtype=final_x.dtype, device=final_x.device),
        torch.linalg.norm(x_true),
    )

    return {
        "median_ms": float(pd.Series(times_ms).median()),
        "mean_ms": float(pd.Series(times_ms).mean()),
        "std_ms": float(pd.Series(times_ms).std(ddof=0)),
        "min_ms": float(pd.Series(times_ms).min()),
        "max_ms": float(pd.Series(times_ms).max()),
        "ms_per_iter": float(pd.Series(times_ms).median() / num_iters),
        "final_objective": total_obj,
        "final_data_term": data_term,
        "final_reg_term": reg_term,
        "relative_error_to_true": float(rel_err.item()),
        "nonzeros_abs_gt_1e-6": int((torch.abs(final_x) > 1e-6).sum().item()),
        "checksum": float(final_x[: min(1024, final_x.numel())].sum().item()),
        "final_x": final_x,
    }


def run_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    device = get_device(args.device)
    dtype = parse_dtype(args.dtype)
    p = parse_p(args.p)

    print("=" * 80)
    print("fast-lp-prox: Lp-Lasso Forward-Backward benchmark")
    print("=" * 80)
    print(f"device     : {device}")
    print(f"dtype      : {dtype}")
    print(f"p          : {args.p} = {p}")
    print(f"lambda     : {args.lam}")
    print(f"m, n       : {args.m}, {args.n}")
    print(f"sparsity   : {args.sparsity}")
    print(f"noise_std  : {args.noise_std}")
    print(f"iterations : {args.num_iters}")
    print(f"methods    : {args.methods}")
    print("=" * 80)

    A, b, x_true = make_synthetic_problem(
        m=args.m,
        n=args.n,
        sparsity=args.sparsity,
        noise_std=args.noise_std,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )

    L = spectral_norm_squared(A, iters=args.power_iters)
    alpha = args.step_scale / L

    print(f"L estimate : {L:.6e}")
    print(f"alpha      : {alpha:.6e}")

    rows = []
    outputs = {}

    for method in parse_methods(args.methods):
        print("\n" + "-" * 80)
        print(f"Method: {method}")
        print("-" * 80)

        stats = time_fb_method(
            A=A,
            b=b,
            x_true=x_true,
            p=p,
            lam=args.lam,
            alpha=alpha,
            method=method,
            num_iters=args.num_iters,
            warmup=args.warmup,
            repeats=args.repeats,
            target_error=args.target_error,
            max_degree=args.max_degree,
            n_fit=args.n_fit,
            n_val=args.n_val,
        )

        final_x = stats.pop("final_x")
        outputs[method] = final_x

        row = {
            "p": p,
            "lambda": args.lam,
            "m": args.m,
            "n": args.n,
            "sparsity": args.sparsity,
            "noise_std": args.noise_std,
            "num_iters": args.num_iters,
            "L_estimate": L,
            "alpha": alpha,
            "step_scale": args.step_scale,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "method": method,
            **stats,
        }

        rows.append(row)

        print(
            f"{method:10s}: "
            f"median={row['median_ms']:.3f} ms, "
            f"ms/iter={row['ms_per_iter']:.4f}, "
            f"obj={row['final_objective']:.6e}, "
            f"nnz={row['nonzeros_abs_gt_1e-6']}"
        )

    df = pd.DataFrame(rows)

    if "poly" in outputs:
        x_poly = outputs["poly"]

        for row in rows:
            method = row["method"]
            diff = torch.linalg.norm(outputs[method] - x_poly) / torch.maximum(
                torch.tensor(1.0, dtype=x_poly.dtype, device=x_poly.device),
                torch.linalg.norm(x_poly),
            )
            row["rel_diff_vs_poly"] = float(diff.item())

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
                    "runtime_ratio_baseline_vs_poly": float(row["median_ms"]) / poly_ms,
                }
            )

        if speedup_rows:
            speedup_df = pd.DataFrame(speedup_rows)
            print("\nRuntime ratio relative to poly:")
            print(speedup_df.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved benchmark results to: {output_path}")

    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Forward-Backward for synthetic Lp-Lasso using different Lp prox methods."
    )

    parser.add_argument("--p", type=str, default="1/3")
    parser.add_argument("--lam", type=float, default=0.05)

    parser.add_argument("--m", type=int, default=400)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--sparsity", type=int, default=40)
    parser.add_argument("--noise-std", type=float, default=0.01)

    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--step-scale", type=float, default=0.95)
    parser.add_argument("--power-iters", type=int, default=80)

    parser.add_argument(
        "--methods",
        type=str,
        default="poly,newton8,newton12,bisect40",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
    )

    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--target-error", type=float, default=2e-7)
    parser.add_argument("--max-degree", type=int, default=128)
    parser.add_argument("--n-fit", type=int, default=8192)
    parser.add_argument("--n-val", type=int, default=80001)

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