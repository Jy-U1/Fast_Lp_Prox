from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from fast_lp_prox import make_lp_prox


def make_synthetic_problem(
    m: int = 400,
    n: int = 800,
    sparsity: int = 40,
    noise_std: float = 0.01,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
):
    """
    Synthetic sparse regression problem:

        b = A x_true + noise

    Objective:

        0.5 * ||A x - b||_2^2 + lambda * sum_i |x_i|^p
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    A = torch.randn(m, n, generator=generator, dtype=dtype) / (m**0.5)

    x_true = torch.zeros(n, dtype=dtype)
    support = torch.randperm(n, generator=generator)[:sparsity]
    x_true[support] = torch.randn(sparsity, generator=generator, dtype=dtype)

    noise = noise_std * torch.randn(m, generator=generator, dtype=dtype)
    b = A @ x_true + noise

    return A, b, x_true


def spectral_norm_squared(A: torch.Tensor, iters: int = 80) -> float:
    """
    Estimate ||A||_2^2 by power iteration.
    """

    n = A.shape[1]
    v = torch.randn(n, dtype=A.dtype, device=A.device)
    v = v / torch.linalg.norm(v)

    for _ in range(iters):
        v = A.T @ (A @ v)
        v = v / torch.linalg.norm(v)

    Av = A @ v
    return float(torch.dot(Av, Av).item())


def objective(
    A: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    p: float,
    lam: float,
) -> float:
    residual = A @ x - b
    data_term = 0.5 * torch.dot(residual, residual)
    reg_term = lam * torch.sum(torch.pow(torch.abs(x), p))
    return float((data_term + reg_term).item())


def forward_backward_lasso(
    A: torch.Tensor,
    b: torch.Tensor,
    p: float,
    lam: float,
    alpha: float,
    method: str,
    num_iters: int,
):
    """
    Forward-Backward iterations for

        min_x 0.5 ||A x - b||_2^2 + lambda * sum_i |x_i|^p

    Step:

        y_k = x_k - alpha * A^T(A x_k - b)
        x_{k+1} = prox_{alpha * lambda |.|^p}(y_k)
    """

    prox = make_lp_prox(
        p=p,
        lam=alpha * lam,
        method=method,
    )

    x = torch.zeros(A.shape[1], dtype=A.dtype, device=A.device)

    obj_history = []

    t0 = time.perf_counter()

    for _ in range(num_iters):
        grad = A.T @ (A @ x - b)
        y = x - alpha * grad
        x = prox(y)

        obj_history.append(objective(A, b, x, p=p, lam=lam))

    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0

    return x, obj_history, elapsed_ms


def main():
    p = 1 / 3
    lam = 0.05
    num_iters = 100

    A, b, x_true = make_synthetic_problem(
        m=400,
        n=800,
        sparsity=40,
        noise_std=0.01,
        seed=0,
    )

    L = spectral_norm_squared(A)
    alpha = 0.95 / L

    print("Lp-Lasso Forward-Backward demo")
    print("=" * 80)
    print(f"p          : {p}")
    print(f"lambda     : {lam}")
    print(f"A shape    : {tuple(A.shape)}")
    print(f"L estimate : {L:.6f}")
    print(f"alpha      : {alpha:.6f}")
    print(f"iterations : {num_iters}")
    print("=" * 80)

    methods = ["poly", "newton8", "newton12", "bisect40"]

    histories = {}
    rows = []

    for method in methods:
        x, history, elapsed_ms = forward_backward_lasso(
            A=A,
            b=b,
            p=p,
            lam=lam,
            alpha=alpha,
            method=method,
            num_iters=num_iters,
        )

        histories[method] = history

        final_obj = history[-1]
        rel_err = torch.linalg.norm(x - x_true) / torch.maximum(
            torch.tensor(1.0, dtype=x.dtype),
            torch.linalg.norm(x_true),
        )

        row = {
            "method": method,
            "total_ms": elapsed_ms,
            "ms_per_iter": elapsed_ms / num_iters,
            "final_objective": final_obj,
            "relative_error_to_true": float(rel_err.item()),
            "nonzeros_abs_gt_1e-6": int((torch.abs(x) > 1e-6).sum().item()),
        }
        rows.append(row)

        print(
            f"{method:10s}: "
            f"total={elapsed_ms:9.3f} ms, "
            f"ms/iter={elapsed_ms / num_iters:8.4f}, "
            f"final_obj={final_obj:.6e}, "
            f"rel_err={float(rel_err.item()):.3e}, "
            f"nnz={(torch.abs(x) > 1e-6).sum().item()}"
        )

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))

    for method in methods:
        plt.plot(histories[method], label=method, linewidth=1.5)

    plt.xlabel("iteration")
    plt.ylabel("objective")
    plt.title("Lp-Lasso Forward-Backward convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = output_dir / "lp_lasso_fb_convergence_demo.png"
    plt.savefig(output_path, dpi=200)

    print(f"\nSaved convergence figure to: {output_path}")


if __name__ == "__main__":
    main()