"""Check support decisions for all legacy Phase-3 scalar backends.

Use --device cuda --n 1000000 for the same 1M-input size as Phase-5 A100.
This is a support-only smoke test, not an accuracy-matched timing experiment.
"""
import argparse

import numpy as np
import torch

from fast_lp_prox.constants import lp_constants
from fast_lp_prox.revision_bench import fit_degree_family, make_preloaded_prox
from revision.phase5.scripts.benchmark_phase5_scalar import generate_shared_inputs


CASES = (
    (1.0 / 3.0, 6, 3),
    (0.4, 6, 3),
    (0.75, 12, 4),
    (0.9, 24, 4),
    (0.99, 54, 5),
    (0.999, 116, 4),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.n < 32:
        parser.error("--n must be at least 32")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(args.device)
    total_errors = 0
    for p, degree, nstep in CASES:
        tau = lp_constants(p, 1.0).tau
        y = generate_shared_inputs(n=args.n, p=p, lam=1.0,
                                   active_fraction=0.9, ymax_factor=10.0,
                                   seed=args.seed, dtype=torch.float32,
                                   device=device)
        expected = y.double().abs() > tau
        fit = fit_degree_family(p, [degree], n_fit=2048,
                                n_validation=2049)
        configs = (("chebyshev", degree, fit.coefficients[:degree + 1]),
                   ("newton", nstep, None),
                   ("bisection", 20, None))
        for family, work, coefficients in configs:
            evaluator = make_preloaded_prox(
                p=p, lam=1.0, family=family, device=device,
                dtype=torch.float32, work=work, coefficients=coefficients)
            with torch.no_grad():
                support = evaluator(y) != 0.0
                mismatches = int((support != expected).sum().item())
            total_errors += mismatches
            print(f"p={p:.6g} method={family:9} mismatches={mismatches:5d}/{args.n}")
    if total_errors:
        raise SystemExit(f"FAIL: {total_errors} support mismatches")
    print("PASS: zero support mismatches for all 18 checks")


if __name__ == "__main__":
    main()
