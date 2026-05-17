from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

from fast_lp_prox import build_cheb_coeffs, save_cheb_approx


def parse_p(value: str) -> float:
    """
    Parse p from either decimal form or rational form.

    Examples:
        0.5
        1/3
        2/5
        3/4
    """

    value = value.strip()

    if "/" in value:
        return float(Fraction(value))

    return float(value)


def p_to_filename_token(p_text: str) -> str:
    """
    Convert p string to a safe filename token.

    Examples:
        1/3 -> p_1_3
        0.5 -> p_0_5
    """

    token = p_text.strip().replace("/", "_").replace(".", "_")
    return f"p_{token}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Chebyshev coefficient table for fast-lp-prox."
    )

    parser.add_argument(
        "--p",
        type=str,
        required=True,
        help="Exponent p, for example 1/3, 2/5, 3/4, or 0.5.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
    )

    parser.add_argument("--target-error", type=float, default=2e-7)
    parser.add_argument("--max-degree", type=int, default=128)
    parser.add_argument("--n-fit", type=int, default=8192)
    parser.add_argument("--n-val", type=int, default=80001)

    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON path. If omitted, saves into src/fast_lp_prox/coeff_tables/.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print degree search progress.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    p = parse_p(args.p)

    print("=" * 80)
    print("fast-lp-prox: generate Chebyshev coefficients")
    print("=" * 80)
    print(f"p            : {args.p} = {p}")
    print(f"dtype        : {args.dtype}")
    print(f"target_error : {args.target_error}")
    print(f"max_degree   : {args.max_degree}")
    print(f"n_fit        : {args.n_fit}")
    print(f"n_val        : {args.n_val}")
    print("=" * 80)

    approx = build_cheb_coeffs(
        p=p,
        target_error=args.target_error,
        max_degree=args.max_degree,
        n_fit=args.n_fit,
        n_val=args.n_val,
        verbose=args.verbose,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        token = p_to_filename_token(args.p)
        output_path = Path("src") / "fast_lp_prox" / "coeff_tables" / f"{token}_{args.dtype}.json"

    save_cheb_approx(
        approx,
        output_path,
        dtype=args.dtype,
    )

    print("\nGenerated coefficient table:")
    print(f"degree          : {approx.degree}")
    print(f"estimated_error : {approx.estimated_error:.3e}")
    print(f"saved to        : {output_path}")


if __name__ == "__main__":
    main()