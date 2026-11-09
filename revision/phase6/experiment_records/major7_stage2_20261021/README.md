# Original Stage-2 MB parameter scans

These are unmodified measurements produced on the user's Mac. `standard/` contains 6 exponents x 4 N x 2 T = 48 rows; `longT/` contains 4 exponents x 3 N x 1 T = 12 rows. Input SHA-256 hashes and all parameter metadata are retained.

The paper-consistent raw Mellin--Barnes candidate is tested without bisection fallback. The integration is ordinary floating-point quadrature, not a formally certified enclosure. These experiments use the full-active radius policy, not the original MATLAB default R=0.9*u_star. Compare with the Stage-3 same-CPU results in the adjacent folder; never mix these CPU timings with the A100 benchmarks.
