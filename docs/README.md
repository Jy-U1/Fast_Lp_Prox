# Documentation

This directory contains benchmark reports, reproduction workflows, and application notes for `fast-lp-prox`.

The main contribution of this project is fast evaluation of nonconvex Lp proximal operators. Proximal-gradient and Forward-Backward routines are included as lightweight application demos and benchmark workloads, not as the main focus of the package.

---

## Benchmark reports

- [A100 CUDA prox benchmark](benchmark_a100.md)  
  Isolated CUDA benchmark for the Lp prox evaluator. Compares `poly`, `newton8`, `newton12`, and `bisect40` on an NVIDIA A100 GPU.

- [Lp-Lasso Forward-Backward CUDA demo](lp_lasso_fb_cuda.md)  
  Application benchmark showing how the Lp prox evaluator can be inserted into a Forward-Backward loop for a synthetic nonconvex Lp-Lasso problem.

- [CUDA prox scaling benchmark](prox_scaling_cuda.md)  
  Scaling benchmark for isolated Lp prox evaluation from `n = 10,000` to `n = 10,000,000`.
---

## Reproduction workflows

- [Colab CUDA benchmark workflow](colab_cuda_benchmark_workflow.md)  
  Step-by-step workflow for reproducing CUDA benchmarks in Google Colab, including private repo cloning, package installation, smoke tests, benchmark sweeps, validation, figure generation, and result packaging.

---

## Assets

Selected figures used by the reports are stored under:

```text
docs/assets/