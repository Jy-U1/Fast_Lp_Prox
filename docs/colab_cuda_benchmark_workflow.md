# Colab CUDA benchmark workflow

This document describes a reproducible Colab workflow for running CUDA benchmarks for `fast-lp-prox`.

The focus of this workflow is the isolated Lp proximal operator evaluator, not the Forward-Backward / proximal-gradient application demo.

---

## 1. Start a Colab GPU runtime

In Colab:

```text
Runtime -> Change runtime type -> GPU
```

Then verify the GPU:

```bash
!nvidia-smi
```

Check PyTorch and CUDA:

```python
import torch, sys, platform

print("Python:", sys.version)
print("Platform:", platform.platform())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("Total memory GB:", props.total_memory / 1024**3)
```

If `torch.cuda.is_available()` is `False`, stop here and switch the Colab runtime to GPU before running benchmarks.

---

## 2. Clone the private GitHub repository

For a private repository, store a GitHub Personal Access Token in Colab Secrets.

Recommended secret name:

```text
GITHUB_TOKEN
```

The secret value should be the token string itself, for example a value beginning with `github_pat_...` or `ghp_...`.

Load the token:

```python
from google.colab import userdata
from google.colab.errors import SecretNotFoundError, NotebookAccessError
import os

try:
    token = userdata.get("GITHUB_TOKEN")
except SecretNotFoundError:
    raise RuntimeError(
        "Missing Colab secret: GITHUB_TOKEN. "
        "Create it in the Colab Secrets panel and restart the runtime."
    )
except NotebookAccessError:
    raise RuntimeError(
        "GITHUB_TOKEN exists, but this notebook does not have access. "
        "Enable notebook access for this secret in the Colab Secrets panel."
    )

assert token and len(token) > 0, "GITHUB_TOKEN is empty"
os.environ["GITHUB_TOKEN"] = token

print("GITHUB_TOKEN loaded, length:", len(token))
```

Clone the repository:

```bash
!rm -rf YU
!git clone https://$GITHUB_TOKEN@github.com/Jy-U1/YU.git
%cd YU
```

Do not print the token itself.

---

## 3. Install the package

```bash
!python -m pip install --upgrade pip
!python -m pip install -e ".[dev]"
```

Verify the project root:

```bash
!pwd
!ls -la
```

The root directory should contain:

```text
README.md
pyproject.toml
src/
scripts/
examples/
tests/
```

---

## 4. Run tests and basic examples

```bash
!python -m pytest -q
```

Expected result:

```text
26 passed
```

Run basic examples:

```bash
!python examples/basic_usage.py
!python examples/compare_methods.py
```

---

## 5. Save environment information

Create output directories:

```bash
!mkdir -p results_cuda figures_cuda
```

Save GPU information:

```bash
!nvidia-smi
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > results_cuda/gpu_info.csv
```

Save PyTorch environment information:

```python
import torch, sys, platform, json

env = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}

with open("results_cuda/torch_env.json", "w") as f:
    json.dump(env, f, indent=2)

env
```

---

## 6. CUDA smoke benchmark

Before running the full sweep, run a small smoke benchmark for `p = 1/3`.

```bash
!python scripts/benchmark_prox.py \
  --device cuda \
  --dtype float32 \
  --p 0.3333333333333333 \
  --n 100000 \
  --methods poly,newton8,newton12,bisect40 \
  --warmup 3 \
  --repeats 5 \
  --output results_cuda/prox_smoke_p_1_3_cuda.csv
```

Inspect the output:

```bash
!ls -lh results_cuda
!cat results_cuda/prox_smoke_p_1_3_cuda.csv
```

Check:

```text
device should be cuda
all_finite should be True
methods should include poly, newton8, newton12, bisect40
```

---

## 7. CUDA smoke validation

Run accuracy validation:

```bash
!python scripts/validate_accuracy.py \
  --device cuda \
  --dtype float32 \
  --p 0.3333333333333333 \
  --n 100000 \
  --sample-n 10000 \
  --methods poly,newton8,newton12,bisect40 \
  --output results_cuda/accuracy_smoke_p_1_3_cuda.csv
```

Run residual validation:

```bash
!python scripts/validate_residual.py \
  --device cuda \
  --dtype float32 \
  --p 0.3333333333333333 \
  --n 100000 \
  --sample-n 10000 \
  --methods poly,newton8,newton12,bisect40 \
  --output results_cuda/residual_smoke_p_1_3_cuda.csv
```

Inspect:

```bash
!cat results_cuda/accuracy_smoke_p_1_3_cuda.csv
!cat results_cuda/residual_smoke_p_1_3_cuda.csv
```

---

## 8. Full CUDA prox benchmark sweep

Run benchmark sweep for the built-in coefficient table cases:

```bash
%%bash
set -e

mkdir -p results_cuda figures_cuda

for item in \
  "p_1_3 0.3333333333333333" \
  "p_2_5 0.4" \
  "p_3_4 0.75"
do
  set -- $item
  PNAME=$1
  P=$2

  echo "Running prox benchmark for $PNAME, p=$P"

  python scripts/benchmark_prox.py \
    --device cuda \
    --dtype float32 \
    --p $P \
    --n 1000000 \
    --methods poly,newton8,newton12,bisect40 \
    --warmup 10 \
    --repeats 30 \
    --output results_cuda/prox_benchmark_${PNAME}_cuda.csv
done
```

---

## 9. Full CUDA accuracy and residual validation

Run accuracy validation:

```bash
%%bash
set -e

mkdir -p results_cuda

for item in \
  "p_1_3 0.3333333333333333" \
  "p_2_5 0.4" \
  "p_3_4 0.75"
do
  set -- $item
  PNAME=$1
  P=$2

  echo "Running accuracy validation for $PNAME, p=$P"

  python scripts/validate_accuracy.py \
    --device cuda \
    --dtype float32 \
    --p $P \
    --n 1000000 \
    --sample-n 100000 \
    --methods poly,newton8,newton12,bisect40 \
    --output results_cuda/accuracy_${PNAME}_cuda.csv
done
```

Run residual validation:

```bash
%%bash
set -e

mkdir -p results_cuda

for item in \
  "p_1_3 0.3333333333333333" \
  "p_2_5 0.4" \
  "p_3_4 0.75"
do
  set -- $item
  PNAME=$1
  P=$2

  echo "Running residual validation for $PNAME, p=$P"

  python scripts/validate_residual.py \
    --device cuda \
    --dtype float32 \
    --p $P \
    --n 1000000 \
    --sample-n 100000 \
    --methods poly,newton8,newton12,bisect40 \
    --output results_cuda/residual_${PNAME}_cuda.csv
done
```

---

## 10. Generate figures

Benchmark figures:

```bash
%%bash
set -e

mkdir -p figures_cuda

for CSV in results_cuda/prox_benchmark_*_cuda.csv
do
  echo "Plotting $CSV"
  python scripts/plot_benchmark_results.py \
    --input "$CSV" \
    --output-dir figures_cuda
done
```

Validation figures:

```bash
%%bash
set -e

for CSV in results_cuda/accuracy_*_cuda.csv
do
  echo "Plotting $CSV"
  python scripts/plot_validation_results.py \
    --input "$CSV" \
    --kind accuracy \
    --output-dir figures_cuda
done

for CSV in results_cuda/residual_*_cuda.csv
do
  echo "Plotting $CSV"
  python scripts/plot_validation_results.py \
    --input "$CSV" \
    --kind residual \
    --output-dir figures_cuda
done
```

Inspect figures:

```bash
!ls -lh figures_cuda
```

---

## 11. Package and download results

```bash
!zip -r fast_lp_prox_cuda_results.zip results_cuda figures_cuda
```

```python
from google.colab import files
files.download("fast_lp_prox_cuda_results.zip")
```

Do not commit this zip file to the repository. It should be stored as a local or external benchmark artifact.

---

## 12. Expected A100 reference result

On one A100-SXM4-80GB run with `float32`, `n = 1,000,000`, `lambda = 1.0`, and `active_ratio = 0.9`, the polynomial evaluator achieved:

| p | poly median time | poly throughput | speedup vs newton8 | speedup vs newton12 | speedup vs bisect40 |
|---:|---:|---:|---:|---:|---:|
| 1/3 | 0.663 ms | 1508 M elems/s | 1.87x | 2.64x | 7.25x |
| 2/5 | 0.663 ms | 1509 M elems/s | 1.90x | 2.63x | 7.29x |
| 3/4 | 0.928 ms | 1078 M elems/s | 1.32x | 1.86x | 5.14x |

See [`docs/benchmark_a100.md`](benchmark_a100.md) for the full A100 benchmark report.

---

## 13. Common issues

### `SecretNotFoundError: Secret GITHUB_TOKEN does not exist`

Check that:

- the secret name is exactly `GITHUB_TOKEN`
- the secret has been saved in the Colab Secrets panel
- notebook access is enabled for the secret
- the runtime has been restarted after creating the secret

### `NotebookAccessError`

The secret exists, but the notebook is not allowed to access it. Enable notebook access in the Colab Secrets panel.

### `torch.cuda.is_available()` is `False`

Check that the Colab runtime type is set to GPU:

```text
Runtime -> Change runtime type -> GPU
```

Then restart the runtime and run the CUDA check again.

### CUDA out-of-memory

Reduce `--n` and/or `--sample-n`. For example:

```bash
--n 100000
--sample-n 10000
```

### Benchmark is much slower than expected

Possible reasons:

- Colab assigned a slower GPU such as T4 instead of A100
- small `n` does not saturate the GPU
- the first run includes warm-up or compilation overhead
- the runtime is shared or throttled

Always record `gpu_info.csv` and `torch_env.json`.
