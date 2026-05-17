from __future__ import annotations

import time
from typing import Callable, Dict, Tuple

import numpy as np
import torch


def get_device(device: str = "auto") -> torch.device:
    """
    Choose a torch device.

    device can be:
    - "auto"
    - "cpu"
    - "cuda"
    - "mps"

    On macOS Apple Silicon, "mps" may be available.
    On Colab / NVIDIA machines, "cuda" may be available.
    """

    device = device.lower()

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")

    if device == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device: {device!r}")


def synchronize_device(device: torch.device) -> None:
    """
    Synchronize accelerator device when needed.
    """

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        # Available in recent PyTorch versions.
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def make_test_vector(
    n: int,
    tau: float,
    active_ratio: float = 0.9,
    ymax_factor: float = 10.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    seed: int | None = 0,
) -> torch.Tensor:
    """
    Generate a test vector y with controlled active ratio.

    inactive: |y| in [0, tau)
    active:   |y| in [tau, ymax_factor * tau]
    """

    device = torch.device(device)

    if n <= 0:
        raise ValueError("n must be positive.")

    if not (0.0 <= active_ratio <= 1.0):
        raise ValueError("active_ratio must be in [0, 1].")

    if ymax_factor <= 1.0:
        raise ValueError("ymax_factor must be greater than 1.")

    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    else:
        generator = None

    u = torch.rand(n, device=device, dtype=dtype, generator=generator)
    is_active = u < active_ratio

    a_inactive = tau * torch.rand(n, device=device, dtype=dtype, generator=generator)
    a_active = tau * (
        1.000001
        + (ymax_factor - 1.000001)
        * torch.rand(n, device=device, dtype=dtype, generator=generator)
    )

    a = torch.where(is_active, a_active, a_inactive)

    signs = torch.where(
        torch.rand(n, device=device, dtype=dtype, generator=generator) < 0.5,
        torch.tensor(-1.0, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    )

    return signs * a


def time_function(
    fn: Callable[[torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    warmup: int = 5,
    repeats: int = 20,
) -> Tuple[Dict[str, float], torch.Tensor]:
    """
    Time a function y -> x.

    Uses wall-clock timing. This works on CPU, CUDA, and MPS.
    For CUDA/MPS, synchronization is used before and after timing.
    """

    if warmup < 0:
        raise ValueError("warmup must be non-negative.")

    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    device = y.device

    out = None

    for _ in range(warmup):
        out = fn(y)

    synchronize_device(device)

    times_ms = []

    for _ in range(repeats):
        synchronize_device(device)
        t0 = time.perf_counter()

        out = fn(y)

        synchronize_device(device)
        t1 = time.perf_counter()

        times_ms.append((t1 - t0) * 1000.0)

    # Touch output to discourage lazy behavior.
    checksum = float(out[: min(1024, out.numel())].sum().item())

    stats = {
        "median_ms": float(np.median(times_ms)),
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "checksum": checksum,
    }

    return stats, out


def tensor_summary(x: torch.Tensor) -> Dict[str, float | bool]:
    """
    Basic numerical summary of a tensor.
    """

    return {
        "all_finite": bool(torch.isfinite(x).all().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
    }