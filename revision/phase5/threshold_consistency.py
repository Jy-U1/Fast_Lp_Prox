"""Exact strict activation threshold test for float32/float64 torch tensors.

For y represented in float32, compare |y| against the greatest float32
number <= the float64-computed mathematical activation threshold tau.
This is mathematically equivalent to comparing float64(y) > float64(tau),
but avoids a whole-vector float64 promotion in the online kernel.
"""
from functools import lru_cache
import math
import numpy as np
import torch


@lru_cache(maxsize=256)
def float32_floor_threshold(tau: float) -> float:
    tau = float(tau)
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("finite nonnegative threshold required")
    # The intended parameter range of the scalar solver uses finite
    # float32 thresholds; fail explicitly if the scale exceeds it.
    rounded = np.float32(tau)
    if not np.isfinite(rounded):
        raise OverflowError("threshold cannot be represented in float32")
    if float(rounded) > tau:
        rounded = np.nextafter(rounded, np.float32(-math.inf))
    return float(rounded)


def strict_activation_mask(yabs: torch.Tensor, tau: float) -> torch.Tensor:
    if yabs.dtype == torch.float32:
        return yabs > float32_floor_threshold(tau)
    if yabs.dtype == torch.float64:
        return yabs > tau
    raise TypeError("supported dtypes: torch.float32 and torch.float64")
