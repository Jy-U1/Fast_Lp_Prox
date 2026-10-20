import math
import pytest
import torch

from revision.phase5.threshold_consistency import (
    float32_floor_threshold, strict_activation_mask,
)
from revision.phase5.backends import proximal_constants
from revision.phase5.scripts.benchmark_phase5_scalar import generate_shared_inputs


@pytest.mark.parametrize("p", [1/3, .4, .75, .9, .99, .999])
def test_float32_threshold_consistent_with_float64_reference(p):
    tau = proximal_constants(p, 1.0).tau
    q = torch.tensor(tau, dtype=torch.float32)
    values = torch.stack((torch.nextafter(q, torch.tensor(-math.inf, dtype=torch.float32)),
                          q,
                          torch.nextafter(q, torch.tensor(math.inf, dtype=torch.float32))))
    actual = strict_activation_mask(values.abs(), tau)
    expected = values.double().abs() > tau
    assert torch.equal(actual, expected)
    floor = float32_floor_threshold(tau)
    assert floor <= tau
    assert float(torch.nextafter(torch.tensor(floor), torch.tensor(math.inf))) > tau


@pytest.mark.parametrize("p", [1/3, .4, .75, .9, .99, .999])
def test_generator_sample_threshold_support_agreement(p):
    y = generate_shared_inputs(
        n=20000, p=p, lam=1.0, active_fraction=0.9,
        ymax_factor=10.0, seed=0, dtype=torch.float32,
        device=torch.device("cpu"),
    )
    tau = proximal_constants(p, 1.).tau
    assert torch.equal(strict_activation_mask(y.abs(), tau),
                       y.double().abs() > tau)


def test_float64_threshold_no_rounding():
    tau=proximal_constants(.9, 1.0).tau
    ys=torch.tensor([tau*(1-1e-8), tau*(1+1e-8)], dtype=torch.float64)
    assert strict_activation_mask(ys,tau).tolist()==[False,True]
