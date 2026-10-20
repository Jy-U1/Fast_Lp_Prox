from types import SimpleNamespace
import sys
import types

import numpy as np
import torch
import pytest

from revision.phase5.scripts.benchmark_phase5_scalar import existing_backends


def test_phase3_existing_adapter_calls_unmodified_api(monkeypatch):
    """Interface contract mirrored from proven Phase-4 diagnostics usage."""
    imported = []
    mod=types.ModuleType('fast_lp_prox.revision_bench')
    def fit_degree_family(p,degrees,*,n_fit,n_validation):
        imported.append(('fit',p,degrees,n_fit,n_validation))
        return SimpleNamespace(coefficients=np.ones(21))
    def make_preloaded_prox(**kwargs):
        imported.append(('preload',kwargs['family'],kwargs['work'],kwargs['coefficients']))
        return lambda x: x.clone()
    mod.fit_degree_family=fit_degree_family
    mod.make_preloaded_prox=make_preloaded_prox
    pkg=types.ModuleType('fast_lp_prox')
    pkg.__path__=[]
    monkeypatch.setitem(sys.modules,'fast_lp_prox',pkg)
    monkeypatch.setitem(sys.modules,'fast_lp_prox.revision_bench',mod)
    backends=existing_backends(.75,1.,device=torch.device('cpu'),dtype=torch.float32,
                               cheb=(6,12),newton=(4,),bisect=(20,),
                               n_fit=1024,n_validation=2049)
    assert len(backends)==4
    assert imported[0]==('fit',.75,[6,12],1024,2049)
    assert [row[0] for row in backends]==['chebyshev','chebyshev','newton','bisection']
    assert len(imported[1][3])==7
    assert len(imported[2][3])==13
    assert imported[3][3] is None
    assert torch.equal(backends[0][2](torch.ones(4)),torch.ones(4))
