import numpy as np

from fast_lp_prox import build_cheb_coeffs, load_cheb_approx, save_cheb_approx


def test_save_and_load_cheb_approx(tmp_path):
    approx = build_cheb_coeffs(
        p=0.5,
        target_error=1e-4,
        max_degree=24,
        n_fit=512,
        n_val=1001,
        verbose=False,
    )

    path = tmp_path / "coeffs.json"

    save_cheb_approx(approx, path, dtype="float32")
    loaded = load_cheb_approx(path)

    assert loaded.p == approx.p
    assert loaded.degree == approx.degree
    assert loaded.target_error == approx.target_error
    assert loaded.estimated_error == approx.estimated_error
    assert np.allclose(loaded.coeffs, approx.coeffs)