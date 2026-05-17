from fast_lp_prox import (
    candidate_coeff_table_names,
    get_or_build_cheb_approx,
    p_to_rational_token,
)


def test_p_to_rational_token():
    assert p_to_rational_token(1 / 3) == "p_1_3"
    assert p_to_rational_token(2 / 5) == "p_2_5"
    assert p_to_rational_token(3 / 4) == "p_3_4"
    assert p_to_rational_token(0.5) == "p_1_2"


def test_candidate_coeff_table_names():
    names = candidate_coeff_table_names(1 / 3, dtype="float32")

    assert "p_1_3_float32.json" in names


def test_get_or_build_cheb_approx_fallback_builds():
    approx = get_or_build_cheb_approx(
        p=0.5,
        dtype="float32",
        use_table=True,
        target_error=1e-4,
        max_degree=24,
        n_fit=512,
        n_val=1001,
        verbose=False,
    )

    assert approx.p == 0.5
    assert approx.degree <= 24
    assert approx.estimated_error <= 1e-4