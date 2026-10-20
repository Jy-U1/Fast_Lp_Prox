from revision.phase5.scripts.summarize_phase5 import fastest_passing


def test_fastest_passing_never_selects_inaccurate_method():
    base={'p':'0.75','family':'fixed_series','configuration':'a',
          'meets_accuracy_target':'False','all_finite':'True','support_mismatch_count':'0',
          'median_ms':'1','max_norm_error':'0.01','max_active_stationarity_residual':'0',
          'mean_prox_objective_gap':'0','fallback_count':'','offline_family_seconds':'0'}
    rows=[base,dict(base,configuration='b',median_ms='3',
                    meets_accuracy_target='True',max_norm_error='1e-8')]
    selected=fastest_passing(rows)
    assert len(selected)==1
    assert selected[0]['configuration']=='b'
    assert selected[0]['status']=='PASS'
    fail=fastest_passing([base])
    assert fail[0]['status']=='NO_PASSING_CONFIGURATION'
