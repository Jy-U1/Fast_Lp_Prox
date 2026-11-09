"""CPU checks for published MATLAB formula vs arXiv v2 Eq. (21) sign."""
import math
import numpy as np
import pytest
from scipy.special import gamma
from revision.phase6.mb_port import (
    constants, coefficients_scaled, series_normalized, positive_root_bisect,
    mb_grid, complex_mb_correction, MBPort,
    matlab_empirical_series_indicator, matlab_empirical_mb_indicator
)
from revision.phase6.scripts.benchmark_mb_cpu import inputs_phase5


def test_coefficients_reproduce_lagrange_gamma_expression():
    for p in [1/3,.5,.75,.9]:
        u=constants(p,1.).ustar
        coeffs=coefficients_scaled(p,8)
        assert coeffs[0]==pytest.approx(u)
        for n in range(1,9):
            original=gamma((2-p)*n-1)/gamma((1-p)*n)/math.factorial(n)
            assert coeffs[n-1] == pytest.approx(original*u**n,rel=1e-13)


def test_threshold_and_scale():
    for p in [1/3,.5,.9,.999]:
        c=constants(p,.8)
        assert c.rho<c.tau and c.qp<1
        xs=positive_root_bisect(np.array([-2*c.tau, c.tau, 0., c.tau*(1+1e-7)]),p=p,lam=.8)
        assert xs[0]<0 and xs[1]==0 and xs[2]==0 and xs[3]>=c.rho


def test_reference_stationarity_and_mpmath():
    mp=pytest.importorskip('mpmath');mp.mp.dps=60
    for p,lam,y in [(1/3,.8,1.5),(.5,1.,2.5),(.9,1.,1.3)]:
        c=constants(p,lam)
        yy=np.array([y]);x=positive_root_bisect(yy,p=p,lam=lam)[0]
        assert x>=c.rho
        assert abs(x-y+lam*p*x**(p-1))<3e-15
        x_m=mp.findroot(lambda z:z-mp.mpf(str(y))+mp.mpf(str(lam))*mp.mpf(str(p))*z**(mp.mpf(str(p))-1),
                         (mp.mpf(str(c.rho)),mp.mpf(str(y))),solver='anderson')
        assert abs(float(x_m)-x) < 4e-14


def test_mellin_barnes_paper_sign_corrects_matlab_sign():
    # These two modes differ only by the sign in front of the MB correction.
    # The published MATLAB gamma ratio has no minus sign, unlike v2 H(s).
    p=.5;lam=.8;y=np.array([1.5]);reference=positive_root_bisect(y,p=p,lam=lam)
    src=MBPort(p=p,lam=lam,n=2,T=60,nt=6000,mode='matlab')(y)
    paper=MBPort(p=p,lam=lam,n=2,T=60,nt=6000,mode='paper')(y)
    es=float(abs(src.x[0]-reference[0]));ep=float(abs(paper.x[0]-reference[0]))
    assert es>2e-2
    assert ep<4e-4
    assert es>50*ep
    assert src.z_series[0]==paper.z_series[0]
    assert src.z_mb[0]==paper.z_mb[0]


def test_mellin_barnes_complex_remainder_not_silent():
    p=2/3;lam=.8;y=np.array([1.5]);r=MBPort(p=p,lam=lam,n=3,T=60,nt=4000,mode='paper')(y)
    assert r.max_imag_normalized>1e-7
    assert r.fallback_count==0
    assert r.statuses[0]=='raw_unverified'
    assert np.iscomplexobj(r.x_complex)


def test_sampled_gammas_are_not_certificates():
    c=constants(.5,1)
    a=matlab_empirical_series_indicator(p=.5,N=3,R=.75*c.ustar)
    b=matlab_empirical_mb_indicator(p=.5,N=3,R=.75*c.ustar,T=60,samples=64)
    assert np.isfinite(a) and a>0
    assert np.isfinite(b) and b>0


def test_source_domain_rejection_and_full_active_policy():
    c=constants(.99,1.)
    y=np.array([c.tau*(1+1e-7)])
    assert c.qp>.9
    with pytest.raises(ValueError,match='epsilon>R'):
        MBPort(p=.99,lam=1,n=2,T=50,nt=400,mode='matlab')(y)
    R=c.ustar*(1+c.qp)/2
    assert np.all(MBPort(p=.99,lam=1,n=2,T=50,nt=400,mode='matlab',R=R)(y).statuses=='raw_unverified')


def test_residual_checked_or_bisection_fallback():
    for p in [.5,.9]:
        c=constants(p,1)
        y=np.array([c.tau*(1+d) for d in [1e-7,.001,.05,.5,2.]])
        R=c.ustar*max(.9,.5*(1+c.qp))
        r=MBPort(p=p,lam=1,n=3,T=60,nt=1200,mode='paper',R=R)(y,tol=1e-6,imag_tol=1e-6)
        ref=positive_root_bisect(y,p=p,lam=1)
        assert np.max(np.abs(r.x-ref)/y)<1e-6
        assert r.fallback_count>0
        assert all(v in ('bisection_fallback','accepted_residual') for v in r.statuses)
        assert r.max_accepted_residual_indicator<=1e-6


def test_series_converges_moderate_exponent():
    p=1/3;lam=1;c=constants(p,lam)
    y=np.array([c.tau*1.1,c.tau*2,c.tau*4]);eps=lam*p*y**(p-2)
    z16=series_normalized(eps,p=p,n=16)
    z64=series_normalized(eps,p=p,n=64)
    ref=positive_root_bisect(y,p=p,lam=lam)/y
    assert np.max(abs(z64-ref)) < np.max(abs(z16-ref))


def test_input_generator_deterministic():
    y1=inputs_phase5(n=128,p=.9,lam=1,seed=0)
    y2=inputs_phase5(n=128,p=.9,lam=1,seed=0)
    np.testing.assert_array_equal(y1,y2)
    assert np.any(y1>0) and np.any(y1<0)


def test_unsupported_inputs_are_rejected():
    with pytest.raises(ValueError): constants(1,1)
    with pytest.raises(ValueError): constants(.5,0)
    with pytest.raises(ValueError): mb_grid(.5,2,20,3)
    with pytest.raises(ValueError): complex_mb_correction(np.array([-1.]),mb_grid(.5,2,20,64))



def test_matlab_parameter_selection_is_labeled_empirical():
    from revision.phase6.mb_port import choose_matlab_parameters
    c=constants(2/3,.8)
    x=choose_matlab_parameters(p=2/3,R=.4,eta=1e-3,maxN=15,maxT=80,
                                envelope_samples=64)
    assert 1<=x.N<=15 and x.T>=50 and x.sigma==x.N+.5
    assert x.series_estimate>0 and x.mb_estimate>0
    # Source can hit budgets without making the empirical targets.
    c2=constants(.999,1)
    difficult=choose_matlab_parameters(p=.999,R=(c2.qp+1)/2*c2.ustar,
                            eta=1e-8,maxN=4,maxT=80,envelope_samples=64)
    assert not difficult.meets_empirical_series_target
