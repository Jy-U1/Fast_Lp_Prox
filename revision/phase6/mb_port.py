"""Python reimplementation of the released Mellin--Barnes lp-prox evaluator.

Provenance: https://github.com/lshen03/Lp-Proximity-Operator
MATLAB reference: xplus_lp_hybrid.m (accessed 2026-10-20)
Paper: arXiv:2606.31951v2, Section 5.2, Eq. (21)-(28).

IMPORTANT: two modes are intentionally exposed:
  matlab  -- PLUS the gamma-ratio integral, matching the publicly released
             xplus_lp_hybrid.m source. This differs in sign from the paper.
  paper   -- MINUS the same gamma-ratio integral, following the paper's
             H(s) = -Gamma(-s)Gamma((2-p)s-1)/Gamma((1-p)s).
Neither mode has a rigorous floating-point error certificate: trapezoidal
quadrature, empirical gamma envelopes, and ordinary float64 are non-validated.

This is an independent mathematical reimplementation, not a verbatim copy of
upstream MATLAB code. Do not describe a safeguarded bisection fallback as a
raw MATLAB-method timing.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Literal

import numpy as np
from scipy.special import gammaln, loggamma
from scipy.integrate import trapezoid


Mode = Literal["matlab", "paper"]


@dataclass(frozen=True)
class Constants:
    p: float
    lam: float
    rho: float
    tau: float
    ustar: float
    qp: float


def constants(p: float, lam: float) -> Constants:
    p, lam = float(p), float(lam)
    if not (math.isfinite(p) and 0 < p < 1 and math.isfinite(lam) and lam > 0):
        raise ValueError("Require finite 0 < p < 1 and lam > 0")
    logrho = math.log(2 * lam * (1 - p)) / (2 - p)
    rho = math.exp(logrho)
    tau = rho * (2 - p) / (2 * (1 - p))
    logu = (1-p) * math.log1p(-p) - (2-p) * math.log(2-p)
    ustar = math.exp(logu)
    qp = p * (2.0 ** (1.0-p))
    return Constants(p, lam, rho, tau, ustar, qp)


@lru_cache(maxsize=128)
def coefficients_scaled(p: float, n: int) -> np.ndarray:
    """b_k = a_k ustar**k, a_k as in Eq. (20), for k=1..n."""
    if not (0 < p < 1 and 1 <= n <= 2048):
        raise ValueError("Require 0<p<1, 1<=N<=2048")
    ks = np.arange(1, n+1, dtype=np.float64)
    logu = (1-p)*math.log1p(-p) - (2-p)*math.log(2-p)
    logb = gammaln((2-p)*ks-1) - gammaln((1-p)*ks) - gammaln(ks+1) + ks*logu
    b = np.exp(logb)
    b.setflags(write=False)
    return b


def series_normalized(eps: np.ndarray, *, p: float, n: int) -> np.ndarray:
    """Evaluate z_N(eps) = 1 - sum_{k=1}^N a_k eps**k with Horner."""
    eps = np.asarray(eps, dtype=np.float64)
    if np.any(~np.isfinite(eps)) or np.any(eps < 0):
        raise ValueError("epsilon must be nonnegative and finite")
    u = constants(p, 1).ustar
    t = eps/u
    b = coefficients_scaled(p,n)
    v = np.full_like(t, b[-1])
    for a in b[-2::-1]:
        v = v*t + a
    return 1.0 - t*v


def positive_root_bisect(y: np.ndarray, *, p: float, lam: float,
                         steps: int=80) -> np.ndarray:
    """Numerical reference; 0 at/below activation. NOT interval certified."""
    c = constants(p,lam)
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise ValueError("nonfinite input")
    a = np.abs(y)
    active = a > c.tau
    ya = np.maximum(a, c.tau)
    lo = np.full_like(a, c.rho)
    hi = ya.copy()
    for _ in range(steps):
        mid = lo + (hi-lo)*0.5
        fmid = mid - ya + lam*p*np.power(mid,p-1)
        hi = np.where(fmid>0, mid, hi)
        lo = np.where(fmid<=0, mid, lo)
    return np.where(active, np.sign(y)*(lo+hi)/2, 0.0)


@dataclass(frozen=True)
class MBGrid:
    p: float
    n: int
    T: float
    nt: int
    sigma: float
    t: np.ndarray
    log_gamma_ratio: np.ndarray


@lru_cache(maxsize=32)
def mb_grid(p: float, n: int, T: float, nt: int, sigma: float|None=None) -> MBGrid:
    """Precompute complex Gamma ratio along the MATLAB vertical segment.

    The complex log-Gamma is SciPy's principal analytic continuation. The
    product is evaluated in the log domain to avoid overflow from exp(pi*|t|).
    """
    p=float(p); n=int(n); T=float(T); nt=int(nt)
    sigma=float(sigma) if sigma is not None else n+0.5
    if not (0<p<1 and 1<=n<=2048 and T>0 and math.isfinite(T)
            and 16<=nt<=100000 and n<sigma<n+1):
        raise ValueError("Bad MB contour options")
    t = np.linspace(-T,T,nt+1,dtype=np.float64)
    s = sigma + 1j*t
    # Gamma(-s)*Gamma((2-p)s-1)/Gamma((1-p)s), WITHOUT paper's minus.
    logH = loggamma(-s) + loggamma((2-p)*s-1) - loggamma((1-p)*s)
    if not np.all(np.isfinite(logH)):
        raise ArithmeticError("nonfinite gamma ratio on contour")
    t.setflags(write=False)
    logH.setflags(write=False)
    return MBGrid(p,n,T,nt,sigma,t,logH)


def complex_mb_correction(eps: np.ndarray, grid: MBGrid, *, chunk: int=32) -> np.ndarray:
    """MATLAB-sign integral: +1/(2pi) int_{-T}^T GammaRatio * (-eps)^s dt.

    Returns complex results: imaginary parts from finite quadrature are
    meaningful diagnostics and must not be silently discarded.
    """
    arr=np.asarray(eps,dtype=np.float64)
    shape=arr.shape
    flat=arr.reshape(-1)
    if np.any(~np.isfinite(flat)) or np.any(flat<=0):
        raise ValueError("MB epsilon must be strictly positive and finite")
    if chunk<1: raise ValueError("chunk must be positive")
    out=np.empty(len(flat), dtype=np.complex128)
    logH = grid.log_gamma_ratio[None,:]
    s=(grid.sigma+1j*grid.t)[None,:]
    for i in range(0,len(flat),chunk):
        ell=np.log(flat[i:i+chunk])[:,None] + 1j*np.pi
        integral=np.exp(logH+s*ell)
        with np.errstate(over='ignore', invalid='ignore'):
            out[i:i+chunk]=trapezoid(integral,grid.t,axis=1)/(2*np.pi)
    return out.reshape(shape)


@dataclass
class Evaluation:
    x: np.ndarray
    x_complex: np.ndarray
    z_series: np.ndarray
    z_mb: np.ndarray
    max_imag_normalized: float
    n: int
    T: float
    nt: int
    mode: Mode
    fallback_count: int
    rejected_count: int
    max_accepted_residual_indicator: float
    statuses: np.ndarray


class MBPort:
    """Reusable CPU evaluator, preloading the complex Gamma grid.

    source_R=True reproduces MATLAB's domain rejection for eps>R. Default
    R=0.9*u_star. For a full active-domain study, set R between qp*u_star
    and u_star explicitly, and *disclose* this change.

    verified=False is essential: the MATLAB 'certificates' use finite
    sampled envelopes and do not include validated quadrature error.
    """
    def __init__(self, *, p:float, lam:float=1, n:int=3,T:float=60,nt:int=6000,
                 mode:Mode="paper",R:float|None=None, source_R:bool=True,
                 sigma:float|None=None,chunk:int=32):
        self.c=constants(p,lam)
        if mode not in ("matlab","paper"):
            raise ValueError("mode must be 'matlab' or 'paper'")
        self.mode=mode
        self.R=float(R) if R is not None else .9*self.c.ustar
        if not 0 < self.R < self.c.ustar:
            raise ValueError("R must lie strictly between 0 and ustar")
        self.source_R=bool(source_R)
        self.grid=mb_grid(p,n,T,nt,sigma)
        self.n=n;self.T=T;self.nt=nt;self.chunk=chunk

    def __call__(self, y:np.ndarray, *, tol:float|None=None,
                 fallback_steps:int=80, imag_tol:float=1e-6) -> Evaluation:
        """Evaluate prox. With tol=None, do *not* fallback or hide failures.

        With tol>0, reject candidates failing the posterior residual bound,
        bracket, or imaginary-part test; use bisection. This is a separate
        safeguarded variant, not the raw published MATLAB method.
        """
        yy=np.asarray(y,dtype=np.float64)
        if np.any(~np.isfinite(yy)):
            raise ValueError("nonfinite y")
        if tol is not None and not (0<tol<1):
            raise ValueError("tol must be in (0,1)")
        a=np.abs(yy)
        active=a>self.c.tau
        eps=np.zeros_like(a)
        eps[active]=self.c.lam*self.c.p * a[active]**(self.c.p-2)
        rejected=active & (eps>self.R)
        statuses=np.full(a.shape,"inactive",dtype="<U24")
        statuses[active]="candidate"
        statuses[rejected]="rejected_R"
        if self.source_R and np.any(rejected) and tol is None:
            raise ValueError(f"{int(rejected.sum())} active inputs have epsilon>R: published MATLAB rejects them")
        allowed=active & ~rejected if self.source_R else active
        x_complex=np.zeros_like(a, dtype=np.complex128)
        zs=np.full_like(a,np.nan)
        zm=np.zeros_like(a,dtype=np.complex128)
        if np.any(allowed):
            ee=eps[allowed]
            zz=series_normalized(ee,p=self.c.p,n=self.n)
            correction=complex_mb_correction(ee,self.grid,chunk=self.chunk)
            z_complex = zz+correction if self.mode=="matlab" else zz-correction
            zs[allowed]=zz
            zm[allowed]=correction
            x_complex[allowed] = np.sign(yy[allowed])*a[allowed]*z_complex
        x=x_complex.real.copy()
        fallback_count=0
        max_accepted=0.0
        if tol is not None:
            for_test=allowed
            w=np.zeros_like(a)
            w[for_test]=np.abs(x[for_test])/a[for_test]
            bracket=(w >= self.c.rho/np.maximum(a,self.c.tau)-2e-15)&(w<=1+2e-15)
            sign_ok=(yy*x>0)
            resid=np.full_like(a,np.inf)
            resid[for_test]=np.abs(w[for_test]-1+eps[for_test]*w[for_test]**(self.c.p-1))
            indicator=resid/(1-self.c.p/2)
            imag=np.zeros_like(a)
            imag[for_test]=np.abs(x_complex[for_test].imag)/a[for_test]
            accepted=for_test & bracket & sign_ok & np.isfinite(indicator) & (indicator<=tol) & (imag<=imag_tol)
            statuses[accepted]="accepted_residual"
            if np.any(accepted):max_accepted=float(np.max(indicator[accepted]))
            fail=active & ~accepted
            fallback_count=int(np.count_nonzero(fail))
            if fallback_count:
                x[fail]=positive_root_bisect(yy[fail],p=self.c.p,lam=self.c.lam,steps=fallback_steps)
                statuses[fail]="bisection_fallback"
        elif np.any(allowed):
            # 'raw' candidates are not certified, even when they look accurate.
            statuses[allowed]="raw_unverified"
        imax=float(np.max(np.abs(x_complex[allowed].imag)/a[allowed])) if np.any(allowed) else 0.
        return Evaluation(x=x, x_complex=x_complex,z_series=zs,z_mb=zm,
                          max_imag_normalized=imax,n=self.n,T=self.T,nt=self.nt,
                          mode=self.mode,fallback_count=fallback_count,
                          rejected_count=int(np.count_nonzero(rejected)),
                          max_accepted_residual_indicator=max_accepted,
                          statuses=statuses)


def matlab_empirical_series_indicator(*,p:float,N:int,R:float,samples:int=1000)->float:
    """Replicate MATLAB's sampled C2 *geometric tail* diagnostic, not a proof."""
    c=constants(p,1)
    if not 0<R<c.ustar:raise ValueError("R outside radius")
    ks=np.arange(1,max(samples,N+400)+1,dtype=np.float64)
    loga=gammaln((2-p)*ks-1)-gammaln((1-p)*ks)-gammaln(ks+1)
    finite=loga+ks*np.log(c.ustar)
    C2=1.1*np.exp(np.max(finite[np.isfinite(finite)]))
    r=R/c.ustar
    return float(C2 * r**(N+1)/(1-r))


def matlab_empirical_mb_indicator(*,p:float,N:int,R:float,T:float,
                                   samples:int=4000)->float:
    """Sampled Gamma envelope following released MATLAB logic: diagnostic only.

    Source Cnum derives from a finite t-grid, not a validated majorant.
    """
    c=constants(p,1)
    if not 0<R<c.ustar:raise ValueError("R outside radius")
    sigma=N+.5
    t=np.linspace(T,max(1000,5*T),samples)
    s=sigma+1j*t
    logH=loggamma(-s)+loggamma((2-p)*s-1)-loggamma((1-p)*s)
    A=(2-p)*math.log(2-p)-(1-p)*math.log1p(-p)
    logval=logH.real+np.pi*np.abs(t)+1.5*np.log(t)-A*sigma
    finite=logval[np.isfinite(logval)]
    logC=max(0.,float(np.max(finite))) if finite.size else 0.
    logbound=math.log(2/np.pi) + logC + sigma*math.log(R) - .5*math.log(T)
    return math.exp(logbound) if logbound<math.log(np.finfo(float).max) else math.inf

@dataclass(frozen=True)
class MATLABParameterChoice:
    """Source-style empirical parameter choices, NOT validated bounds."""
    N: int
    T: float
    R: float
    sigma: float
    series_estimate: float
    mb_estimate: float
    meets_empirical_series_target: bool
    meets_empirical_mb_target: bool


def choose_matlab_parameters(*,p:float,R:float,eta:float=1e-8,
                             maxN:int=200,maxT:float=50000.,
                             T0:float=50.,Tfactor:float=1.5,
                             envelope_samples:int=4000) -> MATLABParameterChoice:
    """Mirror released MATLAB's finite sampled-constant N/T search.

    MATLAB uses a sampled maximum in place of the unknown asymptotic Gamma
    majorant.  This is a *heuristic diagnostic*; even if the returned
    estimated errors are small, the quadrature and rounding errors are not
    certified, and the sampled envelope need not dominate its entire tail.
    """
    c=constants(p,1.)
    if not (0<R<c.ustar and 0<eta<1 and 1<=maxN<=2048 and 0<T0<=maxT
            and Tfactor>1 and envelope_samples>=32):
        raise ValueError("invalid MATLAB auto-select options")
    series_est=math.inf
    N=maxN
    for candidate in range(1,maxN+1):
        series_est=matlab_empirical_series_indicator(p=p,N=candidate,R=R)
        if series_est<=eta/2:
            N=candidate
            break
    T=T0
    while T<maxT:
        mb_est=matlab_empirical_mb_indicator(p=p,N=N,R=R,T=T,
                                            samples=envelope_samples)
        if math.isfinite(mb_est) and mb_est<=eta/2:
            break
        T*=Tfactor
    else:
        mb_est=matlab_empirical_mb_indicator(p=p,N=N,R=R,T=T,
                                            samples=envelope_samples)
    return MATLABParameterChoice(N=N,T=T,R=R,sigma=N+0.5,
              series_estimate=series_est,mb_estimate=mb_est,
              meets_empirical_series_target=bool(series_est<=eta/2),
              meets_empirical_mb_target=bool(mb_est<=eta/2))
