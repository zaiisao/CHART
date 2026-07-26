"""MLE for concentrated circular densities on residuals (numerically safe for huge kappa)."""
import numpy as np
from scipy.special import ive, i0e
from scipy.optimize import minimize_scalar

def vm_logpdf(r, kappa, mu=0.0):
    return kappa*(np.cos(r-mu)-1.0) - np.log(2*np.pi*i0e(kappa))

def vm_fit(r):
    C, S = np.cos(r).mean(), np.sin(r).mean()
    mu = np.arctan2(S, C); R = np.hypot(C, S)
    R = min(R, 1-1e-12)
    f = lambda lk: -vm_logpdf(r, np.exp(lk), mu).mean()
    lo, hi = np.log(1e-3), np.log(1e12)
    res = minimize_scalar(f, bounds=(lo, hi), method='bounded',
                          options=dict(xatol=1e-8))
    return float(np.exp(res.x)), float(mu)

def wc_logpdf(r, rho, mu=0.0):
    rho = min(max(rho, 1e-9), 1-1e-12)
    return np.log(1-rho**2) - np.log(2*np.pi) - np.log1p(rho**2 - 2*rho*np.cos(r-mu))

def wc_fit(r):
    C, S = np.cos(r).mean(), np.sin(r).mean()
    mu = np.arctan2(S, C)
    f = lambda u: -wc_logpdf(r, 1-np.exp(u), mu).mean()   # u = log(1-rho)
    res = minimize_scalar(f, bounds=(np.log(1e-14), np.log(0.99)),
                          method='bounded', options=dict(xatol=1e-10))
    return float(1-np.exp(res.x)), float(mu)

def wc_gamma(rho):
    """wrapped-Cauchy scale gamma = -log(rho) (rad); ~ half-width for concentrated rho."""
    return -np.log(max(rho, 1e-300))
