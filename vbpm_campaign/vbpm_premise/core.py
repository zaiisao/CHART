"""PREMISE-2 core: discretized-likelihood transition models on the bar-pointer tempo state.

LATENT: u_k = log(phidot_k), phidot = bar advance rad/frame implied by beat interval I_k
        (phidot = 2*pi / (m * I_k * FPS)).  Beat times are on a 1-ms grid, so the observed
        I_k is an INTEGER number of ms; every model is scored by the probability MASS it puts
        on that 1-ms bin.  This kills the ~20% exactly-zero-increment atom as a way to cheat.
"""
import sys, math, numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, laplace, t as student
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from data import build, FPS, TWO_PI

MS = 1e-3

def prep(D):
    """attach discretized-bin edges in u-space + prediction index arrays."""
    for d in D:
        b = np.round(d['beats']*1000.0)/1000.0     # snap to the 1-ms annotation grid
        d['beats'] = b
        I = np.diff(b)
        keep = I > 5*MS
        d['I'] = I
        A = math.log(TWO_PI/(d['meter']*FPS))
        d['u']  = A - np.log(I)
        d['uhi'] = A - np.log(np.maximum(I-0.5*MS, 1e-4))   # upper u edge of the 1-ms bin
        d['ulo'] = A - np.log(I+0.5*MS)
        d['valid'] = keep
    return D

def logmass(u_lo, u_hi, mu, s, fam='laplace', nu=3.0):
    """log P(u in [u_lo,u_hi]) for location mu, scale s."""
    s = np.maximum(s, 1e-5)
    zl, zh = (u_lo-mu)/s, (u_hi-mu)/s
    if fam=='gauss':   c = norm.cdf
    elif fam=='laplace': c = laplace.cdf
    else: c = lambda z: student.cdf(z, nu)
    p = c(zh)-c(zl)
    # tail-safe: when the bin is far out, fall back to density*width
    bad = p < 1e-300
    if np.any(bad):
        if fam=='gauss': ld = norm.logpdf(np.where(bad,zl,0.0))
        elif fam=='laplace': ld = laplace.logpdf(np.where(bad,zl,0.0))
        else: ld = student.logpdf(np.where(bad,zl,0.0), nu)
        out = np.empty_like(p)
        out[~bad] = np.log(p[~bad])
        w = np.log(np.maximum(u_hi-u_lo,1e-300)) - np.log(s)
        out[bad] = (ld + w)[bad] if np.ndim(w) else (ld+w)[bad]
        return out
    return np.log(p)

# ------------------------------------------------------------------ index sets
def pairs(D, part='all'):
    """returns list of (song, k) prediction targets: predict u_k from u_{k-1}, k>=1.
    part='first'/'second' = first/second half of each song's k-range."""
    out=[]
    for d in D:
        n = len(d['u'])
        ks = [k for k in range(1,n) if d['valid'][k] and d['valid'][k-1]]
        if len(ks)<4: continue
        h = len(ks)//2
        sel = ks if part=='all' else (ks[:h] if part=='first' else ks[h:])
        for k in sel: out.append((d,k))
    return out

def gather(P):
    d0=None
    u_prev = np.array([d['u'][k-1] for d,k in P])
    ulo    = np.array([d['ulo'][k] for d,k in P])
    uhi    = np.array([d['uhi'][k] for d,k in P])
    u      = np.array([d['u'][k]   for d,k in P])
    meter  = np.array([d['meter']  for d,k in P])
    bib    = np.array([d['bib'][k] if k < len(d['bib']) else -1 for d,k in P])
    ds     = np.array([d['dataset'] for d,k in P])
    stem   = np.array([d['stem'] for d,k in P])
    return dict(u_prev=u_prev, ulo=ulo, uhi=uhi, u=u, meter=meter, bib=bib,
                dataset=ds, stem=stem, n=len(u))

# ------------------------------------------------------------------ fixed laws
def fit_rw(G, fam='laplace', nu=3.0, ou=False):
    """random walk mu = u_prev + c  (ou=True: mu = a*u_prev + b)."""
    up, lo, hi = G['u_prev'], G['ulo'], G['uhi']
    def nll(th):
        if ou: a,b,ls = th; mu = a*up + b
        else:  c,ls = th;   mu = up + c
        return -logmass(lo, hi, mu, math.exp(ls), fam, nu).mean()
    x0 = [1.0,0.0,math.log(0.03)] if ou else [0.0, math.log(0.03)]
    r = minimize(nll, x0, method='Nelder-Mead',
                 options=dict(maxiter=4000, xatol=1e-7, fatol=1e-9))
    return dict(th=r.x, ou=ou, fam=fam, nu=nu)

def score_rw(M, G):
    up, lo, hi = G['u_prev'], G['ulo'], G['uhi']
    if M['ou']: a,b,ls = M['th']; mu = a*up + b
    else:       c,ls = M['th'];   mu = up + c
    return logmass(lo, hi, mu, math.exp(ls), M['fam'], M['nu'])
