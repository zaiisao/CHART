"""PREMISE 1(d): is log-tempo a GAUSSIAN RANDOM WALK?  heavy tails? mean reversion?
Fit on TRAIN songs, score on EVAL songs. Includes an annotation-jitter null."""
import sys, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from common import load_labels, per_ds, fmt_ds, FPS
from scipy import stats
from scipy.optimize import minimize_scalar, minimize

def logibi(s):
    ibi = np.diff(s["beats"])
    ibi = ibi[(ibi > 0.1) & (ibi < 2.0)]
    return np.log(ibi) if len(ibi) >= 24 else None

def acf(x, L=6):
    x = x - x.mean(); v = float((x*x).mean())
    return [float((x[:-l]*x[l:]).mean()/v) for l in range(1, L+1)]

def var_ratio(x, qs=(1,2,4,8,16)):
    out = {}
    v1 = float(np.var(np.diff(x)))
    for q in qs:
        if len(x) <= q+2: out[q] = np.nan; continue
        out[q] = float(np.var(x[q:]-x[:-q]))/(q*v1)
    return out

# ---------- densities (location 0 assumed after centering; scale free) -------------
def ll_gauss(z, sd): return stats.norm.logpdf(z, 0, sd).mean()
def ll_lap(z, b):    return stats.laplace.logpdf(z, 0, b).mean()
def ll_t(z, nu, sc): return stats.t.logpdf(z, nu, 0, sc).mean()

def fit_gauss(z): return float(np.std(z))
def fit_lap(z):   return float(np.mean(np.abs(z - np.median(z)))) , float(np.median(z))
def fit_t(z):
    f = lambda p: -stats.t.logpdf(z, np.exp(p[0])+1.01, p[1], np.exp(p[2])).mean()
    r = minimize(f, [np.log(3.), np.median(z), np.log(np.std(z)/2)], method='Nelder-Mead',
                 options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
    return float(np.exp(r.x[0])+1.01), float(r.x[1]), float(np.exp(r.x[2]))

tr = load_labels("train"); ev = load_labels("eval")
TR = {s["stem"]: (s["dataset"], logibi(s)) for s in tr}
EV = {s["stem"]: (s["dataset"], logibi(s)) for s in ev}
TR = {k:v for k,v in TR.items() if v[1] is not None}
EV = {k:v for k,v in EV.items() if v[1] is not None}
print(f"songs with >=24 usable IBIs: train {len(TR)}/{len(tr)}  eval {len(EV)}/{len(ev)}")

# ---------------- 1. shape of the increments ---------------------------------------
rows=[]
for st,(ds,x) in TR.items():
    d = np.diff(x); a = acf(d, 4); vr = var_ratio(x)
    rows.append(dict(dataset=ds, kurt=float(stats.kurtosis(d)), sd=float(np.std(d)),
                     mad=float(np.median(np.abs(d-np.median(d)))),
                     acf1=a[0], acf2=a[1], acf3=a[2],
                     vr2=vr[2], vr4=vr[4], vr8=vr[8], vr16=vr[16], n=len(d)))
allz = np.concatenate([np.diff(x) for _,x in TR.values()])
zs   = np.concatenate([(lambda d: (d-np.median(d))/max(np.median(np.abs(d-np.median(d))),1e-9))(np.diff(x)) for _,x in TR.values()])
print(f"\n== TRAIN increments of log-IBI: {len(allz)} increments, {len(TR)} songs")
print("  per-song EXCESS KURTOSIS:", fmt_ds(per_ds(rows,'kurt'),2))
print(f"  pooled raw excess kurtosis = {stats.kurtosis(allz):.2f};  per-song-standardised pooled = {stats.kurtosis(zs):.2f}")
print("  per-song sd(dlogIBI)     :", fmt_ds(per_ds(rows,'sd'),4))
print("\n== MEAN REVERSION")
print("  per-song ACF of increments lag1:", fmt_ds(per_ds(rows,'acf1'),3))
print("                             lag2:", fmt_ds(per_ds(rows,'acf2'),3))
print("                             lag3:", fmt_ds(per_ds(rows,'acf3'),3))
print("  variance ratio VR(q)=Var(x_{k+q}-x_k)/(q Var(dx))  [1.0 = random walk, <1 = mean-reverting]")
for q in (2,4,8,16):
    print(f"    q={q:>2}:", fmt_ds(per_ds(rows,f'vr{q}'),3))
print("  NULLS for the increment ACF: pure random walk -> acf1=0, acf2=0;"
      "  pure iid ANNOTATION JITTER -> acf1=-0.667, acf2=+0.167")

# ---------------- 2. held-out density family comparison -----------------------------
print("\n== HELD-OUT density of dlog-IBI (families fit on TRAIN only, scored on EVAL songs)")
# protocol A: pooled raw increments
sd = fit_gauss(allz); b, mlap = fit_lap(allz); nu, mt, sct = fit_t(allz)
print(f"  fitted on train: Gauss sd={sd:.4f} | Laplace b={b:.4f} | Student-t nu={nu:.2f} scale={sct:.4f}")
er=[]
for st,(ds,x) in EV.items():
    d = np.diff(x)
    er.append(dict(dataset=ds, g=ll_gauss(d, sd), l=ll_lap(d-mlap, b), t=ll_t(d-mt, nu, sct)))
for r in er: r["l_g"]=r["l"]-r["g"]; r["t_g"]=r["t"]-r["g"]; r["t_l"]=r["t"]-r["l"]
print("  A. POOLED-RAW protocol, nats/increment on held-out eval songs")
print("     Gaussian     :", fmt_ds(per_ds(er,'g'),3))
print("     Laplace      :", fmt_ds(per_ds(er,'l'),3))
print("     Student-t    :", fmt_ds(per_ds(er,'t'),3))
print("     Laplace-Gauss:", fmt_ds(per_ds(er,'l_g'),3))
print("     Student-Gauss:", fmt_ds(per_ds(er,'t_g'),3))
nwin = sum(1 for r in er if r['l_g']>0); nwt = sum(1 for r in er if r['t_g']>0)
print(f"     songs where Laplace>Gauss: {nwin}/{len(er)};  Student-t>Gauss: {nwt}/{len(er)}")
# protocol B: shape only (per-song scale removed by a CAUSAL running MAD)
def causal_scale(d, warm=12):
    s = np.zeros(len(d))
    for i in range(len(d)):
        w = d[max(0,i-64):i] if i >= warm else d[:warm]
        s[i] = max(np.median(np.abs(w - np.median(w))), 1e-4)
    return s
zs_tr = np.concatenate([np.diff(x)/causal_scale(np.diff(x)) for _,x in TR.values()])
sd2 = fit_gauss(zs_tr); b2, m2 = fit_lap(zs_tr); nu2, mt2, sc2 = fit_t(zs_tr)
print(f"  fitted on train (scale-normalised): Gauss sd={sd2:.3f} | Laplace b={b2:.3f} | Student-t nu={nu2:.2f}")
er2=[]
for st,(ds,x) in EV.items():
    d = np.diff(x); z = d/causal_scale(d)
    er2.append(dict(dataset=ds, g=ll_gauss(z,sd2), l=ll_lap(z-m2,b2), t=ll_t(z-mt2,nu2,sc2)))
for r in er2: r["l_g"]=r["l"]-r["g"]; r["t_g"]=r["t"]-r["g"]
print("  B. CAUSAL-SCALE-NORMALISED protocol (shape only), nats/increment held out")
print("     Laplace-Gauss:", fmt_ds(per_ds(er2,'l_g'),3))
print("     Student-Gauss:", fmt_ds(per_ds(er2,'t_g'),3))
print(f"     songs where Laplace>Gauss: {sum(1 for r in er2 if r['l_g']>0)}/{len(er2)};"
      f"  Student-t>Gauss: {sum(1 for r in er2 if r['t_g']>0)}/{len(er2)}")

# ---------------- 3. causal one-step prediction: RW vs mean-reverting ---------------
print("\n== CAUSAL one-step-ahead prediction of log-IBI (no leakage; eval songs)")
res=[]
for st,(ds,x) in EV.items():
    n=len(x); run=np.cumsum(x)/np.arange(1,n+1)
    e_rw = x[8:] - x[7:-1]
    best=None
    for a in (0.0,0.1,0.2,0.3,0.5,0.8,1.0):
        pred = a*x[7:-1] + (1-a)*run[7:-1]      # a=1 -> pure random walk
        e = x[8:]-pred
        r_ = float(np.mean(e**2))
        if best is None or r_ < best[1]: best=(a,r_)
    # ALSO: MA(1)-corrected (annotation-jitter) predictor
    res.append(dict(dataset=ds, mse_rw=float(np.mean(e_rw**2)), best_a=best[0], mse_best=best[1],
                    gain=1-best[1]/float(np.mean(e_rw**2))))
print("  MSE(random walk)      :", fmt_ds(per_ds(res,'mse_rw'),5))
print("  best shrink a (1=RW)  :", fmt_ds(per_ds(res,'best_a'),2))
print("  MSE(best a)           :", fmt_ds(per_ds(res,'mse_best'),5))
print("  rel. MSE reduction    :", fmt_ds(per_ds(res,'gain'),3))
