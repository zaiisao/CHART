"""PREMISE 1(d) part 3: WHERE do the heavy tails live -- in the tempo innovation or in the
per-beat jitter?  Shape tests that cannot be explained by between-song scale heterogeneity."""
import sys, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from common import load_labels, per_ds, fmt_ds
from scipy import stats
from scipy.signal import medfilt

def logibi(s):
    ibi=np.diff(s["beats"]); ibi=ibi[(ibi>0.1)&(ibi<2.0)]
    return np.log(ibi) if len(ibi)>=48 else None
tr=[(s["dataset"],logibi(s)) for s in load_labels("train")]
ev=[(s["dataset"],logibi(s)) for s in load_labels("eval")]
tr=[(d,x) for d,x in tr if x is not None]; ev=[(d,x) for d,x in ev if x is not None]
print(f"songs with >=48 IBIs: train {len(tr)}  eval {len(ev)}")

def fit_score(fit_seg, sc_seg):
    """Fit Gaussian & Laplace & t on fit_seg, score on sc_seg. Returns nats/sample deltas."""
    mu=np.median(fit_seg)
    sd=max(np.std(fit_seg),1e-6); b=max(np.mean(np.abs(fit_seg-mu)),1e-6)
    g=stats.norm.logpdf(sc_seg, np.mean(fit_seg), sd).mean()
    l=stats.laplace.logpdf(sc_seg, mu, b).mean()
    from scipy.optimize import minimize
    f=lambda p:-stats.t.logpdf(fit_seg, np.exp(p[0])+1.01, p[1], np.exp(p[2])).mean()
    r=minimize(f,[np.log(3.),mu,np.log(sd/2)],method='Nelder-Mead',options=dict(maxiter=3000))
    nu=np.exp(r.x[0])+1.01
    t=stats.t.logpdf(sc_seg, nu, r.x[1], np.exp(r.x[2])).mean()
    return float(g), float(l), float(t), float(nu)

# ---- C. WITHIN-SONG holdout (first half fits scale AND shape, second half scores) ----
print("\n== C. WITHIN-SONG holdout on eval songs (no between-song scale confound)")
rows=[]
for d,x in ev:
    dx=np.diff(x); h=len(dx)//2
    g,l,t,nu=fit_score(dx[:h], dx[h:])
    rows.append(dict(dataset=d,g=g,l=l,t=t,l_g=l-g,t_g=t-g,nu=nu))
print("   Laplace - Gaussian:", fmt_ds(per_ds(rows,'l_g'),3))
print("   Student-t - Gauss :", fmt_ds(per_ds(rows,'t_g'),3))
print("   fitted nu         :", fmt_ds(per_ds(rows,'nu',np.median),2))
print(f"   songs Laplace>Gauss: {sum(1 for r in rows if r['l_g']>0)}/{len(rows)}; "
      f"Student-t>Gauss: {sum(1 for r in rows if r['t_g']>0)}/{len(rows)}")

# ---- D. per-song-MAD-normalised pooled shape test (scale nuisance removed) -----------
print("\n== D. per-song-scale-normalised, shape fit on TRAIN songs, scored on EVAL songs")
nz=lambda x:(lambda d:(d-np.median(d))/max(np.median(np.abs(d-np.median(d))),1e-9))(np.diff(x))
Z=np.concatenate([nz(x) for _,x in tr])
sd=np.std(Z); b=np.mean(np.abs(Z-np.median(Z)))
from scipy.optimize import minimize
f=lambda p:-stats.t.logpdf(Z,np.exp(p[0])+1.01,p[1],np.exp(p[2])).mean()
r=minimize(f,[np.log(3.),0.,np.log(1.)],method='Nelder-Mead',options=dict(maxiter=4000))
nu=float(np.exp(r.x[0])+1.01)
print(f"   train-fitted shapes: Gauss sd={sd:.3f}  Laplace b={b:.3f}  Student-t nu={nu:.2f}")
rows=[]
for d,x in ev:
    z=nz(x)
    g=stats.norm.logpdf(z,0,sd).mean(); l=stats.laplace.logpdf(z,np.median(Z),b).mean()
    t=stats.t.logpdf(z,nu,r.x[1],np.exp(r.x[2])).mean()
    rows.append(dict(dataset=d,l_g=float(l-g),t_g=float(t-g)))
print("   Laplace - Gaussian:", fmt_ds(per_ds(rows,'l_g'),3))
print("   Student-t - Gauss :", fmt_ds(per_ds(rows,'t_g'),3))
print(f"   songs Laplace>Gauss: {sum(1 for r in rows if r['l_g']>0)}/{len(rows)}; t>Gauss: {sum(1 for r in rows if r['t_g']>0)}/{len(rows)}")

# ---- E. split increment into DRIFT and JITTER; test the shape of each ---------------
print("\n== E. decomposed shape: median-filtered tempo DRIFT vs per-beat JITTER (W=9 beats)")
rowsJ=[]; rowsD=[]
for d,x in ev:
    s=medfilt(x, 9); e=x-s; ds=np.diff(s)
    e=e[4:-4]; ds=ds[4:-4]
    for arr,rows_ in ((e,rowsJ),(ds,rowsD)):
        h=len(arr)//2
        if h<20: continue
        g,l,t,nu=fit_score(arr[:h],arr[h:])
        rows_.append(dict(dataset=d,l_g=l-g,t_g=t-g,kurt=float(stats.kurtosis(arr)),sd=float(np.std(arr))))
print(f"   JITTER residual x - medfilt(x)   ({len(rowsJ)} songs)")
print("     excess kurtosis:", fmt_ds(per_ds(rowsJ,'kurt',np.median),2), " sd:", fmt_ds(per_ds(rowsJ,'sd',np.median),4))
print("     Laplace-Gauss  :", fmt_ds(per_ds(rowsJ,'l_g'),3), f" won {sum(1 for r in rowsJ if r['l_g']>0)}/{len(rowsJ)}")
print("     Student-Gauss  :", fmt_ds(per_ds(rowsJ,'t_g'),3), f" won {sum(1 for r in rowsJ if r['t_g']>0)}/{len(rowsJ)}")
print(f"   DRIFT increments d medfilt(x)    ({len(rowsD)} songs)")
print("     excess kurtosis:", fmt_ds(per_ds(rowsD,'kurt',np.median),2), " sd:", fmt_ds(per_ds(rowsD,'sd',np.median),4))
print("     Laplace-Gauss  :", fmt_ds(per_ds(rowsD,'l_g'),3), f" won {sum(1 for r in rowsD if r['l_g']>0)}/{len(rowsD)}")
print("     Student-Gauss  :", fmt_ds(per_ds(rowsD,'t_g'),3), f" won {sum(1 for r in rowsD if r['t_g']>0)}/{len(rowsD)}")
