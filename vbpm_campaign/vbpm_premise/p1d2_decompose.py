"""PREMISE 1(d) part 2: (i) leak-free shrinkage test, (ii) variance decomposition
separating ANNOTATION/MICROTIMING JITTER from genuine tempo drift, (iii) tail audit."""
import sys, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from common import load_labels, per_ds, fmt_ds
from scipy import stats

def logibi(s):
    ibi = np.diff(s["beats"]); ibi = ibi[(ibi>0.1)&(ibi<2.0)]
    return np.log(ibi) if len(ibi)>=24 else None

tr=[(s["dataset"],logibi(s)) for s in load_labels("train")]
ev=[(s["dataset"],logibi(s)) for s in load_labels("eval")]
tr=[(d,x) for d,x in tr if x is not None]; ev=[(d,x) for d,x in ev if x is not None]

# ---- (i) LEAK-FREE shrinkage: choose ONE a on train, apply to eval -----------------
def mse_a(data, a):
    num=den=0.0
    for d,x in data:
        n=len(x); run=np.cumsum(x)/np.arange(1,n+1)
        pred = a*x[7:-1] + (1-a)*run[7:-1]
        e = x[8:]-pred; num += float((e**2).sum()); den += len(e)
    return num/den
grid=np.round(np.arange(0.0,1.01,0.05),2)
tr_mse=[mse_a(tr,a) for a in grid]
a_star=float(grid[int(np.argmin(tr_mse))])
print("== (i) LEAK-FREE mean-reversion test (single shrinkage a fit on 147 TRAIN songs)")
print(f"   train-optimal a = {a_star}  (a=1 is the pure Gaussian random walk)")
print("   train MSE curve:", {float(a):round(m,5) for a,m in zip(grid,tr_mse) if a in (0.0,0.2,0.3,0.4,0.5,0.8,1.0)})
rows=[]
for d,x in ev:
    n=len(x); run=np.cumsum(x)/np.arange(1,n+1)
    e_rw=x[8:]-x[7:-1]; e_a=x[8:]-(a_star*x[7:-1]+(1-a_star)*run[7:-1])
    rows.append(dict(dataset=d, mse_rw=float((e_rw**2).mean()), mse_a=float((e_a**2).mean()),
                     gain=1-float((e_a**2).mean())/float((e_rw**2).mean())))
print("   HELD-OUT eval MSE, random walk :", fmt_ds(per_ds(rows,'mse_rw'),5))
print(f"   HELD-OUT eval MSE, a={a_star}      :", fmt_ds(per_ds(rows,'mse_a'),5))
print("   relative MSE reduction         :", fmt_ds(per_ds(rows,'gain'),3))
print(f"   songs improved by shrinkage: {sum(1 for r in rows if r['gain']>0)}/{len(rows)}")

# ---- (ii) variance decomposition:  D_q = Var(x_{k+q}-x_k) = 2*sigma_noise^2 + q*sigma_w^2
print("\n== (ii) VARIANCE DECOMPOSITION  D_q = c + q*sigma_w^2   (c = 2x jitter var)")
qs=np.arange(2,17)
rows2=[]
for d,x in tr+ev:
    if len(x) < 40: continue
    D=np.array([np.var(x[q:]-x[:-q]) for q in qs])
    A=np.vstack([qs,np.ones_like(qs)]).T
    coef,*_=np.linalg.lstsq(A,D,rcond=None)
    sw2=max(coef[0],1e-12); c=max(coef[1],1e-12)
    pred=A@coef; r2=1-((D-pred)**2).sum()/((D-D.mean())**2).sum()
    rows2.append(dict(dataset=d, sw=float(np.sqrt(sw2)), sjit=float(np.sqrt(c/2)),
                      r2=float(r2), sd_dx=float(np.std(np.diff(x))),
                      frac_true=float(sw2/np.var(np.diff(x)))))
print(f"   {len(rows2)} songs (train+eval)")
print("   sigma_w  (TRUE tempo RW innovation, per beat, log units):", fmt_ds(per_ds(rows2,'sw'),4))
print("   sigma_jitter (per-beat timing noise, log units)         :", fmt_ds(per_ds(rows2,'sjit'),4))
print("   raw sd(dlogIBI) for comparison                          :", fmt_ds(per_ds(rows2,'sd_dx'),4))
print("   fraction of increment variance that is TRUE drift       :", fmt_ds(per_ds(rows2,'frac_true'),3))
print("   R^2 of the LINEAR (random-walk) fit to D_q              :", fmt_ds(per_ds(rows2,'r2'),3))
print("   (R^2 near 1 => the residual drift IS a random walk once jitter is removed;")
print("    systematic concavity would mean OU/mean-reverting drift.)")
Dp=np.zeros(len(qs)); n=0
for d,x in tr+ev:
    if len(x)<40: continue
    Dp += np.array([np.var(x[q:]-x[:-q]) for q in qs]); n+=1
print("   pooled mean D_q:", {int(q): round(float(v/n),5) for q,v in zip(qs,Dp)})

# ---- (iii) tail audit --------------------------------------------------------------
print("\n== (iii) TAIL AUDIT of dlog-IBI")
alld=np.concatenate([np.diff(x) for _,x in tr])
for thr,lab in ((np.log(1.15),'15%'),(np.log(1.5),'50%'),(np.log(1.9),'90% (~beat drop/insert)')):
    print(f"   |dlogIBI| > {lab}: {float((np.abs(alld)>thr).mean()):.5f} ({int((np.abs(alld)>thr).sum())}/{len(alld)})")
ks=[float(stats.kurtosis(np.diff(x))) for _,x in tr]
print(f"   per-song excess kurtosis: median={np.median(ks):.2f} p25={np.percentile(ks,25):.2f} p75={np.percentile(ks,75):.2f} max={max(ks):.1f}")
trim=np.abs(alld)<np.log(1.15)
print(f"   pooled excess kurtosis after trimming |dlogIBI|>15%: {stats.kurtosis(alld[trim]):.2f} (was {stats.kurtosis(alld):.1f})")
