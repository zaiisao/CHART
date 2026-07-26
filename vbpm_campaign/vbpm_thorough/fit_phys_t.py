"""Fit the STUDENT-T physical law to the SAME slow-level increments D that
vbpm_anchor/fit_phys.py fit its Gaussian sigma_dlt to (train fold, labels only).
Writes vbpm_thorough/phys_t_params.json  (nu, scale, plus loglik comparison)."""
import sys, math, json
import numpy as np
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")
from audit_common import load_split, ideal_beatlinear_barphase, FPS
from vbpm.evaluate import _estimate_meter
from scipy import stats
TWO_PI = 2*math.pi
METERS=(2,3,4)

def wrap(x): return (x+math.pi)%TWO_PI - math.pi

def collect(split):
    songs = load_split(split)
    dlt_all=[]
    for s in songs:
        if len(s["downs"])<3 or len(s["beats"])<8: continue
        m=_estimate_meter(s["beats"],s["downs"])
        if m not in METERS: continue
        T=s["T"]
        ph=ideal_beatlinear_barphase(s["beats"],s["downs"],T,FPS)
        if ph is None: continue
        t=(np.arange(T)+0.5)/FPS
        ins=(t>=s["downs"][0])&(t<s["downs"][-1])
        ph=ph[ins]
        if len(ph)<200: continue
        d=wrap(np.diff(ph))
        d=np.clip(d,1e-6,None)
        lt=np.log(d)
        win=int(np.median(np.diff(s["downs"]))*FPS)//2*2+1
        win=max(11,min(win,401))
        k=np.ones(win)/win
        lts=np.convolve(lt,k,mode="same")
        lts[:win]=lt[:win].mean(); lts[-win:]=lt[-win:].mean()
        dlt_all.append(np.diff(lts)[win:-win] if len(lts)>2*win+10 else np.diff(lts))
    return np.concatenate(dlt_all)

D = collect("train")
print(f"train D: n={len(D)} std={D.std():.6g} kurtosis={stats.kurtosis(D):.1f}")

# Gaussian (what fit_phys anchored to)
s_g = D.std()
ll_g = stats.norm.logpdf(D, 0.0, s_g).mean()
# Student-t, loc pinned to 0 (symmetric physical random walk)
nu, loc, sc = stats.t.fit(D, floc=0.0)
ll_t = stats.t.logpdf(D, nu, 0.0, sc).mean()
# Laplace for reference
b = np.abs(D).mean()
ll_l = stats.laplace.logpdf(D, 0.0, b).mean()
print(f"Gaussian  sigma={s_g:.6g}            train ll/frame = {ll_g:.4f}")
print(f"Student-t nu={nu:.3f} scale={sc:.6g}  train ll/frame = {ll_t:.4f}")
print(f"Laplace   b={b:.6g}                train ll/frame = {ll_l:.4f}")

# held-out check on the eval fold (report only; params from train)
De = collect("eval")
print(f"eval D: n={len(De)} kurtosis={stats.kurtosis(De):.1f}")
print(f"held-out ll/frame: gauss {stats.norm.logpdf(De,0,s_g).mean():.4f}  "
      f"student {stats.t.logpdf(De,nu,0,sc).mean():.4f}  "
      f"laplace {stats.laplace.logpdf(De,0,b).mean():.4f}")

json.dump(dict(nu=float(nu), scale=float(sc), sigma_gauss=float(s_g),
               ll_train=dict(gauss=float(ll_g), student=float(ll_t), laplace=float(ll_l)),
               ll_eval=dict(gauss=float(stats.norm.logpdf(De,0,s_g).mean()),
                            student=float(stats.t.logpdf(De,nu,0,sc).mean()),
                            laplace=float(stats.laplace.logpdf(De,0,b).mean())),
               kurtosis_train=float(stats.kurtosis(D)), n_train=int(len(D))),
          open("/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough/phys_t_params.json","w"), indent=1)
print("WROTE phys_t_params.json")
