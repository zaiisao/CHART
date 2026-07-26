"""Fit the PHYSICAL bar-pointer prior p_physical on the TRAIN fold (labels only).
Design-time measurement for SPEC.md; no model is trained here."""
import sys, math, json
import numpy as np
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")
from audit_common import load_split, ideal_beatlinear_barphase, ideal_barphase, FPS
from vbpm.evaluate import _estimate_meter
TWO_PI = 2*math.pi
METERS=(2,3,4)

tr = load_split("train")
print("train songs", len(tr))
from collections import Counter
print("fold counts train:", Counter(s["fold"] for s in tr))
print("dataset counts train:", Counter(s["dataset"] for s in tr))
ev = load_split("eval"); print("eval folds:", Counter(s["fold"] for s in ev), "n=",len(ev))

def wrap(x): return (x+math.pi)%TWO_PI - math.pi

res_all=[]; dlt_all=[]; lt_by_m={m:[] for m in METERS}; negfrac=[]
for s in tr:
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
    d=wrap(np.diff(ph))                       # per-frame advance phidot_t
    negfrac.append(float(np.mean(d<0)))
    d=np.clip(d,1e-6,None)
    lt=np.log(d)
    # slow component: median filter over ~1 bar
    win=int(np.median(np.diff(s["downs"]))*FPS)//2*2+1
    win=max(11,min(win,401))
    k=np.ones(win)/win
    lts=np.convolve(np.unwrap(np.zeros_like(lt))+lt,k,mode="same")
    lts[:win]=lt[:win].mean(); lts[-win:]=lt[-win:].mean()
    # phase residual around physical advance exp(lt_slow)
    r=wrap(np.diff(ph)-np.exp(lts))
    res_all.append(r[win:-win] if len(r)>2*win+10 else r)
    dlt_all.append(np.diff(lts)[win:-win] if len(lts)>2*win+10 else np.diff(lts))
    ibi=float(np.median(np.diff(s["beats"])))
    lt_by_m[m].append(math.log(TWO_PI/max(m*ibi*FPS,1e-6)))

R=np.concatenate(res_all); D=np.concatenate(dlt_all)
def cauchy_mle(x, g0):
    g=g0
    for _ in range(200):
        w=1.0/(g*g+x*x)
        g=math.sqrt(len(x)/(2*np.sum(w)))
    return g
print("\n--- PHASE RESIDUAL around physical advance (rad/frame) ---")
print("n=%d  median|r|=%.5f  IQR=%.5f  std=%.5f  p95=%.5f" % (
    len(R), np.median(np.abs(R)), np.subtract(*np.percentile(R,[75,25])), R.std(), np.percentile(np.abs(R),95)))
g=cauchy_mle(R, np.median(np.abs(R))+1e-6)
print("Cauchy MLE gamma = %.6f  -> rho=exp(-gamma)= %.6f" % (g, math.exp(-g)))
print("half-IQR gamma   = %.6f  -> rho= %.6f" % (np.subtract(*np.percentile(R,[75,25]))/2, math.exp(-np.subtract(*np.percentile(R,[75,25]))/2)))
print("frac true increments negative (ideal phase): mean %.4f" % np.mean(negfrac))
print("\n--- log bar-advance INCREMENT (slow level) ---")
print("n=%d std=%.6f  half-IQR=%.6f  MAD=%.6f" % (len(D), D.std(), np.subtract(*np.percentile(D,[75,25]))/2, np.median(np.abs(D-np.median(D)))))
print("\n--- log bar-advance LEVEL per meter (train) ---")
for m in METERS:
    v=np.array(lt_by_m[m]); 
    if len(v): print("  m=%d n=%3d mean=%.4f sd=%.4f  min=%.3f max=%.3f  (exp(mean)=%.5f rad/frame)"%(m,len(v),v.mean(),v.std(),v.min(),v.max(),math.exp(v.mean())))
allv=np.concatenate([np.array(lt_by_m[m]) for m in METERS])
print("  pooled: mean=%.4f sd=%.4f  range=[%.3f, %.3f]"%(allv.mean(),allv.std(),allv.min(),allv.max()))
# monotonicity floor for a wrapped Cauchy step
print("\n--- frac_neg FLOOR of a wrapped-Cauchy step: (1/pi) atan(gamma/phidot) ---")
for gg in (0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001):
    for pd in (0.0626,):
        print("   gamma=%.4f (rho=%.4f) phidot=%.4f -> frac_neg=%.4f"%(gg,math.exp(-gg),pd,math.atan(gg/pd)/math.pi))
json.dump(dict(gamma_cauchy=g, rho_phys=math.exp(-g), sigma_dlt=float(D.std()),
               lt_mean={str(m):float(np.mean(lt_by_m[m])) for m in METERS},
               lt_sd={str(m):float(np.std(lt_by_m[m])) for m in METERS}),
          open("/home/sogang/jaehoon/VBPM_reintegration/vbpm_anchor/phys_params.json","w"), indent=1)
