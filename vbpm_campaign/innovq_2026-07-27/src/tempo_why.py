"""Why does tempo estimation fail? Is the head (a) outputting a near-CONSTANT (corpus mean,
audio-blind) or (b) genuinely estimating but noisily? Decisive: correlation between inferred
and true tempo across crops, and the spread of each."""
import sys, math, glob
import numpy as np, torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
dev="cuda:0"; torch.manual_seed(0)
D=P.build_crops(P.load_songs("eval"),n_per_song=1,seed=1,crop=1500,dev=dev)
m=IQ.InnovQ().to(dev)
ck=glob.glob("innovq_pf_sm101_s0.pt")
if ck: m.load_state_dict(torch.load(ck[0],map_location=dev,weights_only=False).get("model"),strict=False)
m.eval()
n=min(60,D["h"].shape[0])
with torch.no_grad():
    ro=IQ.rollout(m,D["h"][:n],D["b"][:n],sample=False)
    inf=torch.log(torch.exp(ro["lt"]).median(1).values).cpu().numpy()   # inferred log-tempo
    tru=torch.log(torch.exp(D["lt"][:n]).median(1).values).cpu().numpy()
print(f"n={n} eval crops")
print(f"  TRUE  log-tempo: mean {tru.mean():+.3f}  sd {tru.std():.3f}  range [{tru.min():+.2f},{tru.max():+.2f}]")
print(f"  INFER log-tempo: mean {inf.mean():+.3f}  sd {inf.std():.3f}  range [{inf.min():+.2f},{inf.max():+.2f}]")
print(f"  sd ratio (infer/true): {inf.std()/tru.std():.3f}   <- ~0 means CONSTANT output (audio-blind)")
r=np.corrcoef(inf,tru)[0,1]
print(f"  corr(inferred, true) across crops: {r:+.3f}   <- ~0 means no estimation at all")
sl=np.polyfit(tru,inf,1)[0]
print(f"  regression slope inferred~true: {sl:+.3f}   (1.0 = perfect tracking, 0 = constant)")
print(f"  model's prior init_level_mu = {P.PHYS.get('init_level_mu')}, level_offset = {float(m.level_offset):+.3f}")
# what does a constant-at-corpus-mean predictor score?
const=np.full_like(inf,tru.mean())
print(f"\n  MAE of model      : {np.abs(inf-tru).mean():.3f} nats")
print(f"  MAE of const-mean : {np.abs(const-tru).mean():.3f} nats  <- if model ~= this, head learned nothing")
