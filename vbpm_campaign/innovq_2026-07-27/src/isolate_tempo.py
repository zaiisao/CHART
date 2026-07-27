"""ISOLATION: everything deterministic (already true). Decompose phase corr into its two
inputs: start phase phi_1 and the tempo trajectory.
  A) true phi_1 + TRUE tempo        -> ceiling
  B) true phi_1 + MODEL tempo       -> cost of tempo error ALONE
  C) model phi_1 + TRUE tempo       -> cost of offset error ALONE
  D) model phi_1 + MODEL tempo      -> what we get
  E) requirement curve: corr vs injected tempo MAE at T=1500"""
import sys, math
import numpy as np, torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
from rollout_vec import rollout_vec
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi
P.PHYS["gamma_phase"]=0.06; P.PHYS["dev_sigma"]=0.02
ev=P.build_crops(P.load_songs("eval"),n_per_song=1,seed=1,crop=T,dev=dev)
n=min(64,ev["h"].shape[0])
m=IQ.InnovQ().to(dev)
m.load_state_dict(torch.load("tempofix_stage1.pt",map_location=dev,weights_only=False)["model"],strict=False)
m.eval()
with torch.no_grad():
    ro=rollout_vec(m,ev["h"][:n],ev["b"][:n],n_picard=3)
tphi=ev["phi"][:n]; tlt=ev["lt"][:n]
mphi1=ro["phi"][:,:1]; mlt=ro["lt"][:n]
def build(phi1, lt):
    steps=torch.exp(lt.clamp(-12,6))
    inc=torch.nn.functional.pad(steps[:,:-1],(1,0))
    return (phi1+torch.cumsum(inc,1))%TWO_PI
def corr(p): return float(torch.abs(torch.exp(1j*(p-tphi)).mean(1)).mean())
with torch.no_grad():
    A=corr(build(tphi[:,:1],tlt)); B=corr(build(tphi[:,:1],mlt))
    C=corr(build(mphi1,tlt));      Dd=corr(build(mphi1,mlt))
    mae=float((mlt-tlt).abs().mean())
print(f"deterministic path, T={T}, n={n} eval crops   (model tempo MAE = {mae:.3f} log-nats = {100*mae:.1f}%)")
print(f"  A true phi1 + TRUE tempo : corr {A:.3f}   <- ceiling")
print(f"  B true phi1 + MODEL tempo: corr {B:.3f}   <- cost of TEMPO error alone")
print(f"  C model phi1 + TRUE tempo: corr {C:.3f}   <- cost of OFFSET error alone")
print(f"  D model phi1 + MODEL tempo: corr {Dd:.3f}  <- actual")
print(f"\n(E) requirement: corr vs injected tempo error (true phi1, true tempo + bias)")
for pct in (0.5,1,2,5,10,15):
    with torch.no_grad():
        lt2=tlt+math.log(1+pct/100.0)
        print(f"    {pct:5.1f}% tempo error -> corr {corr(build(tphi[:,:1],lt2)):.3f}")
