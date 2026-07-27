"""Trace the 1.1e-3 rad residual between rollout_vec and the loop, component by component."""
import sys, math, torch, torch.nn.functional as F, glob
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
from innovq import TWO_PI
from rollout_vec import rollout_vec
dev="cuda:0"; torch.manual_seed(0)
D=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=1500,dev=dev)
m=IQ.InnovQ().to(dev)
ck=glob.glob("innovq_pf_sm101_s0.pt")
if ck: m.load_state_dict(torch.load(ck[0],map_location=dev,weights_only=False).get("model"),strict=False)
m.eval(); h,b=D["h"][:3],D["b"][:3]
def dmax(a,r): return float(torch.abs(torch.angle(torch.exp(1j*(a-r)))).max())
# (1) ZERO-INNOVATION: pure prior recursion. Must be EXACT (no head involved).
ref0=IQ.rollout(m,h,b,sample=False,zero_innov=True)
with torch.no_grad():
    ctx=m.encode_posterior(h,b); K=m.K
    init=m.init_head(torch.cat([ctx.mean(1),ctx[:,0]],-1))
    mu_phi1=torch.atan2(init[:,K+1],init[:,K])%TWO_PI; mu_l1=init[:,K+3]+m.level_offset
    T=h.shape[1]; lt=mu_l1.unsqueeze(1).expand(-1,T)
    steps=torch.exp(lt.clamp(-12.,6.))
    phi0=(mu_phi1.unsqueeze(1)+torch.cumsum(F.pad(steps[:,:-1],(1,0)),1))%TWO_PI
print(f"(1) zero-innovation cumsum vs loop : max|dphi| = {dmax(phi0,ref0['phi']):.3e} rad   <- must be ~1e-7")
# (2) full vec vs loop, and where the error lives in TIME
ref=IQ.rollout(m,h,b,sample=False); v=rollout_vec(m,h,b,n_picard=3)
e=torch.abs(torch.angle(torch.exp(1j*(v["phi"]-ref["phi"]))))
print(f"(2) full: max {float(e.max()):.3e} | at t=0 {float(e[:,0].max()):.3e} | t=1 {float(e[:,1].max()):.3e} | t=100 {float(e[:,100].max()):.3e} | t=-1 {float(e[:,-1].max()):.3e}")
print(f"    error grows with t? first-half max {float(e[:,:750].max()):.3e} second-half {float(e[:,750:].max()):.3e}")
# (3) is it the meter path? compare meter trajectories
with torch.no_grad():
    mv=F.softmax(m.init_head(torch.cat([ctx.mean(1),ctx[:,0]],-1))[:,:K]/0.3,-1)
    mref=ref["Z"][...,3:]
    dm=float((mref-mv.unsqueeze(1)).abs().max())
print(f"(3) meter: loop-vs-constant(meter0) max |diff| = {dm:.3e}   <- if ~0, meter is NOT the culprit")
# (4) feed the loop the vec trajectory's innovations: does the loop reproduce vec?
print(f"(4) lt agreement: max|dlt| = {float((v['lt']-ref['lt']).abs().max()):.3e}")
