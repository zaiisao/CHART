"""The two questions the puzzle turns on:
 Q1 does the tg_pre target MEAN the same thing as train["lt"]? (the hardcoded 4.0)
 Q2 does the 1.77% level init SURVIVE ELBO training, or is it un-learned like the decoder was?"""
import sys, math, numpy as np, torch, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P, innovq as IQ
from innovq_tg import InnovQT
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi
P.PHYS["gamma_phase"]=0.06; IQ.RHO_P=math.exp(-0.06)
tr=P.build_crops(P.load_songs("train"),n_per_song=2,seed=0,crop=T,dev=dev)
N=tr["b"].shape[0]
# ---- Q1: my tg_pre target vs the ground-truth lt the isolation harness uses
tgt=[]
for i in range(N):
    idx=torch.nonzero(tr["b"][i]>0.5).squeeze(-1)
    tgt.append(math.log(TWO_PI/(4.0*float(torch.median(torch.diff(idx).float())))) if len(idx)>=8 else float("nan"))
tgt=torch.tensor(tgt,device=dev); ok=~torch.isnan(tgt)
true_lt=tr["lt"].mean(1)                      # what rollout/isolation actually call log-tempo
d=(tgt[ok]-true_lt[ok])
print(f"Q1  tg_pre target vs train['lt']:  mean diff {float(d.mean()):+.4f}  sd {float(d.std()):.4f}")
print(f"    -> ratio exp(mean diff) = {math.exp(float(d.mean())):.4f}   (1.000 = same quantity)")
print(f"    |diff| median {float(d.abs().median()):.4f} nats = {100*float(d.abs().median()):.1f}% tempo error")
# ---- Q2: init the level head, then run ELBO and re-measure
m=InnovQT().to(dev); d0,h0=P.new_decoders(dev); dec,hdec=IQ.Cut(d0),IQ.Cut(h0)
def lvl_mae():
    with torch.no_grad():
        c=m.encode_posterior(tr["h"][ok][:200],tr["b"][ok][:200])
        mu=m.init_head(torch.cat([c.mean(1),c[:,0]],-1))[:,m.K+3]+m.level_offset
        return float((mu-tgt[ok][:200]).abs().mean()), float((mu-true_lt[ok][:200]).abs().mean())
opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
idx_ok=torch.nonzero(ok).squeeze(-1)
for s in range(800):
    opt.zero_grad(); sel=idx_ok[torch.randperm(len(idx_ok),device=dev)[:24]]
    c=m.encode_posterior(tr["h"][sel],tr["b"][sel])
    mu=m.init_head(torch.cat([c.mean(1),c[:,0]],-1))[:,m.K+3]+m.level_offset
    (mu-tgt[sel]).abs().mean().backward(); opt.step()
a,b=lvl_mae(); print(f"\nQ2  after tg_pre:            MAE vs target {100*a:.2f}%   vs train['lt'] {100*b:.2f}%")
allp=list(m.parameters())+list(d0.parameters())+list(h0.parameters())
opt2=torch.optim.Adam(allp,lr=3e-4)
for s in range(300):
    opt2.zero_grad()
    loss,info,_=IQ.elbo_innovq(m,tr,dec,hdec,idx=torch.randperm(N,device=dev)[:16],beta=1.0,recon="bce")
    loss.backward(); opt2.step()
    if s in (49,149,299):
        a2,b2=lvl_mae(); print(f"    after {s+1:3d} ELBO steps:   MAE vs target {100*a2:.2f}%   vs train['lt'] {100*b2:.2f}%")
