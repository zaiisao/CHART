"""Q1: does the tg_pre target mean the same thing as train['lt']?  (the hardcoded 4.0)
   Q2: does the level init SURVIVE ELBO training?"""
import sys, math, torch, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P, innovq as IQ
from innovq_tg import InnovQT
from rollout_vec_s import rollout_vec_s
IQ.ROLLOUT_FN=rollout_vec_s   # use the fast path in the diagnostic too
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi
P.PHYS["gamma_phase"]=0.06; IQ.RHO_P=math.exp(-0.06)
tr=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev)
N=tr["b"].shape[0]; print(f"loaded {N} crops of {T}",flush=True)
tg=[]
for i in range(N):
    idx=torch.nonzero(tr["b"][i]>0.5).squeeze(-1)
    tg.append(math.log(TWO_PI/(4.0*float(torch.median(torch.diff(idx).float())))) if len(idx)>=8 else float("nan"))
tg=torch.tensor(tg,device=dev); ok=torch.nonzero(~torch.isnan(tg)).squeeze(-1)
true_lt=tr["lt"].mean(1)
d=tg[ok]-true_lt[ok]
print(f"Q1  tg_pre target vs train['lt']: mean diff {float(d.mean()):+.4f} nats  sd {float(d.std()):.4f}")
print(f"    ratio exp(mean) = {math.exp(float(d.mean())):.4f}  (1.000 => same quantity)")
print(f"    median |diff| = {100*float(d.abs().median()):.1f}% tempo error",flush=True)
m=InnovQT().to(dev); d0,h0=P.new_decoders(dev); dec,hdec=IQ.Cut(d0),IQ.Cut(h0)
def mae():
    with torch.no_grad():
        c=m.encode_posterior(tr["h"][ok[:150]],tr["b"][ok[:150]])
        mu=m.init_head(torch.cat([c.mean(1),c[:,0]],-1))[:,m.K+3]+m.level_offset
        return 100*float((mu-tg[ok[:150]]).abs().mean()), 100*float((mu-true_lt[ok[:150]]).abs().mean())
o1=torch.optim.AdamW(m.parameters(),lr=1e-3)
for s in range(400):
    o1.zero_grad(); sel=ok[torch.randperm(len(ok),device=dev)[:24]]
    c=m.encode_posterior(tr["h"][sel],tr["b"][sel])
    mu=m.init_head(torch.cat([c.mean(1),c[:,0]],-1))[:,m.K+3]+m.level_offset
    (mu-tg[sel]).abs().mean().backward(); o1.step()
a,b=mae(); print(f"\nQ2  after tg_pre:          {a:.2f}% vs target | {b:.2f}% vs train['lt']",flush=True)
o2=torch.optim.Adam(list(m.parameters())+list(d0.parameters())+list(h0.parameters()),lr=3e-4)
for s in range(200):
    o2.zero_grad()
    loss,_,_=IQ.elbo_innovq(m,tr,dec,hdec,idx=torch.randperm(N,device=dev)[:16],beta=1.0,recon="bce")
    loss.backward(); o2.step()
    if s in (24,74,199):
        a2,b2=mae(); print(f"    after {s+1:3d} ELBO steps: {a2:.2f}% vs target | {b2:.2f}% vs train['lt']",flush=True)
