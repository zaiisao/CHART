"""Wiring check + does the tempogram pathway train the LEVEL end-to-end?
 (1) gradient reaches tg_embed from mu_l1
 (2) supervise mu_l1 on true median tempo -> MAE, compare 15.3% GRU baseline
 (3) phase corr through the certified rollout with the learned level"""
import sys, math, numpy as np, torch, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration", "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq", "/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0, p)
import pm_common as P, innovq as IQ
from innovq_tg import InnovQT
from rollout_vec import rollout_vec
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi
P.PHYS["gamma_phase"]=0.06; P.PHYS["dev_sigma"]=0.02
tr=P.build_crops(P.load_songs("train"),n_per_song=8,seed=0,crop=T,dev=dev)
ev=P.build_crops(P.load_songs("eval"),n_per_song=1,seed=1,crop=T,dev=dev)
m=InnovQT().to(dev)
# ---- (1) gradient reaches the tempogram pathway
ctx=m.encode_posterior(tr["h"][:4],tr["b"][:4])
init=m.init_head(torch.cat([ctx.mean(1),ctx[:,0]],-1))
init[:,m.K+3].sum().backward()
g=sum(float(p.grad.abs().sum()) for p in m.tg_embed.parameters() if p.grad is not None)
print(f"(1) grad|tg_embed| from mu_l1 = {g:.4e}   {'OK' if g>0 else 'DEAD PATHWAY'}")
m.zero_grad(set_to_none=True)
# ---- (2) supervise the level (this is stage-1 warm start, not the ELBO)
def lvl_target(d):
    out=[]
    for i in range(d["h"].shape[0]):
        idx=np.where(d["b"][i].cpu().numpy()>0.5)[0]
        out.append(math.log(TWO_PI/(4.0*np.median(np.diff(idx)))) if len(idx)>=8 else float("nan"))
    return torch.tensor(out,device=dev)
ytr=lvl_target(tr); yev=lvl_target(ev)
ktr=~torch.isnan(ytr); kev=~torch.isnan(yev)
opt=torch.optim.AdamW(list(m.tg_embed.parameters())+list(m.init_head.parameters())+
                      list(m.encoder.parameters() if hasattr(m,"encoder") else []),lr=1e-3,weight_decay=1e-3)
idx_tr=torch.where(ktr)[0]
def mu_l1_of(d,sl):
    c=m.encode_posterior(d["h"][sl],d["b"][sl])
    return m.init_head(torch.cat([c.mean(1),c[:,0]],-1))[:,m.K+3]+m.level_offset
best=9
for s in range(600):
    m.train(); opt.zero_grad()
    sel=idx_tr[torch.randperm(len(idx_tr),device=dev)[:24]]
    loss=(mu_l1_of(tr,sel)-ytr[sel]).abs().mean(); loss.backward(); opt.step()
    if s%100==0 or s==599:
        m.eval()
        with torch.no_grad():
            tm=(mu_l1_of(tr,idx_tr[:200])-ytr[idx_tr[:200]]).abs().mean().item()
            em=(mu_l1_of(ev,torch.where(kev)[0])-yev[kev]).abs().mean().item()
        best=min(best,em)
        print(f"    step {s:3d}  train level MAE {100*tm:5.2f}%  eval {100*em:5.2f}%")
print(f"(2) BEST eval level MAE {100*best:.2f}%   vs GRU pooled 15.3%   target 2%")
# ---- (3) phase corr through the certified rollout
m.eval(); n=min(64,ev["h"].shape[0])
with torch.no_grad(): ro=rollout_vec(m,ev["h"][:n],ev["b"][:n],n_picard=3)
tphi=ev["phi"][:n]
c=float(torch.abs(torch.exp(1j*(ro["phi"]-tphi)).mean(1)).mean())
mae=float((ro["lt"]-ev["lt"][:n]).abs().mean())
print(f"(3) rollout phase corr {c:.3f} | rollout tempo MAE {100*mae:.1f}%   (GRU model was corr 0.270 / 15.3%)")
torch.save({"model":m.state_dict()},"tg_stage1.pt")
