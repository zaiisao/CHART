"""STAGE 2: with tempo unpinned, RE-PLACE phase against the moving clock, jointly.
Loss = w_t*tempo MSE + phase circular loss (both supervised, both vectorized).
Readouts: corr(tempo), slope, and per-crop PHASE corr -- the number that must reach ~0.9."""
import sys, math, glob, time
import numpy as np, torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
from rollout_vec import rollout_vec
dev="cuda:0"; torch.manual_seed(0); rng=np.random.default_rng(0)
T=1500; MB=18
P.PHYS["gamma_phase"]=0.06; P.PHYS["dev_sigma"]=0.02
tr=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev)
ev=P.build_crops(P.load_songs("eval"), n_per_song=1,seed=1,crop=T,dev=dev)
Ntr=tr["h"].shape[0]; Nev=min(64,ev["h"].shape[0])
m=IQ.InnovQ().to(dev)
m.load_state_dict(torch.load("tempofix_stage1.pt",map_location=dev,weights_only=False)["model"],strict=False)
print("loaded tempofix_stage1.pt (tempo r=0.78)")
@torch.no_grad()
def readout():
    m.eval(); I=[];Tr=[];C=[]
    for a in range(0,Nev,8):
        ro=rollout_vec(m,ev["h"][a:a+8],ev["b"][a:a+8],n_picard=3)
        I.append(torch.log(torch.exp(ro["lt"]).median(1).values).cpu().numpy())
        Tr.append(torch.log(torch.exp(ev["lt"][a:a+8]).median(1).values).cpu().numpy())
        C.append(float(torch.abs(torch.exp(1j*(ro["phi"]-ev["phi"][a:a+8])).mean(1)).mean()))
    m.train(); I=np.concatenate(I);Tr=np.concatenate(Tr)
    return float(np.corrcoef(I,Tr)[0,1]), float(np.polyfit(Tr,I,1)[0]), float(np.mean(C))
r,sl,c=readout(); print(f"START tempo_r={r:+.3f} slope={sl:+.3f} | PHASE corr={c:.3f}",flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
t0=time.time()
for s in range(1,4001):
    idx=torch.tensor(rng.integers(0,Ntr,MB),device=dev,dtype=torch.long)
    ro=rollout_vec(m,tr["h"][idx],tr["b"][idx],n_picard=3)
    L_t=((ro["lt"]-tr["lt"][idx])**2).mean()*300.0
    L_p=(1-torch.cos(ro["phi"]-tr["phi"][idx])).mean()*10.0
    L=L_t+L_p
    opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
    if s%500==0:
        r,sl,c=readout()
        print(f"[JOINT] s{s:4d} L_t={float(L_t):.3f} L_p={float(L_p):.3f} | tempo_r={r:+.3f} slope={sl:+.3f} | PHASE corr={c:.3f} | {s/(time.time()-t0):.0f} it/s",flush=True)
torch.save({"model":m.state_dict()},"tempofix_stage2.pt")
print("STAGE2 DONE",flush=True)
