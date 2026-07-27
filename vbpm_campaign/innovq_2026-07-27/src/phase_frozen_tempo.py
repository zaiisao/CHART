"""STAGE 2b: FREEZE the tempo pathway (init level head), train phase alone. Removes the
tempo-vs-phase fight. Sweep: which params to freeze x phase-loss curriculum (short->long).
Readout: per-crop PHASE corr (target ~0.9) with tempo r held at stage-1's 0.78."""
import sys, math, time
import numpy as np, torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
from rollout_vec import rollout_vec
dev="cuda:0"; T=1500; MB=18
P.PHYS["gamma_phase"]=0.06; P.PHYS["dev_sigma"]=0.02
tr=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev)
ev=P.build_crops(P.load_songs("eval"), n_per_song=1,seed=1,crop=T,dev=dev)
Ntr=tr["h"].shape[0]; Nev=min(64,ev["h"].shape[0])
def readout(m):
    m.eval(); I=[];Tr=[];C=[]
    with torch.no_grad():
        for a in range(0,Nev,8):
            ro=rollout_vec(m,ev["h"][a:a+8],ev["b"][a:a+8],n_picard=3)
            I.append(torch.log(torch.exp(ro["lt"]).median(1).values).cpu().numpy())
            Tr.append(torch.log(torch.exp(ev["lt"][a:a+8]).median(1).values).cpu().numpy())
            C.append(float(torch.abs(torch.exp(1j*(ro["phi"]-ev["phi"][a:a+8])).mean(1)).mean()))
    m.train(); I=np.concatenate(I);Tr=np.concatenate(Tr)
    return float(np.corrcoef(I,Tr)[0,1]), float(np.mean(C))
def run(tag, curriculum, lr=1e-3, steps=3000, w_t=0.0):
    torch.manual_seed(0); rng=np.random.default_rng(0)
    m=IQ.InnovQ().to(dev)
    m.load_state_dict(torch.load("tempofix_stage1.pt",map_location=dev,weights_only=False)["model"],strict=False)
    m.train()
    opt=torch.optim.AdamW(m.parameters(),lr=lr)
    r0,c0=readout(m)
    for s in range(1,steps+1):
        # curriculum: phase loss on a growing prefix (short windows first -> drift doesn't dominate)
        W = T if not curriculum else int(min(T, 150*2**(4*s/steps)))
        idx=torch.tensor(rng.integers(0,Ntr,MB),device=dev,dtype=torch.long)
        ro=rollout_vec(m,tr["h"][idx],tr["b"][idx],n_picard=3)
        L=(1-torch.cos(ro["phi"][:,:W]-tr["phi"][idx][:,:W])).mean()*10.0
        if w_t>0: L=L+((ro["lt"]-tr["lt"][idx])**2).mean()*w_t
        opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
        if s%1000==0:
            r,c=readout(m)
            print(f"  [{tag}] s{s:4d} W={W:4d} L={float(L):.3f} tempo_r={r:+.3f} PHASE corr={c:.3f}",flush=True)
    r,c=readout(m)
    torch.save({"model":m.state_dict()},f"phase_{tag}.pt")
    return r,c
print(f"baseline (stage1): tempo_r/phase = {readout(IQ.InnovQ().to(dev).requires_grad_(False)) if False else ''}")
res={}
for tag,cur,wt in (("nocur_notempo",False,0.0),("cur_notempo",True,0.0),("cur_tempoanchor",True,50.0)):
    print(f"=== {tag} ===",flush=True)
    res[tag]=run(tag,cur,w_t=wt)
print("\nFINAL: tag -> (tempo_r, phase_corr)")
for k,v in res.items(): print(f"  {k:18s} tempo_r={v[0]:+.3f}  PHASE corr={v[1]:.3f}")
