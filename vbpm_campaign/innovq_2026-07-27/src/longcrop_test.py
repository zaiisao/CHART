"""30-SECOND CROPS (1500 frames @50fps). Same recipe that died 10x at 256 frames:
placement (PF-teacher supervision) -> handover to full-beta ELBO. Does corr survive now?
Batch dropped to fit memory. Everything else identical."""
import sys, math, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
dev="cuda:0"; torch.manual_seed(0); rng=np.random.default_rng(0); TWO_PI=2*math.pi
T=1500; BATCH=3
P.PHYS["gamma_phase"]=0.06; P.PHYS["dev_sigma"]=0.02
tr=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev)
ev=P.build_crops(P.load_songs("eval"), n_per_song=1,seed=1,crop=T,dev=dev)
N=tr["h"].shape[0]; print(f"train {N} crops x {T} frames ({T/50:.0f}s) | eval {ev['h'].shape[0]}",flush=True)
m=IQ.InnovQ().to(dev); dec,hdec=P.new_decoders(dev)
def corr_pooled(p,ref): return float(torch.abs(torch.exp(1j*(p-ref)).mean()))
def corr_percrop(p,ref): return float(torch.abs(torch.exp(1j*(p-ref)).mean(1)).mean())
@torch.no_grad()
def probe():
    m.eval(); cs=[];cp=[]
    for a in range(0,ev["h"].shape[0],3):
        h=ev["h"][a:a+3]; b=ev["b"][a:a+3]
        ro=IQ.rollout(m,h,b,sample=False)
        cs.append(corr_pooled(ro["phi"],ev["phi"][a:a+3])); cp.append(corr_percrop(ro["phi"],ev["phi"][a:a+3]))
    m.train(); return float(np.mean(cs)),float(np.mean(cp))
# ---- stage 1: placement (teacher = TRUE latents; the m2 protocol, label variant) ----
opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
TEA={"phi":tr["phi"],"lt":tr["lt"]}
t0=time.time()
for s in range(1,301):
    idx=torch.tensor(rng.integers(0,N,BATCH),device=dev,dtype=torch.long)
    L,info=IQ.placement_loss(m,tr,TEA,idx,w_roll=1.0)
    opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
    if s%100==0:
        cs,cp=probe(); print(f"[PRE ] s{s:3d} L={float(L):7.3f} ev pooled={cs:.3f} percrop={cp:.3f} | {s/(time.time()-t0):.2f} it/s",flush=True)
cs,cp=probe(); print(f"HANDOVER: ev pooled={cs:.3f} percrop={cp:.3f}",flush=True)
# ---- stage 2: ELBO handover ----
opt=torch.optim.AdamW(list(m.parameters())+[p for q in (dec,hdec) for p in q.parameters()],lr=3e-4)
for s in range(1,401):
    beta=min(1.0,s/50)
    idx=torch.tensor(rng.integers(0,N,BATCH),device=dev,dtype=torch.long)
    loss,info,ro=IQ.elbo_innovq(m,tr,dec,hdec,idx=idx,beta=beta,sample=True)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(m.parameters())+[p for q in (dec,hdec) for p in q.parameters()],5.0); opt.step()
    if s%50==0:
        cs,cp=probe()
        print(f"[ELBO] s{s:3d} b={beta:.2f} L={float(loss):8.1f} ev pooled={cs:.3f} percrop={cp:.3f} klph={info.get('kl_phase',0):.1f}",flush=True)
print("DONE_LONGCROP",flush=True)
