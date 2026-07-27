"""EXPERIMENT: make the tempo head actually estimate. Start from the placed ckpt (phase
machinery already good: corr 0.983 given true tempo). Train at 30s crops with a real
effective batch (grad accumulation) and retuned LR. READOUT = corr(inferred tempo, true)
across crops -- currently +0.09, slope +0.003, sd 0.009 vs true 0.304.
Stage 1: supervise tempo only (fast, direct).  Stage 2: full placement.  Stage 3: ELBO handover."""
import sys, math, glob, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
from rollout_vec import rollout_vec
dev="cuda:0"; torch.manual_seed(0); rng=np.random.default_rng(0); TWO_PI=2*math.pi
T=1500; ACC=1; MB=18         # vectorized: full batch in one pass
P.PHYS["gamma_phase"]=0.06; P.PHYS["dev_sigma"]=0.02
tr=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev)
ev=P.build_crops(P.load_songs("eval"), n_per_song=1,seed=1,crop=T,dev=dev)
Ntr=tr["h"].shape[0]; Nev=min(60,ev["h"].shape[0])
m=IQ.InnovQ().to(dev)
ck=glob.glob("innovq_pf_sm101_s0.pt")
if ck:
    m.load_state_dict(torch.load(ck[0],map_location=dev,weights_only=False).get("model"),strict=False)
    print(f"loaded {ck[0]}")
m.train()

@torch.no_grad()
def tempo_readout():
    m.eval(); I=[];Tr=[]
    for a in range(0,Nev,8):
        ro=rollout_vec(m,ev["h"][a:a+8],ev["b"][a:a+8],n_picard=3)
        I.append(torch.log(torch.exp(ro["lt"]).median(1).values).cpu().numpy())
        Tr.append(torch.log(torch.exp(ev["lt"][a:a+8]).median(1).values).cpu().numpy())
    m.train()
    I=np.concatenate(I); Tr=np.concatenate(Tr)
    r=float(np.corrcoef(I,Tr)[0,1]); sl=float(np.polyfit(Tr,I,1)[0])
    return r, sl, float(I.std()), float(np.abs(I-Tr).mean())

r,sl,sd,mae=tempo_readout()
print(f"START  corr(inf,true)={r:+.3f} slope={sl:+.3f} sd={sd:.3f} MAE={mae:.3f}  (true sd 0.304)",flush=True)

# ---------- STAGE 1: direct tempo supervision (the head only sees audio; target = true log-tempo) ----------
opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
t0=time.time()
for s in range(1,2001):
    opt.zero_grad()
    for _ in range(ACC):
        idx=torch.tensor(rng.integers(0,Ntr,MB),device=dev,dtype=torch.long)
        ro=rollout_vec(m,tr["h"][idx],tr["b"][idx],n_picard=3)
        L=((ro["lt"]-tr["lt"][idx])**2).mean()/ACC
        L.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
    if s%200==0:
        r,sl,sd,mae=tempo_readout()
        print(f"[TEMPO] s{s:3d} corr={r:+.3f} slope={sl:+.3f} sd={sd:.3f} MAE={mae:.3f} | {s/(time.time()-t0):.2f} it/s",flush=True)
torch.save({"model":m.state_dict()},"tempofix_stage1.pt")
# phase corr with the now-trained tempo
with torch.no_grad():
    m.eval(); cs=[]
    for a in range(0,Nev,8):
        ro=rollout_vec(m,ev["h"][a:a+8],ev["b"][a:a+8],n_picard=3)
        cs.append(float(torch.abs(torch.exp(1j*(ro["phi"]-ev["phi"][a:a+8])).mean(1)).mean()))
    m.train()
print(f"STAGE1 DONE: per-crop phase corr = {np.mean(cs):.3f}",flush=True)
