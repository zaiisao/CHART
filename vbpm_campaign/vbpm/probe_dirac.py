"""CEILING TEST: feed VBPM a DIRAC h (impulses at true beat/downbeat frames) instead of MERT.
The observation is now perfectly informative. Question: can the VBPM machinery reach high
FREE-RUN beat-F when the input literally IS the answer?
  - if NO  -> the failure is the generative/inference machinery itself (not perception)
  - if YES -> machinery works given clean evidence; failure is in using noisy evidence
NOTE (honesty): the prior p(z|h) reads h, so this is an UPPER BOUND / sanity ceiling, not proof
of inference -- a good score here only means the prior can exploit a marked input.
Same fold-honest split + free-run eval as train_mert.py.
"""
import sys, glob, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run
from vbpm.evaluate import (beats_from_barphase, downbeats_from_barphase, beats_from_activation,
                           metronome, f_measure, _estimate_meter)
CACHE="/disk1/jaehoon/vbpm_mert_cache"; dev="cuda:0"; fps=50.0
H_DIM=8   # small dirac channel bank

def load(split, cap=None):
    out=[]
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d=np.load(f,allow_pickle=True)
        out.append(dict(T=int(d["feats"].shape[1]),beats=np.asarray(d["beats"],float),
                        downs=np.asarray(d["downs"],float)))
    return out[:cap] if cap else out
train=load("train"); ev=load("eval",30)
print(f"dirac ceiling: train {len(train)} eval {len(ev)}",flush=True)

def dirac_h(beats,downs,start,n):
    """h[:,0]=beat impulses, h[:,1]=downbeat impulses, rest zeros (+tiny noise)."""
    h=np.random.randn(n,H_DIM).astype(np.float32)*0.01
    for t in beats:
        i=int(round(t*fps))-start
        if 0<=i<n: h[i,0]+=1.0
    for t in downs:
        i=int(round(t*fps))-start
        if 0<=i<n: h[i,1]+=1.0
    return h

def targets(beats,downs,start,n):
    b=np.zeros(n,np.float32); db=np.zeros(n,np.float32)
    for t in beats:
        i=int(round(t*fps))-start
        if 0<=i<n: b[i]=1.0
    for t in downs:
        i=int(round(t*fps))-start
        if 0<=i<n: db[i]=1.0
    return b,db

@torch.no_grad()
def ev_freerun(model,songs,max_frames=1600):
    model.eval(); acc={"beat_phase":[],"downbeat_phase":[],"decoder":[],"metronome":[]}
    for s in songs:
        T=min(s["T"],max_frames)
        h=torch.from_numpy(dirac_h(s["beats"],s["downs"],0,T)).unsqueeze(0).to(dev)
        out=free_run(model,h)
        pm=out["phase_mu"][0,:T].cpu().numpy(); dec=out["decoder_prob"][0,:T].cpu().numpy()
        ref=s["beats"][s["beats"]<T/fps]; dref=s["downs"][s["downs"]<T/fps]
        if len(ref)<2: continue
        m=_estimate_meter(ref,dref)
        acc["beat_phase"].append(f_measure(ref,beats_from_barphase(pm,m,fps)))
        if len(dref)>=2: acc["downbeat_phase"].append(f_measure(dref,downbeats_from_barphase(pm,fps)))
        acc["decoder"].append(f_measure(ref,beats_from_activation(dec,fps)))
        acc["metronome"].append(f_measure(ref,metronome(T,fps)))
    model.train()
    return {k:(float(np.mean(v)) if v else float("nan")) for k,v in acc.items()}

STEPS,WARM,BS,FR=1200,600,16,256
torch.manual_seed(0); rng=np.random.default_rng(0)
model=BarPointerVAE(h_dim=H_DIM,hidden=128,num_meters=4).to(dev)   # latent_only=True
opt=torch.optim.AdamW(model.parameters(),lr=3e-4); t0=time.time(); best=-1
for step in range(1,STEPS+1):
    beta=min(1.0,step/WARM); temp=1.0+(0.3-1.0)*min(step/STEPS,1.0)
    hs,bs_,ds=[],[],[]
    for _ in range(BS):
        s=train[rng.integers(len(train))]
        if s["T"]<=FR: continue
        st=int(rng.integers(0,s["T"]-FR))
        hs.append(torch.from_numpy(dirac_h(s["beats"],s["downs"],st,FR)))
        b,d=targets(s["beats"],s["downs"],st,FR); bs_.append(torch.from_numpy(b)); ds.append(torch.from_numpy(d))
    h=torch.stack(hs).to(dev); b=torch.stack(bs_).to(dev); d=torch.stack(ds).to(dev)
    opt.zero_grad(); loss,info=strict_elbo(model,h,b,d,temperature=temp,beta=beta)
    if not torch.isfinite(loss): print("NaN@",step,flush=True); break
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
    if step%100==0:
        print(f"s{step:4d} b={beta:.2f} rec_b={info['recon_beat']:6.2f} kl(phi={info['kl_phase']:.2f} lv={info['kl_level']:.2f}) nu={info['tempo_dof']:.2f} | {step/(time.time()-t0):.1f} it/s",flush=True)
    if step%300==0 or step==STEPS:
        r=ev_freerun(model,ev)
        print(f"  [DIRAC EVAL s{step}] FREE-RUN beat_F={r['beat_phase']:.3f} db_F={r['downbeat_phase']:.3f} decoder_F={r['decoder']:.3f} metronome={r['metronome']:.3f}",flush=True)
        best=max(best,r["beat_phase"] if r["beat_phase"]==r["beat_phase"] else -1)
print(f"DONE dirac-ceiling best free-run beat_F={best:.3f}",flush=True)
