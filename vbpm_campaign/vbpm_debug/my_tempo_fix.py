"""Independent check of the tempo-init/units fix, via MONKEYPATCH (shared vbpm/ untouched).
A/B: Dirac free-run beat_F with the tempo latent initialized at the WRONG scale (as built,
log_tempo~0 => 1.03 rad/frame) vs the PHYSICALLY CORRECT scale (log(2pi/(IBI*m*fps)) ~ -2.77).
"""
import sys, glob, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
import vbpm.model as M
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run
from vbpm.evaluate import beats_from_barphase, metronome, f_measure, _estimate_meter
CACHE="/disk1/jaehoon/vbpm_mert_cache"; fps=50.0; TWO_PI=2*math.pi; dev="cuda:0"
H=8

def load(split,cap=None):
    o=[]
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d=np.load(f,allow_pickle=True)
        o.append(dict(T=int(d["feats"].shape[1]),beats=np.asarray(d["beats"],float),downs=np.asarray(d["downs"],float)))
    return o[:cap] if cap else o
train=load("train"); ev=load("eval",20)

def dirac(beats,downs,start,n):
    h=np.random.randn(n,H).astype(np.float32)*0.01
    for t in beats:
        i=int(round(t*fps))-start
        if 0<=i<n: h[i,0]+=1.0
    for t in downs:
        i=int(round(t*fps))-start
        if 0<=i<n: h[i,1]+=1.0
    return h
def tg(beats,downs,start,n):
    b=np.zeros(n,np.float32); d=np.zeros(n,np.float32)
    for t in beats:
        i=int(round(t*fps))-start
        if 0<=i<n: b[i]=1.0
    for t in downs:
        i=int(round(t*fps))-start
        if 0<=i<n: d[i]=1.0
    return b,d

ORIG = BarPointerVAE.unpack
def make_patched(offset):
    def unpack(self, vec):
        ml,pm,pr,lmu,ls,dmu,ds = ORIG(self, vec)
        return ml, pm, pr, lmu + offset, ls, dmu, ds      # shift tempo LEVEL to physical scale
    return unpack

@torch.no_grad()
def evaluate(model):
    model.eval(); Fs=[];Fm=[]
    for s in ev:
        T=min(s["T"],1600)
        h=torch.from_numpy(dirac(s["beats"],s["downs"],0,T)).unsqueeze(0).to(dev)
        out=free_run(model,h); pm=out["phase_mu"][0,:T].cpu().numpy()
        ref=s["beats"][s["beats"]<T/fps]; dref=s["downs"][s["downs"]<T/fps]
        if len(ref)<2: continue
        m=_estimate_meter(ref,dref)
        Fs.append(f_measure(ref,beats_from_barphase(pm,m,fps))); Fm.append(f_measure(ref,metronome(T,fps)))
    model.train(); return float(np.mean(Fs)), float(np.mean(Fm))

def run(offset,label,steps=600,FR=256,BS=16):
    BarPointerVAE.unpack = make_patched(offset)
    torch.manual_seed(0); rng=np.random.default_rng(0)
    model=BarPointerVAE(h_dim=H,hidden=128,num_meters=4).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    f0,fm=evaluate(model); print(f"[{label}] BEFORE training: free-run beat_F={f0:.3f} (metronome {fm:.3f})",flush=True)
    for step in range(1,steps+1):
        beta=min(1.0,step/300); temp=1.0+(0.3-1.0)*min(step/steps,1.0)
        hs,bs_,ds=[],[],[]
        for _ in range(BS):
            s=train[rng.integers(len(train))]
            if s["T"]<=FR: continue
            st=int(rng.integers(0,s["T"]-FR))
            hs.append(torch.from_numpy(dirac(s["beats"],s["downs"],st,FR)))
            b,d=tg(s["beats"],s["downs"],st,FR); bs_.append(torch.from_numpy(b)); ds.append(torch.from_numpy(d))
        h=torch.stack(hs).to(dev); b=torch.stack(bs_).to(dev); d=torch.stack(ds).to(dev)
        opt.zero_grad(); loss,info=strict_elbo(model,h,b,d,temperature=temp,beta=beta)
        if not torch.isfinite(loss): print("NaN@",step); break
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        if step%200==0: print(f"   s{step} rec_b={info['recon_beat']:.2f} kl_phi={info['kl_phase']:.2f} kl_lv={info['kl_level']:.2f}",flush=True)
    f1,_=evaluate(model); print(f"[{label}] AFTER  training: free-run beat_F={f1:.3f}",flush=True)
    BarPointerVAE.unpack = ORIG
    return f0,f1

print("="*66)
a0,a1=run(0.0,       "AS-BUILT (log_tempo~0, 16x too fast)")
print("-"*66)
b0,b1=run(-2.77,     "TEMPO-FIXED (log_tempo~-2.77, physical)")
print("="*66)
print(f"RESULT  as-built {a1:.3f}   tempo-fixed {b1:.3f}   (metronome ~0.295)")
