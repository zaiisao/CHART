"""Diagnostic: does a LINEAR probe + peak-picking on the MERT weighted features recover beats?
Separates 'features carry beats' (probe works) from 'the VAE machinery fails' (VBPM free-run ~floor).
Same cached MERT features, same fold-honest split, same eval songs as train_mert.py.
"""
import sys, glob, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.evaluate import beats_from_activation, metronome, f_measure
CACHE="/disk1/jaehoon/vbpm_mert_cache"; dev="cuda:0"; fps=50.0

def load(split):
    out=[]
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d=np.load(f,allow_pickle=True)
        out.append(dict(feats=d["feats"],beats=np.asarray(d["beats"],float),downs=np.asarray(d["downs"],float)))
    return out
train=load("train"); ev=load("eval")[:30]
print(f"probe: train {len(train)} eval {len(ev)}",flush=True)

class Probe(nn.Module):
    def __init__(self, kind="linear"):
        super().__init__()
        self.layer_logits=nn.Parameter(torch.zeros(13))
        if kind=="linear": self.head=nn.Linear(768,1)                       # PURE linear, per-frame
        else: self.head=nn.Sequential(nn.Conv1d(768,128,5,padding=2),nn.ReLU(),nn.Conv1d(128,1,1))  # small temporal
        self.kind=kind
    def forward(self,feats):            # [B,13,T,768]
        w=torch.softmax(self.layer_logits,0); m=torch.einsum("l,bltf->btf",w,feats)  # [B,T,768]
        if self.kind=="linear": return self.head(m).squeeze(-1)            # [B,T]
        return self.head(m.transpose(1,2)).squeeze(1)

def tgt(beats,start,n):
    y=np.zeros(n,np.float32)
    for t in beats:
        i=int(round(t*fps))-start
        if 0<=i<n: y[i]=1.0
    return y

def run(kind, steps=800, frames=512, bs=16):
    torch.manual_seed(0); rng=np.random.default_rng(0)
    p=Probe(kind).to(dev); opt=torch.optim.AdamW(p.parameters(),lr=3e-4)
    pw=torch.tensor([12.0],device=dev)
    for step in range(1,steps+1):
        fe,ys=[],[]
        for _ in range(bs):
            s=train[rng.integers(len(train))]; T=s["feats"].shape[1]
            if T<=frames: continue
            st=int(rng.integers(0,T-frames)); fe.append(torch.from_numpy(s["feats"][:,st:st+frames].astype(np.float32)))
            ys.append(torch.from_numpy(tgt(s["beats"],st,frames)))
        fe=torch.stack(fe).to(dev); ys=torch.stack(ys).to(dev)
        opt.zero_grad(); logit=p(fe); loss=F.binary_cross_entropy_with_logits(logit,ys,pos_weight=pw)
        loss.backward(); opt.step()
        if step%200==0: print(f"  [{kind}] s{step} loss={loss.item():.3f}",flush=True)
    # eval: peak-pick beat prob vs GT beats, fold-0 eval songs
    p.eval(); Fs=[]; Fm=[]
    with torch.no_grad():
        for s in ev:
            T=min(s["feats"].shape[1],1600)
            prob=torch.sigmoid(p(torch.from_numpy(s["feats"][:,:T].astype(np.float32)).unsqueeze(0).to(dev)))[0].cpu().numpy()
            ref=s["beats"][s["beats"]<T/fps]
            if len(ref)<2: continue
            Fs.append(f_measure(ref, beats_from_activation(prob,fps,thr=0.5)))
            Fm.append(f_measure(ref, metronome(T,fps)))
    lw=torch.softmax(p.layer_logits.detach(),0).cpu().numpy()
    print(f"[{kind}] PEAK-PICK beat_F={np.mean(Fs):.3f}  (metronome {np.mean(Fm):.3f})  top layers={np.argsort(lw)[::-1][:4]}",flush=True)
    return float(np.mean(Fs))

run("linear")
run("conv")
