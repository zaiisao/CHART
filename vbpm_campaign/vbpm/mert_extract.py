"""Extract frozen MERT-v1-95M ALL-13-layer features -> [13,T,768] fp16 at 50fps, per song,
for a fold-honest subset. Reuses the v17b_mert_merge extraction (mert_all_layers/interp_to_grid).
Cache -> /disk1 (roomy). Beat/downbeat times stored alongside; binary targets built at train time.
"""
import sys, os, numpy as np, torch, librosa
from pathlib import Path
REPO=Path("/home/sogang/jaehoon/VBPM"); sys.path.insert(0,str(REPO))
from transformers import AutoModel
DEVICE="cuda:0"; MERT_SR=24000; FPS=50; CROP=30
OUT=Path("/disk1/jaehoon/vbpm_mert_cache"); OUT.mkdir(parents=True,exist_ok=True)
ANN=REPO/"dataset_store/beat_this_annotations"
DATASETS=("ballroom","beatles","hainsworth")
CAP_TRAIN, CAP_EVAL = int(sys.argv[1]) if len(sys.argv)>1 else 48, 20

fold_of={}
for d in DATASETS:
    fp=ANN/d/"8-folds.split"
    if fp.exists():
        for line in open(fp):
            p=line.split()
            if len(p)==2: fold_of[p[0]]=int(p[1])
from data.songs import iter_songs
audio_of={s.stem:s.audio_path for s in iter_songs() if s.dataset in DATASETS and s.audio_path}

songs=[]
for d in DATASETS:
    bdir=ANN/d/"annotations"/"beats"
    for ann in sorted(bdir.glob("*.beats")):
        stem=ann.stem if ann.stem.startswith(d) else f"{d}_{ann.stem}"
        bt=np.loadtxt(ann,ndmin=2)
        if len(bt)<8 or stem not in audio_of: continue
        fold=fold_of.get(ann.stem,fold_of.get(stem))
        if fold is None: continue
        down=bt[bt[:,1]==1,0] if bt.shape[1]>1 else np.array([])
        songs.append(dict(stem=stem,dataset=d,fold=fold,beats=bt[:,0],downs=down,audio=audio_of[stem]))
# fold-honest subset: cap per dataset
rng=np.random.default_rng(0)
sub=[]
for split,cap,cond in [("train",CAP_TRAIN,lambda f:f!=0),("eval",CAP_EVAL,lambda f:f==0)]:
    for d in DATASETS:
        pool=[s for s in songs if s["dataset"]==d and cond(s["fold"])]
        pick=[pool[i] for i in rng.choice(len(pool),min(cap,len(pool)),replace=False)] if pool else []
        for s in pick: s["split"]=split
        sub+=pick
print(f"subset: {sum(s['split']=='train' for s in sub)} train + {sum(s['split']=='eval' for s in sub)} eval",flush=True)

mert=AutoModel.from_pretrained(str(REPO/"external/MERT-v1-95M"),trust_remote_code=True).to(DEVICE).eval()
for p in mert.parameters(): p.requires_grad_(False)

@torch.no_grad()
def mert_all_layers(wav24):
    fs,ts,step=[],[],CROP*MERT_SR
    for i in range(0,len(wav24),step):
        seg=wav24[i:i+step]
        if len(seg)<MERT_SR//2: break
        h=mert(torch.from_numpy(seg).float().unsqueeze(0).to(DEVICE),output_hidden_states=True).hidden_states
        f=torch.stack(h)[:,0]; n=f.shape[1]; dur=len(seg)/MERT_SR
        fs.append(f); ts.append(i/MERT_SR+(np.arange(n)+0.5)*(dur/n))
    return torch.cat(fs,1), np.concatenate(ts)

def interp_to_grid(x,t_src,n):
    t_dst=torch.as_tensor((np.arange(n)+0.5)/FPS,dtype=torch.float64,device=DEVICE)
    ts=torch.as_tensor(t_src,dtype=torch.float64,device=DEVICE)
    idx=torch.searchsorted(ts,t_dst).clamp(1,len(ts)-1)
    t0,t1=ts[idx-1],ts[idx]; w=((t_dst-t0)/(t1-t0).clamp(min=1e-9)).clamp(0,1).float().unsqueeze(1)
    return x[idx-1]*(1-w)+x[idx]*w

done=0
for s in sub:
    outp=OUT/f"{s['split']}__{s['stem']}.npz"
    if outp.exists(): done+=1; continue
    try:
        wav,_=librosa.load(str(s["audio"]),sr=MERT_SR,mono=True)
    except Exception as e:
        print("skip",s["stem"],e,flush=True); continue
    feats,tsrc=mert_all_layers(wav)                       # [13,n,768], [n]
    dur=len(wav)/MERT_SR; nfr=int(dur*FPS)
    g=torch.stack([interp_to_grid(feats[l],tsrc,nfr) for l in range(feats.shape[0])])  # [13,nfr,768]
    np.savez(outp, feats=g.half().cpu().numpy(), beats=s["beats"], downs=s["downs"],
             fold=s["fold"], dataset=s["dataset"], fps=FPS)
    done+=1
    if done%10==0: print(f"  {done}/{len(sub)} cached",flush=True)
print(f"DONE {done} songs -> {OUT}",flush=True)
