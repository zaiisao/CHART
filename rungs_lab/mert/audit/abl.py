import sys,json,numpy as np,torch
from pathlib import Path
HERE=Path("/home/sogang/jaehoon/VBPM/rungs_lab/mert"); sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent))
from mert_r4_model import R4Conditioned
DEV="cuda:1"; FPS=44100/1024
cache=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
ck=torch.load(HERE/"runs/mertr4_mertfull_bestsel.pt",weights_only=False)
m=R4Conditioned(fps=FPS,input_mode="featsmert",device=DEV,input_dim=ck["input_dim"]); m.load_state_dict(ck["model"]); m.eval()
mean,std=cache["feat_mean"].to(DEV),cache["feat_std"].to(DEV)
mm,ms=cache["mert_mean"].to(DEV),cache["mert_std"].to(DEV)
rng=np.random.default_rng(0)
ents=cache["val_entries"][:40]
def tin(e):
    f=torch.from_numpy(cache["val_feats"][e["stem"]].astype(np.float32)).to(DEV); f=(f-mean)/std
    mt=torch.from_numpy(cache["val_mert"][e["stem"]].astype(np.float32)).to(DEV); mt=(mt-mm)/ms
    return f,mt
res={k:[] for k in ("mert_zero","mert_shuf","bt_zero","bt_shuf")}
argm={k:[] for k in ("base","mert_zero","bt_zero")}
haz={k:[] for k in ("base","mert_zero","bt_zero")}
with torch.no_grad():
  for e in ents:
    f,mt=tin(e); T=f.shape[0]
    def go(F,M):
        lp,lk,d=m.head_outputs(torch.cat([F,M],1)); return d["prior"].cpu().numpy(), d["component_weights"].cpu().numpy(), d["lambda_t"].cpu().numpy()
    p0,w0,l0=go(f,mt)
    perm=torch.from_numpy(rng.permutation(T)).to(DEV)
    variants={"mert_zero":(f,torch.zeros_like(mt)),"mert_shuf":(f,mt[perm]),
              "bt_zero":(torch.zeros_like(f),mt),"bt_shuf":(f[perm],mt)}
    for k,(F,M) in variants.items():
        p,w,l=go(F,M)
        tv=0.5*np.abs(p-p0).sum()
        dw=np.abs(w-w0).mean(); 
        res[k].append((tv, float(np.corrcoef(w[:,0],w0[:,0])[0,1]), dw, float(np.abs(l-l0).mean()/l0.mean())))
    argm["base"].append(p0.argmax())
    for k in ("mert_zero","bt_zero"):
        F,M=variants[k]; p,_,_=go(F,M); argm[k].append(p.argmax())
for k,v in res.items():
    a=np.array(v); print(f"{k:10s} priorTV {a[:,0].mean():.4f}  corr(hold_w) {np.nanmean(a[:,1]):.4f}  |dw| {a[:,2].mean():.4f}  rel|dlam| {a[:,3].mean():.4f}")
for k in ("mert_zero","bt_zero"):
    print(k,"prior argmax changed in", int((np.array(argm[k])!=np.array(argm["base"])).sum()),"/",len(ents))
