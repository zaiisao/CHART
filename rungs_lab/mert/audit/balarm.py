import sys,time,numpy as np,torch
from pathlib import Path
M=Path("/home/sogang/jaehoon/VBPM/rungs_lab/mert"); sys.path.insert(0,str(M)); sys.path.insert(0,str(M.parent))
from mert_r4_model import R4Conditioned
DEV=sys.argv[1]; SCALE=float(sys.argv[2]); STEPS=int(sys.argv[3]); FPS=44100/1024
torch.manual_seed(0); rng=np.random.default_rng(0)
c=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
mean,std=c["feat_mean"].to(DEV),c["feat_std"].to(DEV); mm,ms=c["mert_mean"].to(DEV),c["mert_std"].to(DEV)
crops=[(torch.from_numpy(x["acts"]).to(DEV),torch.from_numpy(x["feats"]).to(DEV),torch.from_numpy(x["mert"].astype(np.float32)).to(DEV)) for x in c["crops"]]
def ti(a,f,m): return torch.cat([(f-mean)/std, ((m-mm)/ms)*SCALE],1)
model=R4Conditioned(fps=FPS,input_mode="featsmert",device=DEV,input_dim=256+ms.shape[0])
opt=torch.optim.Adam(model.parameters(),lr=1e-3)
def sel():
    with torch.no_grad():
        v=[]
        for e in c["val_entries"][:24]:
            s=e["stem"];a=c["val_acts"][s];f=c["val_feats"][s].astype(np.float32);m=c["val_mert"][s].astype(np.float32)
            L=a.shape[0]
            if L>1400: st=(L-1400)//2;a,f,m=a[st:st+1400],f[st:st+1400],m[st:st+1400]
            a=torch.from_numpy(a).to(DEV);f=torch.from_numpy(f).to(DEV);m=torch.from_numpy(m).to(DEV)
            v.append(float(-model.marginal_ll(a,ti(a,f,m))/a.shape[0]))
    return float(np.mean(v))
t0=time.time()
for step in range(STEPS):
    idx=rng.choice(len(crops),16,replace=False); opt.zero_grad(); tot=0.
    for i in idx:
        a,f,m=crops[i]; l=(-model.marginal_ll(a,ti(a,f,m))/a.shape[0])/16; l.backward(); tot+=float(l)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if step%10==0 or step==STEPS-1: print(f"scale={SCALE} step {step} train {tot:.4f} sel {sel():.5f} t={time.time()-t0:.0f}s",flush=True)
