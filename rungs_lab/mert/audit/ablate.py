import sys, json
from pathlib import Path
import numpy as np, torch
HERE = Path("/home/sogang/jaehoon/VBPM/rungs_lab/mert")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from mert_r4_model import R4Conditioned
DEVICE="cuda:3"; FPS=44100/1024
cache = torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt", weights_only=False)
ck = torch.load(HERE/"runs/mertr4_mertfull_bestsel.pt", weights_only=False)
print("ckpt keys", {k:v for k,v in ck.items() if k!="model"})
model = R4Conditioned(fps=FPS, input_mode=ck["input"], device=DEVICE, input_dim=ck["input_dim"])
model.load_state_dict(ck["model"]); model.eval()
print("num_tempi", model.num_tempi, "min_interval", model._min_interval)
mean,std = cache["feat_mean"].to(DEVICE), cache["feat_std"].to(DEVICE)
mmean,mstd = cache["mert_mean"].to(DEVICE), cache["mert_std"].to(DEVICE)

W = model.trunk.embed.weight.detach()   # [128, 3328]
bt = W[:, :256]; mt = W[:, 256:]
print("embed |W| per-dim rms: BT %.5f MERT %.5f" % (bt.pow(2).mean().sqrt(), mt.pow(2).mean().sqrt()))
print("embed block Frobenius: BT %.4f MERT %.4f  (col-sumsq total BT %.4f MERT %.4f)"%(bt.norm(),mt.norm(),bt.pow(2).sum(),mt.pow(2).sum()))
# contribution variance to embed output assuming unit-var standardized inputs
print("pred output var contribution: BT %.4f MERT %.4f"%(bt.pow(2).sum(1).mean(), mt.pow(2).sum(1).mean()))

def tin(e, mode="full", g=None):
    f = torch.from_numpy(cache["val_feats"][e["stem"]].astype(np.float32)).to(DEVICE); f=(f-mean)/std
    m = torch.from_numpy(cache["val_mert"][e["stem"]].astype(np.float32)).to(DEVICE); m=(m-mmean)/mstd
    if mode=="nomert": m=torch.zeros_like(m)
    if mode=="shufmert": m=m[torch.randperm(m.shape[0],generator=g,device=DEVICE)]
    if mode=="nobt": f=torch.zeros_like(f)
    if mode=="shufbt": f=f[torch.randperm(f.shape[0],generator=g,device=DEVICE)]
    return torch.cat([f,m],1)

g=torch.Generator(device=DEVICE); g.manual_seed(0)
res={k:{"tv":[], "kern_tv":[], "argmax_same":[]} for k in ["nomert","shufmert","nobt","shufbt"]}
with torch.no_grad():
    for e in cache["val_entries"]:
        lp0,lk0,d0 = model.head_outputs(tin(e))
        p0=d0["prior"]
        for k in res:
            _,lk1,d1 = model.head_outputs(tin(e,k,g))
            p1=d1["prior"]
            res[k]["tv"].append(float(0.5*(p0-p1).abs().sum()))
            res[k]["kern_tv"].append(float(0.5*(lk0.exp()-lk1.exp()).abs().sum(-1).mean()))
            res[k]["argmax_same"].append(float(p0.argmax()==p1.argmax()))
for k,v in res.items():
    print(k, "priorTV %.4f  kernelTV %.4f  argmax_same %.3f"%(np.mean(v["tv"]),np.mean(v["kern_tv"]),np.mean(v["argmax_same"])))
