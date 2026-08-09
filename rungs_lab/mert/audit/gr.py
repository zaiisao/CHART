import sys,numpy as np,torch
from pathlib import Path
HERE=Path("/home/sogang/jaehoon/VBPM/rungs_lab/mert"); sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent))
from mert_r4_model import R4Conditioned
DEV="cuda:3"; FPS=44100/1024
cache=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
ck=torch.load(HERE/"runs/mertr4_mertfull_bestsel.pt",weights_only=False)
m=R4Conditioned(fps=FPS,input_mode="featsmert",device=DEV,input_dim=ck["input_dim"]); m.load_state_dict(ck["model"])
mean,std=cache["feat_mean"].to(DEV),cache["feat_std"].to(DEV); mm,ms=cache["mert_mean"].to(DEV),cache["mert_std"].to(DEV)
g={}
for i in range(8):
    c=cache["crops"][i]
    a=torch.from_numpy(c["acts"]).to(DEV); f=(torch.from_numpy(c["feats"]).to(DEV)-mean)/std
    mt=(torch.from_numpy(c["mert"].astype(np.float32)).to(DEV)-mm)/ms
    m.zero_grad(); (-m.marginal_ll(a,torch.cat([f,mt],1))/a.shape[0]).backward()
    for n,p in m.named_parameters():
        if p.grad is not None: g.setdefault(n,[]).append(p.grad.detach().clone())
W=torch.stack(g["trunk.embed.weight"]).mean(0)
print("grad embed BT block rms %.3e  MERT block rms %.3e"%(W[:,:256].pow(2).mean().sqrt(),W[:,256:].pow(2).mean().sqrt()))
tot=sum(torch.stack(v).mean(0).pow(2).sum() for v in g.values()).sqrt()
print("total grad norm (mean over 8 crops) %.4f ; clip is 1.0"%tot)
for n in ("prior_head.weight","kernel_head.weight","trunk.embed.weight"):
    print(n,"gradnorm %.4f"%torch.stack(g[n]).mean(0).norm())
# per-crop grad norms (unaveraged) to see clipping bite
print("per-crop total grad norms:", [round(float(sum(v[i].pow(2).sum() for v in g.values()).sqrt()),3) for i in range(8)])
