"""Is 2% tempo MAE REACHABLE from the model's features? Decisive split:
  train MAE -> capacity/representation limit    |    eval MAE -> generalisation/data limit
If train MAE cannot reach 2%, no loss shape helps: the features don't carry tempo precisely."""
import sys, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
dev="cuda:0"; torch.manual_seed(0); rng=np.random.default_rng(0); T=1500
tr=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev)
ev=P.build_crops(P.load_songs("eval"), n_per_song=1,seed=1,crop=T,dev=dev)
m=IQ.InnovQ().to(dev)
m.load_state_dict(torch.load("tempofix_stage1.pt",map_location=dev,weights_only=False)["model"],strict=False)
m.eval()
with torch.no_grad():                      # the encoder features the tempo head sees
    ctr=m.encode_posterior(tr["h"],tr["b"]); cev=m.encode_posterior(ev["h"],ev["b"])
    Xtr=torch.cat([ctr.mean(1),ctr[:,0]],-1); Xev=torch.cat([cev.mean(1),cev[:,0]],-1)
    ytr=torch.log(torch.exp(tr["lt"]).median(1).values); yev=torch.log(torch.exp(ev["lt"]).median(1).values)
print(f"features {Xtr.shape[1]}-d | train {len(ytr)} eval {len(yev)} | true sd {float(ytr.std()):.3f}")
for name,hid,steps in (("linear",0,4000),("MLP-256",256,8000),("MLP-1024",1024,8000)):
    torch.manual_seed(0)
    net=(nn.Linear(Xtr.shape[1],1) if hid==0 else
         nn.Sequential(nn.Linear(Xtr.shape[1],hid),nn.ReLU(),nn.Linear(hid,hid),nn.ReLU(),nn.Linear(hid,1))).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
    for s in range(steps):
        i=torch.tensor(rng.integers(0,len(ytr),32),device=dev,dtype=torch.long)
        loss=F.mse_loss(net(Xtr[i]).squeeze(-1),ytr[i]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mtr=float((net(Xtr).squeeze(-1)-ytr).abs().mean()); mev=float((net(Xev).squeeze(-1)-yev).abs().mean())
        r=float(torch.corrcoef(torch.stack([net(Xev).squeeze(-1),yev]))[0,1])
    print(f"  {name:9s}: TRAIN MAE {mtr:.3f} ({100*mtr:.1f}%) | EVAL MAE {mev:.3f} ({100*mev:.1f}%) corr {r:+.3f}")
print(f"\n  TARGET 0.020 (2%) for phase corr>=0.82 @T=1500")
print(f"  If TRAIN MAE >> 0.02 -> features lack the information (representation limit)")
print(f"  If TRAIN MAE ~ 0 but EVAL high -> only {len(ytr)} training crops (data limit)")
