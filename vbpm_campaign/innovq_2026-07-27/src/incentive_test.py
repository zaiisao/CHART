"""INCENTIVE LADDER: does listening ever PAY? Measure phase's marginal reconstruction
worth (nats/crop) as a function of the incentive levers. Cheap: no training of the encoder,
oracle-vs-degraded ablation with decoders refit per condition.
  L1 crop length T: 256 vs 512 vs 1000  (does coasting stop being optimal on long windows?)
  L2 emission sharpness: bins/std of the phase->beat map
For each: worth = recon(coasting phase) - recon(true phase). Big worth => listening pays."""
import sys, math, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P
dev="cuda:0"; torch.manual_seed(0); TWO_PI=2*math.pi
def coast(phi_true, lt_true):
    """open-loop: start at true phase, integrate the crop's MEDIAN tempo, never listen."""
    B,T=phi_true.shape
    rate=torch.exp(lt_true).median(dim=1,keepdim=True).values
    t=torch.arange(T,device=phi_true.device).float().unsqueeze(0)
    return (phi_true[:,:1]+rate*t)%TWO_PI
def worth(T, width, n_per_song=1, steps=4000):
    D=P.build_crops(P.load_songs("train"),n_per_song=n_per_song,seed=0,crop=T,dev=dev)
    phi,lt=D["phi"],D["lt"]; mh=F.one_hot(D["m"],4).float().unsqueeze(1).expand(-1,T,-1)
    phic=coast(phi,lt)
    def Z(p): return torch.cat([torch.cos(p).unsqueeze(-1),torch.sin(p).unsqueeze(-1),lt.unsqueeze(-1),mh],-1)
    res={}
    for name,p in (("true",phi),("coast",phic)):
        torch.manual_seed(0)
        dec=nn.Sequential(nn.Linear(7,width),nn.Tanh(),nn.Linear(width,2)).to(dev)
        opt=torch.optim.AdamW(dec.parameters(),lr=1e-3)
        z=Z(p)
        for i in range(steps):
            o=dec(z)
            l=(F.binary_cross_entropy_with_logits(o[...,0],D["b"],reduction='none').sum(-1)
              +F.binary_cross_entropy_with_logits(o[...,1],D["db"],reduction='none').sum(-1)).mean()
            opt.zero_grad(); l.backward(); opt.step()
        res[name]=float(l)
    drift=float(torch.abs(torch.angle(torch.exp(1j*(phic-phi)))).mean())
    return res["coast"]-res["true"], res, drift
print(f"{'crop T':>7} {'width':>6} | {'recon(true)':>11} {'recon(coast)':>12} | {'WORTH of listening':>18} | mean drift(rad)")
for T in (256,512,1000):
    w,res,dr=worth(T,128)
    print(f"{T:7d} {128:6d} | {res['true']:11.2f} {res['coast']:12.2f} | {w:18.2f} | {dr:.3f}")
print()
for width in (256,512):
    w,res,dr=worth(512,width)
    print(f"{512:7d} {width:6d} | {res['true']:11.2f} {res['coast']:12.2f} | {w:18.2f} | {dr:.3f}")
print("\n(worth = nats/crop the model gains by LISTENING instead of coasting; 108 was the old 256-frame figure)")
