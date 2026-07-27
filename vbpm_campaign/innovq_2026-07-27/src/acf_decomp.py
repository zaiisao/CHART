"""Decompose the 9.16% eval error: octave confusions vs local imprecision.
Also: what phase corr does each imply at T=1500?"""
import sys, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P
exec(open("acf_data.py").read().split("# --- classification")[0].replace('print(f"train','#print(f"train'))
NB=96; lo,hi=math.log(LO),math.log(HI)
edges=torch.linspace(lo,hi,NB,device=dev)
def to_soft(l):
    d=(l[:,None]-edges[None,:])/((hi-lo)/NB*1.5); return F.softmax(-0.5*d**2,dim=1)
torch.manual_seed(0)
net=nn.Sequential(nn.Linear(Xt.shape[1],512),nn.GELU(),nn.Dropout(0.3),
                  nn.Linear(512,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,NB)).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-2)
tgt=to_soft(lt)
for s in range(400):
    net.train(); opt.zero_grad(); perm=torch.randperm(len(lt),device=dev)[:256]
    (-(tgt[perm]*F.log_softmax(net(Xt[perm]),1)).sum(1).mean()).backward(); opt.step()
net.eval()
with torch.no_grad():
    lg=net(Xe); p=F.softmax(lg,1); hard=edges[lg.argmax(1)]
    w=(edges[None,:]-hard[:,None]).abs()<((hi-lo)/NB*3)
    pm=p*w; pm=pm/(pm.sum(1,keepdim=True)+1e-9); pred=(pm*edges[None,:]).sum(1)
err=(pred-le).cpu().numpy(); n=len(err)
OCT=[math.log(f) for f in (0.25,1/3,0.5,2/3,1.5,2,3,4)]
near=np.abs(err)<0.04
octv=np.array([(not near[i]) and min(abs(err[i]-o) for o in OCT)<0.06 for i in range(n)])
other=~near & ~octv
print(f"eval crops {n}")
print(f"  within 4% (correct)      {100*near.mean():5.1f}%   contributes {np.abs(err[near]).mean()*near.mean()*100:5.2f} pts")
print(f"  OCTAVE/metrical confusion{100*octv.mean():5.1f}%   contributes {np.abs(err[octv]).mean()*octv.mean()*100:5.2f} pts")
print(f"  other                    {100*other.mean():5.1f}%   contributes {(np.abs(err[other]).mean()*other.mean()*100) if other.any() else 0:5.2f} pts")
print(f"  TOTAL MAE {100*np.abs(err).mean():.2f}%")
if octv.any():
    from collections import Counter
    c=Counter([min(OCT,key=lambda o:abs(err[i]-o)) for i in range(n) if octv[i]])
    print("  confusion breakdown: "+", ".join(f"x{math.exp(k):.2f}:{v}" for k,v in c.most_common()))
print(f"\n  if octave were resolved -> MAE ~= {100*np.abs(err[near]).mean():.2f}%  (target 2%)")
