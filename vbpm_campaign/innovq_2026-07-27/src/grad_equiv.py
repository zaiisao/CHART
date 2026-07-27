"""Does BPTT through the vectorized rollout give the same PARAMETER GRADIENTS as the loop?
The one property that matters for training use. Compare grad vectors: cosine + relative norm."""
import sys, math, glob
import torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
from rollout_vec import rollout_vec
dev="cuda:0"; torch.manual_seed(0)
D=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=1500,dev=dev)
m=IQ.InnovQ().to(dev)
ck=glob.glob("innovq_pf_sm101_s0.pt")
if ck: m.load_state_dict(torch.load(ck[0],map_location=dev,weights_only=False).get("model"),strict=False)
m.train()
h,b,lt_t=D["h"][:3],D["b"][:3],D["lt"][:3]
ps=[p for p in m.parameters() if p.requires_grad]
def flat(gs): return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(gs,ps)])
# same supervised tempo loss the experiment uses
L1=((IQ.rollout(m,h,b,sample=False)["lt"]-lt_t)**2).mean()
g1=flat(torch.autograd.grad(L1,ps,allow_unused=True))
L2=((rollout_vec(m,h,b,n_picard=3)["lt"]-lt_t)**2).mean()
g2=flat(torch.autograd.grad(L2,ps,allow_unused=True))
cos=float(torch.nn.functional.cosine_similarity(g1,g2,dim=0))
print(f"loss  loop={float(L1):.6f}  vec={float(L2):.6f}  rel diff {abs(float(L1)-float(L2))/max(float(L1),1e-9):.3e}")
print(f"grad  |g_loop|={float(g1.norm()):.4e}  |g_vec|={float(g2.norm()):.4e}  ratio {float(g2.norm()/g1.norm()):.4f}")
print(f"grad  COSINE SIMILARITY = {cos:.6f}   <- >0.99 means the vec path trains the same direction")
