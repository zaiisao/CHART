"""Preliminary MECHANISM check for Fix A: with an audio-conditioned prior mean, does the
deploy phase trajectory RESPOND to the input at all? (untrained -> response is meaningless
in value, but proves the pathway exists). Unfixed model responds 0.0099 rad to a 25-frame shift."""
import sys, glob, math
import numpy as np, torch, torch.nn as nn
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.distributions import TWO_PI, sample_wrapped_cauchy, sample_student_t, gumbel_softmax
CACHE="/disk1/jaehoon/vbpm_mert_cache"; fps=50.0; dev="cuda:0"; H=8; T=800
d=np.load(sorted(glob.glob(f"{CACHE}/eval__*.npz"))[0],allow_pickle=True)
beats=np.asarray(d["beats"],float); downs=np.asarray(d["downs"],float)
def dirac(shift):
    h=np.zeros((T,H),np.float32)
    for t in beats:
        i=int(round(t*fps))+shift
        if 0<=i<T: h[i,0]=1.0
    for t in downs:
        i=int(round(t*fps))+shift
        if 0<=i<T: h[i,1]=1.0
    return torch.from_numpy(h).unsqueeze(0).to(dev)

class FixA(BarPointerVAE):
    """audio-conditioned prior MEAN: mu_phi = phi_prev + phidot_prev + tanh(f(h_t))*scale"""
    def __init__(self,*a,corr_scale=0.5,**k):
        super().__init__(*a,**k)
        self.prior_phase_corr=nn.Linear(self.hidden,1); self.corr_scale=corr_scale
    def phase_corr(self,ctx_t):
        return torch.tanh(self.prior_phase_corr(ctx_t).squeeze(-1))*self.corr_scale

@torch.no_grad()
def deploy(model, h, use_corr, log_tempo_off=-2.77):
    """free-run mean chain; if use_corr, the audio moves the phase mean each frame."""
    B,Tn,_=h.shape; ctx=model.encode_prior(h)
    p=model.unpack(model.prior_init_head(ctx.mean(1)))
    phi_mu=p[1]%TWO_PI; level_mu=p[3]+log_tempo_off
    traj=[phi_mu]
    for t in range(1,Tn):
        adv=phi_mu+torch.exp(level_mu)
        if use_corr: adv=adv+model.phase_corr(ctx[:,t])
        phi_mu=adv%TWO_PI; traj.append(phi_mu)
    return torch.stack(traj,1)[0].cpu().numpy()

torch.manual_seed(0)
m=FixA(h_dim=H,hidden=128,num_meters=4,corr_scale=0.5).to(dev).eval()
print("shift | UNFIXED (no corr) response | FIX-A (audio-cond mean) response   [max |dphase| rad]")
for sh in [5,10,25]:
    a0=deploy(m,dirac(0),False); a1=deploy(m,dirac(sh),False)
    b0=deploy(m,dirac(0),True);  b1=deploy(m,dirac(sh),True)
    da=np.abs(np.angle(np.exp(1j*(a1-a0)))).max()
    db=np.abs(np.angle(np.exp(1j*(b1-b0)))).max()
    print(f" {sh:3d}f |        {da:.6f}            |        {db:.6f}   ({db/max(da,1e-9):.0f}x more responsive)")
