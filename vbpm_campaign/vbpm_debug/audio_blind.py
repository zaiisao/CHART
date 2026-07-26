"""DECISIVE: is the deploy (free_run) phase trajectory audio-blind?
Shift the Dirac beat impulses by k frames. If the model can track audio, the phase_mu
trajectory must shift with them. If it is audio-blind, the trajectory is unchanged."""
import sys, glob, math
import numpy as np, torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import free_run
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
torch.manual_seed(0)
m=BarPointerVAE(h_dim=H,hidden=128,num_meters=4).to(dev).eval()
outs={}
with torch.no_grad():
    for sh in [0,5,10,25]:
        torch.manual_seed(0)                      # same sampling noise; only h differs
        outs[sh]=free_run(m,dirac(sh))["phase_mu"][0].cpu().numpy()
base=outs[0]
print("shift(frames) | max|Dphase_mu| vs shift=0 | mean|Dphase| | identical?")
for sh in [5,10,25]:
    dphi=np.abs(np.angle(np.exp(1j*(outs[sh]-base))))
    print(f"   {sh:3d}        |   {dphi.max():.6f}            |  {dphi.mean():.6f}   | {np.allclose(outs[sh],base,atol=1e-5)}")
print()
print("Also: does the deterministic chain advance at a CONSTANT rate (= metronome)?")
dd=np.diff(base); dd=dd[dd>0]
print(f"  phase_mu per-frame increment: mean={dd.mean():.5f} std={dd.std():.2e}  -> {'CONSTANT (metronome)' if dd.std()<1e-6 else 'varies'}")
