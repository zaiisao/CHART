"""Why does corr go to ~0 instead of ~0.8? Decompose the phase error of the placed model:
is it (a) linear DRIFT (rate error -> error sweeps through 2pi -> corr cancels by construction)
or (b) random scatter? Also: what corr would a model with a small % tempo error score?"""
import sys, math, glob
import numpy as np, torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
dev="cuda:0"; torch.manual_seed(0); TWO_PI=2*math.pi
T=1500
D=P.build_crops(P.load_songs("eval"),n_per_song=1,seed=1,crop=T,dev=dev)
# (A) SENSITIVITY: what corr does a PERFECT model with x% tempo error get, at each crop length?
print("(A) corr of an otherwise-perfect model with a small tempo error:")
print(f"{'tempo err':>9} | " + " | ".join(f"T={t}" for t in (256,512,1500)))
for pct in (0.0,0.5,1.0,2.0,5.0):
    row=[]
    for TT in (256,512,1500):
        phi=D["phi"][:,:TT]; rate=torch.exp(D["lt"][:,:TT])
        t=torch.arange(TT,device=dev).float().unsqueeze(0)
        drift=(pct/100.0)*rate.mean(1,keepdim=True)*t          # accumulating rate error
        c=float(torch.abs(torch.exp(1j*(drift)).mean()))
        row.append(f"{c:.3f}")
    print(f"{pct:8.1f}% | " + " | ".join(row))
# (B) the actual placed model: is its error drift or scatter?
ck=sorted(glob.glob("innovq_pf_sm101_s0.pt"))
m=IQ.InnovQ().to(dev)
if ck: m.load_state_dict(torch.load(ck[0],map_location=dev,weights_only=False).get("model"),strict=False)
m.eval()
with torch.no_grad():
    ro=IQ.rollout(m,D["h"][:6],D["b"][:6],sample=False)
    err=torch.angle(torch.exp(1j*(ro["phi"]-D["phi"][:6])))
    # unwrap the error to see drift
    ue=np.unwrap(err.cpu().numpy(),axis=1)
    slope=np.polyfit(np.arange(T),ue.T,1)[0]              # rad/frame of drift per crop
    resid=ue-(slope[:,None]*np.arange(T)[None,:]+ue[:,:1])
    true_rate=float(torch.exp(D["lt"][:6]).mean())
print("\n(B) placed model's phase error, per crop (6 eval crops):")
print(f"    drift slope (rad/frame): {np.round(slope,5)}")
print(f"    as % of true tempo {true_rate:.4f}: {np.round(100*slope/true_rate,1)} %")
print(f"    residual scatter after removing drift: {np.abs(resid).mean():.3f} rad")
print(f"    total error swing over the crop: {np.round(np.abs(ue[:,-1]-ue[:,0]),1)} rad  (2pi={TWO_PI:.2f})")
