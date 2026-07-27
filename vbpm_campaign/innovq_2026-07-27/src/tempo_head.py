"""Is the tempo failure OCTAVE errors or unconstrained wander? Compare the model's inferred
tempo to truth per crop; check ratio against octave/metrical relations (2, 1/2, 3/2, 2/3, 3, 1/3)."""
import sys, math, glob
import numpy as np, torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
dev="cuda:0"; torch.manual_seed(0)
T=1500
D=P.build_crops(P.load_songs("eval"),n_per_song=1,seed=1,crop=T,dev=dev)
m=IQ.InnovQ().to(dev)
ck=glob.glob("innovq_pf_sm101_s0.pt")
if ck: m.load_state_dict(torch.load(ck[0],map_location=dev,weights_only=False).get("model"),strict=False)
m.eval()
with torch.no_grad():
    ro=IQ.rollout(m,D["h"][:30],D["b"][:30],sample=False)
    inf=torch.exp(ro["lt"]).median(1).values          # inferred bar-advance rad/frame
    tru=torch.exp(D["lt"][:30]).median(1).values
r=(inf/tru).cpu().numpy()
print(f"inferred/true tempo ratio, 30 eval crops:")
print(f"  median {np.median(r):.3f} | mean {r.mean():.3f} | min {r.min():.3f} | max {r.max():.3f}")
print(f"  deciles: {np.round(np.percentile(r,[10,25,50,75,90]),3)}")
cands={"1 (correct)":1.0,"2 (double)":2.0,"1/2 (half)":0.5,"3/2":1.5,"2/3":2/3,"3":3.0,"1/3":1/3,"4":4.0,"1/4":0.25}
print("\n  nearest metrical relation per crop (within 10%):")
hits={k:0 for k in cands}; none=0
for v in r:
    best=None
    for k,c in cands.items():
        if abs(v-c)/c<0.10: best=k; break
    if best: hits[best]+=1
    else: none+=1
for k,n in hits.items():
    if n: print(f"    {k:12s}: {n:2d}/{len(r)}")
print(f"    {'unrelated':12s}: {none:2d}/{len(r)}")
print(f"\n  log2 ratio spread (0=correct, ±1=octave): mean {np.mean(np.log2(r)):+.3f} sd {np.std(np.log2(r)):.3f}")
# what would corr be if ONLY the tempo were fixed to truth?
with torch.no_grad():
    phi0=ro["phi"][:,:1]
    t=torch.arange(T,device=dev).float().unsqueeze(0)
    fixed=(phi0+torch.exp(D["lt"][:30]).mean(1,keepdim=True)*t)%(2*math.pi)
    c_fix=float(torch.abs(torch.exp(1j*(fixed-D["phi"][:30])).mean(1)).mean())
    c_now=float(torch.abs(torch.exp(1j*(ro["phi"]-D["phi"][:30])).mean(1)).mean())
print(f"\n  per-crop corr now: {c_now:.3f} | with TRUE tempo, model's own start phase: {c_fix:.3f}")
