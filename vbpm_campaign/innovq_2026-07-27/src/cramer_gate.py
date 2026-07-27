"""GATE the differentiable displacement loss BEFORE wiring it to training.
  L = sum_t |cumsum(p)_t - cumsum(b)_t|, multi-scale.
Requirements to pass:
  (1) minimizes at tempo mult = 1.00
  (2) FAR-FIELD gradient nonzero (the whole point; BCE ~0)
  (3) monotone basin - no flat plateau
  (4) gradient actually flows to a tempo parameter"""
import sys, math, numpy as np, torch, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P
dev="cuda:0"; T=1500; TWO_PI=2*math.pi
ev=P.build_crops(P.load_songs("eval"),n_per_song=1,seed=1,crop=T,dev=dev)
n=32; b=ev["b"][:n]; tlt=ev["lt"][:n]; tphi=ev["phi"][:n]

def cramer(p, tgt, scales=(50,200,800,1500)):
    tot=0.0
    for L in scales:
        pad=(-p.shape[-1])%L
        pp=F.pad(p,(0,pad)).unflatten(-1,(-1,L)); tt=F.pad(tgt,(0,pad)).unflatten(-1,(-1,L))
        tot=tot+(pp.cumsum(-1)-tt.cumsum(-1)).abs().sum((-1,-2))
    return tot

def pulse_from_phi(phi, w_beats=0.06):
    """soft beat pulse at bar-phase multiples of 2pi/4. w_beats=0.06 ~= 1.5 frames at ibi 24.
    (previous version used w=1.5 BEATS -> pulse ranged 0.946..1.0, i.e. constant 1)"""
    d=torch.remainder(phi*4.0/TWO_PI+0.5,1.0)-0.5          # signed dist to nearest beat, in beats
    return torch.exp(-0.5*(d/w_beats)**2)
def traj(mult):
    lt=tlt+math.log(mult)
    inc=F.pad(torch.exp(lt.clamp(-12,6))[:,:-1],(1,0))
    return (tphi[:,:1]+torch.cumsum(inc,1))%TWO_PI

print(f"{'mult':>6} | {'Cramer':>12} {'BCE':>12}")
rows=[]
for m_ in (0.5,0.7,0.8,0.9,0.95,0.98,1.0,1.02,1.05,1.1,1.2,1.5,2.0):
    with torch.no_grad():
        pl=pulse_from_phi(traj(m_))
        c=float(cramer(pl,b).mean())
        bce=float(F.binary_cross_entropy(pl.clamp(1e-6,1-1e-6),b,reduction="none").sum(-1).mean())
    rows.append((m_,c,bce)); print(f"{m_:6.2f} | {c:12.1f} {bce:12.1f}")
cm=min(rows,key=lambda r:r[1]); bm=min(rows,key=lambda r:r[2])
far_c=abs(rows[1][1]-rows[0][1])/0.2; far_b=abs(rows[1][2]-rows[0][2])/0.2
near=[r for r in rows if 0.9<=r[0]<=1.1]
mono=all(near[i][1]>=near[i+1][1] for i in range(len(near)//2)) 
print(f"\n(1) Cramer min at mult={cm[0]:.2f}  {'PASS' if abs(cm[0]-1)<1e-9 else 'FAIL'}   (BCE min at {bm[0]:.2f})")
print(f"(2) far-field slope: Cramer {far_c:.1f}  vs  BCE {far_b:.1f}   "
      f"{'PASS' if far_c>10*max(far_b,1e-6) else 'FAIL'}")
print(f"(3) basin monotone approaching 1.0: {'PASS' if mono else 'CHECK'}")
# (4) gradient flows to a tempo scalar from far away
s=torch.zeros(1,device=dev,requires_grad=True)
lt=tlt+s; inc=F.pad(torch.exp(lt.clamp(-12,6))[:,:-1],(1,0))
phi=(tphi[:,:1]+torch.cumsum(inc,1))%TWO_PI
loss=cramer(pulse_from_phi(phi),b).mean(); loss.backward()
print(f"(4) d(Cramer)/d(log-tempo) at mult=1.0: {float(s.grad):+.2f}")
s2=torch.full((1,),math.log(1.3),device=dev,requires_grad=True)
lt=tlt+s2; inc=F.pad(torch.exp(lt.clamp(-12,6))[:,:-1],(1,0))
phi=(tphi[:,:1]+torch.cumsum(inc,1))%TWO_PI
cramer(pulse_from_phi(phi),b).mean().backward()
print(f"    d(Cramer)/d(log-tempo) at mult=1.3: {float(s2.grad):+.2f}   "
      f"{'PASS (points home)' if float(s2.grad)>0 else 'FAIL (wrong sign)'}")
