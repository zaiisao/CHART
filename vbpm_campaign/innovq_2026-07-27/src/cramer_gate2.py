"""Displacement loss, correctly normalized: BOTH sides are counting measures, so
cumsum(.) = "beats elapsed by frame t" and |cumsum(p)-cumsum(b)| is a beat-count lag.
  p_t = N(d_t; 0, w) * (beats per frame)   -> sum_t p_t = number of beats  (exact)
  b_t = gaussian-smoothed Dirac train      -> mass preserved
GATE: min at 1.00, far-field slope >> BCE, gradient points home from 1.3x."""
import sys, math, numpy as np, torch, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P
dev="cuda:0"; T=1500; TWO_PI=2*math.pi; SQ2PI=math.sqrt(2*math.pi)
ev=P.build_crops(P.load_songs("eval"),n_per_song=1,seed=1,crop=T,dev=dev)
n=32; b=ev["b"][:n]; tlt=ev["lt"][:n]; tphi=ev["phi"][:n]
W=0.06                                   # pulse sd in BEATS (~1.4 frames at ibi 24)

def smooth_target(bt, sig=1.44):
    k=int(4*sig)|1; t=torch.arange(k,device=bt.device)-k//2
    g=torch.exp(-0.5*(t/sig)**2); g=g/g.sum()
    return F.conv1d(bt.unsqueeze(1), g.view(1,1,-1), padding=k//2).squeeze(1)

def pulse(phi, lt):
    d=torch.remainder(phi*4.0/TWO_PI+0.5,1.0)-0.5          # signed dist to beat, in beats
    bpf=torch.exp(lt.clamp(-12,6))*4.0/TWO_PI              # beats advanced per frame
    return torch.exp(-0.5*(d/W)**2)/(SQ2PI*W)*bpf          # mass == number of beats

def cramer(p, tgt, scales=(50,200,800,1500)):
    tot=0.0
    for L in scales:
        pad=(-p.shape[-1])%L
        pp=F.pad(p,(0,pad)).unflatten(-1,(-1,L)); tt=F.pad(tgt,(0,pad)).unflatten(-1,(-1,L))
        tot=tot+(pp.cumsum(-1)-tt.cumsum(-1)).abs().sum((-1,-2))
    return tot

bs=smooth_target(b)
print(f"mass check: pulse@1.0 {pulse(tphi,tlt).sum(-1).mean():.1f} | target {bs.sum(-1).mean():.1f} "
      f"| true beats {b.sum(-1).mean():.1f}")
def traj(mult):
    lt=tlt+math.log(mult)
    inc=F.pad(torch.exp(lt.clamp(-12,6))[:,:-1],(1,0))
    return (tphi[:,:1]+torch.cumsum(inc,1))%TWO_PI, lt
print(f"\n{'mult':>6} | {'Cramer':>11} {'BCE':>10}")
rows=[]
for m_ in (0.5,0.7,0.8,0.9,0.95,0.98,1.0,1.02,1.05,1.1,1.2,1.5,2.0):
    with torch.no_grad():
        ph,lt=traj(m_); pl=pulse(ph,lt)
        c=float(cramer(pl,bs).mean())
        bce=float(F.binary_cross_entropy(pl.clamp(1e-6,1-1e-6),b,reduction="none").sum(-1).mean())
    rows.append((m_,c,bce)); print(f"{m_:6.2f} | {c:11.1f} {bce:10.1f}")
cm=min(rows,key=lambda r:r[1]); bm=min(rows,key=lambda r:r[2])
far_c=abs(rows[1][1]-rows[0][1])/0.2; far_b=abs(rows[1][2]-rows[0][2])/0.2
print(f"\n(1) Cramer min at mult={cm[0]:.2f}  {'PASS' if abs(cm[0]-1)<1e-9 else 'FAIL'}  (BCE min {bm[0]:.2f})")
print(f"(2) far-field slope Cramer {far_c:.1f} vs BCE {far_b:.1f}  {'PASS' if far_c>5*max(far_b,1e-9) else 'FAIL'}")
g=[]
for st in (1.0,1.1,1.3,0.8,0.7):
    s=torch.full((1,),math.log(st),device=dev,requires_grad=True)
    lt=tlt+s; inc=F.pad(torch.exp(lt.clamp(-12,6))[:,:-1],(1,0))
    ph=(tphi[:,:1]+torch.cumsum(inc,1))%TWO_PI
    cramer(pulse(ph,lt),bs).mean().backward()
    gr=float(s.grad); ok=(gr>0) if st>1 else ((gr<0) if st<1 else True)
    g.append(ok); print(f"(3) d/dlog-tempo at mult={st}: {gr:+11.1f}  {'points home' if ok else 'WRONG SIGN'}")
print(f"\nGATE {'PASSED' if abs(cm[0]-1)<1e-9 and far_c>5*max(far_b,1e-9) and all(g) else 'FAILED'}")
