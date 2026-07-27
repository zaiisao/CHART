"""Is the Cramer loss minimized by a FLAT RAMP instead of sharp beats?
Compare, at identical total mass:
  (a) correct sharp pulses at the true beats   <- must WIN
  (b) constant density  #beats/T               <- the degenerate solution
  (c) half-rate sharp pulses
Then repeat with SHORT scales added (4,8,16) to see if that repairs it."""
import sys, math, numpy as np, torch, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P
dev="cuda:0"; T=1500
ev=P.build_crops(P.load_songs("eval"),n_per_song=1,seed=1,crop=T,dev=dev)
b=ev["b"][:32]
tgt=P._smooth_dirac(b)
mass=tgt.sum(-1,keepdim=True)
sharp=tgt.clone()                                   # (a) exactly right
flat=(mass/T).expand(-1,T).contiguous()             # (b) constant density, same mass
half=P._smooth_dirac(b.clone()); half[:,1::2]=0.0   # crude (c): drop half the beats
half=half*(mass/half.sum(-1,keepdim=True).clamp(min=1e-6))
def cram(p,t,scales):
    tot=0.0
    for L in scales:
        pad=(-p.shape[-1])%L
        pp=F.pad(p,(0,pad)).unflatten(-1,(-1,L)); tt=F.pad(t,(0,pad)).unflatten(-1,(-1,L))
        tot=tot+(pp.cumsum(-1)-tt.cumsum(-1)).abs().sum((-1,-2))
    return float(tot.mean())
for nm,scales in (("SHIPPED   (50,200,800,T)",(50,200,800,T)),
                  ("+short    (4,8,16,50,200,800,T)",(4,8,16,50,200,800,T)),
                  ("short only(2,4,8,16)",(2,4,8,16))):
    a,f_,h=cram(sharp,tgt,scales),cram(flat,tgt,scales),cram(half,tgt,scales)
    verdict="DEGENERATE (flat beats sharp)" if f_<=a else "ok (sharp wins)"
    print(f"  {nm:32s} sharp {a:9.1f} | flat {f_:9.1f} | half-rate {h:9.1f}   {verdict}")
print(f"\n  mean beats/crop {float(mass.mean()):.1f}, ibi ~{T/float(mass.mean()):.1f} frames"
      f"  -> shortest shipped scale (50) spans {50/(T/float(mass.mean())):.1f} beats")
