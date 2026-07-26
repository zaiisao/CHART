"""ZERO-TRAINING probe: does correct tempo INIT alone lift Dirac free-run off the floor?
Sweeps the log-tempo offset; evaluates free_run immediately (no training)."""
import sys, glob, math
import numpy as np, torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import free_run
from vbpm.evaluate import beats_from_barphase, metronome, f_measure, _estimate_meter
CACHE="/disk1/jaehoon/vbpm_mert_cache"; fps=50.0; TWO_PI=2*math.pi; dev="cuda:0"; H=8
ev=[]
for f in sorted(glob.glob(f"{CACHE}/eval__*.npz"))[:20]:
    d=np.load(f,allow_pickle=True)
    ev.append(dict(T=int(d["feats"].shape[1]),beats=np.asarray(d["beats"],float),downs=np.asarray(d["downs"],float)))
def dirac(b_,d_,n):
    h=np.random.randn(n,H).astype(np.float32)*0.01
    for t in b_:
        i=int(round(t*fps))
        if i<n: h[i,0]+=1.0
    for t in d_:
        i=int(round(t*fps))
        if i<n: h[i,1]+=1.0
    return h
ORIG=BarPointerVAE.unpack
# what IS the true log-tempo of these songs?
tl=[]
for s in ev:
    if len(s["beats"])>2:
        ibi=np.median(np.diff(s["beats"])); m=_estimate_meter(s["beats"],s["downs"])
        tl.append(math.log(TWO_PI/(ibi*m*fps)))
print(f"TRUE log-tempo across eval songs: mean={np.mean(tl):.2f} range=[{np.min(tl):.2f},{np.max(tl):.2f}]")
print(f"  (i.e. phidot {math.exp(np.mean(tl)):.4f} rad/frame; model as-built inits near 0.0 => 1.0 rad/frame)\n")
for off in [0.0,-1.0,-2.0,-2.77,-3.5]:
    def up(self,vec,o=off):
        a,b,c,lmu,ls,dmu,ds=ORIG(self,vec); return a,b,c,lmu+o,ls,dmu,ds
    BarPointerVAE.unpack=up
    torch.manual_seed(0)
    model=BarPointerVAE(h_dim=H,hidden=128,num_meters=4).to(dev).eval()
    Fs=[];Fm=[];nb=[];nr=[]
    with torch.no_grad():
        for s in ev:
            T=min(s["T"],1600)
            h=torch.from_numpy(dirac(s["beats"],s["downs"],T)).unsqueeze(0).to(dev)
            pm=free_run(model,h)["phase_mu"][0,:T].cpu().numpy()
            ref=s["beats"][s["beats"]<T/fps]; dref=s["downs"][s["downs"]<T/fps]
            if len(ref)<2: continue
            m=_estimate_meter(ref,dref); est=beats_from_barphase(pm,m,fps)
            Fs.append(f_measure(ref,est)); Fm.append(f_measure(ref,metronome(T,fps)))
            nb.append(len(est)); nr.append(len(ref))
    BarPointerVAE.unpack=ORIG
    print(f"offset {off:6.2f} -> free-run beat_F={np.mean(Fs):.3f}  (metronome {np.mean(Fm):.3f})  "
          f"est beats {np.mean(nb):6.1f} vs true {np.mean(nr):5.1f}  ratio {np.mean(nb)/max(np.mean(nr),1):5.2f}x")
