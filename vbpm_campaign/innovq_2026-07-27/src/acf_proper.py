"""Is tempo IN the 2-ch activation? Proper MIR-style ACF/tempogram on the raw activation,
compared against the encoder's pooled features. If ACF recovers tempo well, the information
is there and the ENCODER is the bottleneck (fixable without rich features)."""
import sys, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P
dev="cuda:0"; T=1500; FPS=50.0; TWO_PI=2*math.pi
ev=P.build_crops(P.load_songs("eval"),n_per_song=1,seed=1,crop=T,dev=dev)
B=ev["h"].shape[0]
act=ev["h"][...,0].cpu().numpy()                       # beat activation channel
true_ibi=np.array([np.median(np.diff(np.where(ev["b"][i].cpu().numpy()>0.5)[0])) for i in range(B)])
def acf_tempo(x, lo=10, hi=120, smooth=3):
    """FFT autocorrelation, restricted to plausible BEAT periods (0.2-2.4 s), parabolic refine."""
    x=x-x.mean()
    n=1<<int(np.ceil(np.log2(2*len(x))))
    f=np.fft.rfft(x,n); a=np.fft.irfft(f*np.conj(f),n)[:hi+2]
    a=a/ (a[0]+1e-9)
    if smooth>1:
        k=np.ones(smooth)/smooth; a=np.convolve(a,k,mode="same")
    seg=a[lo:hi]; i=int(np.argmax(seg))+lo
    # parabolic interpolation for sub-frame precision
    if 0<i<len(a)-1:
        d=(a[i-1]-a[i+1])/(2*(a[i-1]-2*a[i]+a[i+1])+1e-12); i=i+float(np.clip(d,-1,1))
    return i
est=np.array([acf_tempo(act[i]) for i in range(B)])
def report(name,e,t):
    lr=np.log(e/t); mae=np.abs(lr).mean()
    r=np.corrcoef(np.log(e),np.log(t))[0,1]
    oct_ok=np.mean(np.minimum(np.abs(lr),np.minimum(np.abs(lr-math.log(2)),np.abs(lr+math.log(2))))<0.04)
    print(f"  {name:22s} MAE {mae:.3f} ({100*mae:.1f}%) | corr {r:+.3f} | within 4% (octave-tol) {100*oct_ok:.0f}%")
print(f"eval crops {B}, true IBI median {np.median(true_ibi):.1f} frames ({60*FPS/np.median(true_ibi):.0f} BPM)")
report("ACF peak (raw)",est,true_ibi)
# octave-corrected: snap to the octave of the ACF peak closest to the corpus median
med=np.median(true_ibi)
snap=est.copy()
for i in range(B):
    cands=np.array([est[i]*f for f in (0.25,1/3,0.5,2/3,1,1.5,2,3,4)])
    snap[i]=cands[np.argmin(np.abs(np.log(cands/med)))]
report("ACF + octave snap",snap,true_ibi)
print(f"\n  encoder pooled features (measured earlier): TRAIN MAE 8.7%, EVAL 14.4%")
print(f"  TARGET: 2%")
