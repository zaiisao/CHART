"""Stage-by-stage ORACLE decomposition: where is the architecture broken?
S1 read-out : ideal bar phase -> beats_from_barphase  (should be ~0.95+)
S2 tempo    : what BPM does free-run actually emit vs true BPM (units/init sanity)
S3 emission : can the DECODER fit beats from ORACLE latents? (can a small MLP on
              (cos phi, sin phi, log_tempo, meter) express a 1-frame-wide beat comb?)
S4 posterior: given Dirac h + b, does q's phase mean track the true bar phase?
"""
import sys, glob, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import free_run
from vbpm.evaluate import beats_from_barphase, downbeats_from_barphase, metronome, f_measure, _estimate_meter
CACHE="/disk1/jaehoon/vbpm_mert_cache"; fps=50.0; TWO_PI=2*math.pi
dev="cuda:0" if torch.cuda.is_available() else "cpu"

def load(split,cap=None):
    out=[]
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d=np.load(f,allow_pickle=True)
        out.append(dict(T=int(d["feats"].shape[1]),beats=np.asarray(d["beats"],float),
                        downs=np.asarray(d["downs"],float)))
    return out[:cap] if cap else out
ev=load("eval",20)

def oracle_barphase(beats,downs,T):
    """ideal BAR phase: 0 at each downbeat, linear to 2pi at the next downbeat."""
    t=(np.arange(T)+0.5)/fps
    anchors = downs if len(downs)>=2 else beats[::4] if len(beats)>=8 else beats
    if len(anchors)<2: return None
    ph=np.zeros(T)
    for i in range(len(anchors)-1):
        a,b=anchors[i],anchors[i+1]
        msk=(t>=a)&(t<b)
        ph[msk]=TWO_PI*(t[msk]-a)/max(b-a,1e-6)
    msk=t<anchors[0]; ph[msk]=0.0
    msk=t>=anchors[-1]; ph[msk]=0.0
    return ph

print("="*70); print("S1  READ-OUT with ORACLE bar phase"); print("="*70)
f1=[];fm=[]
for s in ev:
    T=min(s["T"],1600); ph=oracle_barphase(s["beats"],s["downs"],T)
    if ph is None: continue
    ref=s["beats"][s["beats"]<T/fps]; dref=s["downs"][s["downs"]<T/fps]
    if len(ref)<2: continue
    m=_estimate_meter(ref,dref)
    f1.append(f_measure(ref,beats_from_barphase(ph,m,fps))); fm.append(f_measure(ref,metronome(T,fps)))
print(f"  oracle-phase read-out beat_F = {np.mean(f1):.3f}   (metronome {np.mean(fm):.3f})")
print(f"  VERDICT: {'READ-OUT OK' if np.mean(f1)>0.85 else '*** READ-OUT BROKEN ***'}")

print(); print("="*70); print("S2  FREE-RUN TEMPO UNITS / INIT"); print("="*70)
torch.manual_seed(0)
m0=BarPointerVAE(h_dim=8,hidden=128,num_meters=4).to(dev)
h=torch.randn(4,600,8,device=dev)*0.01
out=free_run(m0,h)
lt=out["log_tempo"][:, :].detach().cpu().numpy()
pm=out["phase_mu"][0].detach().cpu().numpy()
wraps=int((np.diff(pm)< -math.pi).sum())
# true: 120bpm, m=4 -> 0.5 bar/s -> phidot = 0.5*2pi/50 = 0.0628 rad/frame -> log = -2.77
print(f"  init free-run log_tempo: mean={lt.mean():.2f} (implies {math.exp(lt.mean()):.3f} rad/frame)")
print(f"  physically correct for 120bpm/4-4 @50fps: log_tempo = {math.log(0.5*TWO_PI/fps):.2f} (0.063 rad/frame)")
print(f"  -> model init is {math.exp(lt.mean())/(0.5*TWO_PI/fps):.1f}x TOO FAST")
print(f"  bar wraps in 600 frames (12s): {wraps}  (true ~6 bars)")

print(); print("="*70); print("S3  EMISSION: can decoder fit beats from ORACLE latents?"); print("="*70)
# build oracle z_features for real songs, train ONLY the decoder
Xs,Ys,Ds=[],[],[]
for s in ev[:12]:
    T=min(s["T"],1600); ph=oracle_barphase(s["beats"],s["downs"],T)
    if ph is None: continue
    ibi=np.median(np.diff(s["beats"])) if len(s["beats"])>2 else 0.5
    mm=_estimate_meter(s["beats"],s["downs"])
    phidot=TWO_PI/(ibi*mm*fps)                      # rad/frame, true
    z=np.zeros((T,7),np.float32)
    z[:,0]=np.cos(ph); z[:,1]=np.sin(ph); z[:,2]=math.log(phidot); z[:,2+mm if 2+mm<7 else 6]=1.0
    y=np.zeros(T,np.float32); d=np.zeros(T,np.float32)
    for t in s["beats"]:
        i=int(round(t*fps));  y[i]=1.0 if i<T else 0
    for t in s["downs"]:
        i=int(round(t*fps));  d[i]=1.0 if i<T else 0
    Xs.append(z);Ys.append(y);Ds.append(d)
X=torch.from_numpy(np.concatenate(Xs)).to(dev); Y=torch.from_numpy(np.concatenate(Ys)).to(dev); D=torch.from_numpy(np.concatenate(Ds)).to(dev)
base=F.binary_cross_entropy(torch.full_like(Y,Y.mean()),Y).item()
for width,name in [(128,"MLP-128 (as built)"),(512,"MLP-512 (wider)")]:
    torch.manual_seed(0)
    dec=nn.Sequential(nn.Linear(7,width),nn.Tanh(),nn.Linear(width,2)).to(dev)
    op=torch.optim.AdamW(dec.parameters(),lr=1e-3)
    for i in range(3000):
        op.zero_grad(); o=dec(X)
        l=F.binary_cross_entropy_with_logits(o[:,0],Y)+F.binary_cross_entropy_with_logits(o[:,1],D)
        l.backward(); op.step()
    with torch.no_grad():
        o=dec(X); bl=F.binary_cross_entropy_with_logits(o[:,0],Y).item()
    print(f"  {name}: beat BCE {bl:.4f} vs base-rate {base:.4f}  -> {'FITS' if bl<base*0.8 else '*** CANNOT FIT (stalls at base) ***'}")
print("  (this asks: can a per-frame MLP on smooth (cos phi,sin phi) express a 1-FRAME-wide beat impulse?)")
