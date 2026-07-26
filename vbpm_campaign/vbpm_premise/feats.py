"""Audio features for the AUDIO-CONDITIONED transition (d).
Anchor = frame of beat k (the pointer's position when it must predict the coming interval I_k).
CAUSAL set uses only frames <= anchor; FULL set adds centred/look-ahead windows
(what a bidirectional frontend h_t actually makes available)."""
import numpy as np, math
from data import FPS, TWO_PI

def _ac(x, lo, hi):
    """normalised autocorrelation of x over integer lags [lo,hi]; returns (best_lag, best_val, curve)."""
    x = x - x.mean()
    n = len(x); d = float((x*x).sum()) + 1e-9
    hi = min(hi, n-4)
    if hi <= lo: return float('nan'), 0.0, None
    lags = np.arange(lo, hi+1)
    r = np.array([float((x[:n-l]*x[l:]).sum()) for l in lags]) / d
    j = int(np.argmax(r))
    return float(lags[j]), float(r[j]), (lags, r)

def _peaks(a, thr=0.05):
    m = (a[1:-1] >= a[:-2]) & (a[1:-1] > a[2:]) & (a[1:-1] > thr)
    return np.where(m)[0] + 1

def song_feats(d, W=300, mode='full'):
    """returns X [n_pred, F], and the list of k it corresponds to (k>=1, valid)."""
    a = np.asarray(d['act'], np.float64); T = len(a)
    b = d['beats']; m = d['meter']; A0 = math.log(TWO_PI/(m*FPS))
    pk = _peaks(a[:,0]); pkd = _peaks(a[:,1])
    n = len(d['u'])
    ks = [k for k in range(1,n) if d['valid'][k] and d['valid'][k-1]]
    X = []
    for k in ks:
        f = int(round(b[k]*FPS)); f = max(2, min(T-3, f))
        Lp = d['I'][k-1]*FPS                       # previous interval in frames
        lo, hi = max(6,int(0.55*Lp)), min(T-5,int(1.8*Lp)+2)
        gl, gh = 10, 130                           # global tempo search 0.2-2.6 s
        ft = []
        wins = [('c', max(0,f-W), f)]
        if mode=='full': wins.append(('n', max(0,f-W//2), min(T,f+W//2)))
        for tag, s0, s1 in wins:
            w = a[s0:s1,0]; wd = a[s0:s1,1]
            if len(w) < 30: ft += [0.0]*10; continue
            gL, gV, _ = _ac(w, gl, gh)
            lL, lV, _ = _ac(w, lo, hi)
            ft += [ (A0-math.log(max(gL,1)/FPS)) - d['u'][k-1] if gL==gL else 0.0, gV,
                    (A0-math.log(max(lL,1)/FPS)) - d['u'][k-1] if lL==lL else 0.0, lV,
                    float(w.mean()), float(w.max()), float(w.std()), float(wd.mean()) ]
            # local peak-interval estimate
            p = pk[(pk>=s0)&(pk<s1)]
            iv = np.diff(p)
            iv = iv[(iv>0.55*Lp)&(iv<1.8*Lp)]
            ft += [float(np.log(np.median(iv)/Lp)) if len(iv)>=2 else 0.0, float(len(iv))/max(len(p),1)]
        # look-ahead: nearest beat-activation peak after the anchor in the plausible window
        if mode=='full':
            nx = pk[(pk > f+0.55*Lp) & (pk < f+1.8*Lp)]
            if len(nx):
                j = nx[int(np.argmax(a[nx,0]))]
                ft += [float(np.log((j-f)/max(Lp,1e-6))), float(a[j,0])]
            else: ft += [0.0, 0.0]
            nd = pkd[(pkd > f) & (pkd < f+1.8*Lp*m)]
            ft += [float(a[f,1]), float(len(nd))]
        ft += [float(a[f,0]), float(d['u'][k-1]), float(m==2), float(m==3), float(m==4)]
        X.append(ft)
    return np.asarray(X, np.float32), ks

def build_feats(D, W=300, mode='full'):
    Xs=[]; 
    for d in D:
        X,ks = song_feats(d, W, mode)
        if len(ks): Xs.append(X)
    return np.concatenate(Xs,0) if Xs else np.zeros((0,1),np.float32)
