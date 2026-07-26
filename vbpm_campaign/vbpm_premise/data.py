"""Shared data builder for the PREMISE-2 test: the bar-pointer transition law."""
import sys, math, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_final')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')
from emission import load_act, load_split
from vbpm.evaluate import _estimate_meter

FPS = 50.0
TWO_PI = 2*math.pi

def build(split):
    """Per song: beat times, meter, per-beat bar-rate u_k=log(phidot in rad/frame),
    increment e_k = u_k - u_{k-1}, beat-in-bar index, activation."""
    S = load_split(split); A = load_act(split)
    out = []
    for s in S:
        b = np.asarray(s['beats'], float); d = np.asarray(s['downs'], float)
        if len(b) < 8: continue
        m = int(_estimate_meter(b, d))
        a = A[s['stem']]; T = min(len(a), s['T'])
        b = b[(b >= 0) & (b < T/FPS)]
        if len(b) < 8: continue
        I = np.diff(b)                      # sec
        ok = I > 1e-3
        w = TWO_PI/(m*I*FPS)                # rad/frame bar advance implied by this IBI
        u = np.log(w)
        e = np.diff(u)                      # increment, index k -> u[k+1]-u[k]
        # beat-in-bar index of beat k (0 = downbeat)
        bib = np.zeros(len(b), int)
        if len(d) >= 1:
            for i, t in enumerate(b):
                prev = d[d <= t + 1e-6]
                if len(prev) == 0: bib[i] = -1
                else:
                    j = np.searchsorted(b, prev[-1] - 1e-6)
                    bib[i] = int(round(i - j))
        out.append(dict(stem=s['stem'], dataset=s['dataset'], meter=m, T=T,
                        beats=b, downs=d, act=a[:T], I=I, u=u, e=e, bib=bib,
                        ok=ok))
    return out

def stats(name, x):
    x = np.asarray(x, float)
    return (f"{name}: n={len(x)} mean={x.mean():+.5f} sd={x.std():.5f} "
            f"mad={np.abs(x-np.median(x)).mean():.5f} kurt={((x-x.mean())**4).mean()/max(x.var()**2,1e-30):.2f} "
            f"q[.01,.5,.99]={np.quantile(x,0.01):+.4f},{np.median(x):+.4f},{np.quantile(x,0.99):+.4f} "
            f"frac|e|>0.1={np.mean(np.abs(x)>0.1):.4f}")

if __name__ == '__main__':
    for sp in ('train','eval'):
        D = build(sp)
        e = np.concatenate([d['e'] for d in D])
        print(sp, 'songs', len(D), 'beats', sum(len(d['u']) for d in D))
        print('  ', stats('e (log-tempo incr / beat)', e))
        bpm = np.concatenate([60.0/d['I'] for d in D])
        print('   bpm median', np.median(bpm), 'q01', np.quantile(bpm,0.01), 'q99', np.quantile(bpm,0.99))
        fpb = np.concatenate([d['I']*FPS for d in D]); print('   frames/beat mean', fpb.mean())
