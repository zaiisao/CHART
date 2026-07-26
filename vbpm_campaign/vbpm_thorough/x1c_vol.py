"""X1(c): raw volatility comparison SMC vs main corpus (ballroom+beatles+hainsworth).
Per-song increment kurtosis, lag-1 autocorr, tempo drift measures."""
import sys, json
import numpy as np
sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough')
from data import build
from smc_data import build_smc

def song_stats(d):
    e = d['e']; u = d['u']; I = d['I']
    if len(e) < 6: return None
    bpm = 60.0/I
    ac1 = float(np.corrcoef(e[:-1], e[1:])[0, 1]) if e.std() > 1e-9 else 0.0
    return dict(
        kurt=float(((e-e.mean())**4).mean()/max(e.var()**2, 1e-30)),
        ac1=ac1,
        sd_e=float(e.std()), mean_abs_e=float(np.abs(e).mean()),
        frac_big=float(np.mean(np.abs(e) > 0.1)),
        sd_u=float(u.std()),
        net_drift=float(abs(u[-1]-u[0])),
        bpm_relrange=float((np.quantile(bpm, .95)-np.quantile(bpm, .05))/np.median(bpm)),
        n=len(e))

def corpus(name, D):
    rows = [s for s in (song_stats(d) for d in D) if s]
    e = np.concatenate([d['e'] for d in D])
    pooled_kurt = float(((e-e.mean())**4).mean()/e.var()**2)
    # pooled lag-1 within songs
    a = np.concatenate([d['e'][:-1] for d in D if len(d['e']) > 6])
    b = np.concatenate([d['e'][1:] for d in D if len(d['e']) > 6])
    out = dict(n_songs=len(rows), n_incr=len(e), pooled_kurt=pooled_kurt,
               pooled_ac1=float(np.corrcoef(a, b)[0, 1]),
               pooled_mean_abs_e=float(np.abs(e).mean()),
               pooled_frac_big=float(np.mean(np.abs(e) > 0.1)))
    for k in rows[0]:
        if k == 'n': continue
        v = np.array([r[k] for r in rows])
        out[f'{k}_med'] = float(np.median(v)); out[f'{k}_p90'] = float(np.percentile(v, 90))
    print(f'\n== {name}: songs {out["n_songs"]} incr {out["n_incr"]} ==')
    print(f'  pooled: kurt {out["pooled_kurt"]:.1f}  lag1-ac {out["pooled_ac1"]:+.3f}  '
          f'mean|e| {out["pooled_mean_abs_e"]:.4f}  frac|e|>0.1 {out["pooled_frac_big"]:.4f}')
    print(f'  per-song median (p90): kurt {out["kurt_med"]:.1f} ({out["kurt_p90"]:.1f})  '
          f'ac1 {out["ac1_med"]:+.3f} ({out["ac1_p90"]:+.3f})  sd_e {out["sd_e_med"]:.4f} ({out["sd_e_p90"]:.4f})')
    print(f'  tempo drift: sd_u med {out["sd_u_med"]:.4f} (p90 {out["sd_u_p90"]:.4f})  '
          f'net|u_end-u_0| med {out["net_drift_med"]:.4f} (p90 {out["net_drift_p90"]:.4f})  '
          f'bpm rel-range med {out["bpm_relrange_med"]:.3f} (p90 {out["bpm_relrange_p90"]:.3f})')
    return out

RES = {}
Dm = build('train') + build('eval')
RES['main'] = corpus('MAIN (ballroom+beatles+hainsworth)', Dm)
for ds in ('ballroom', 'beatles', 'hainsworth'):
    RES[ds] = corpus(ds, [d for d in Dm if d['dataset'] == ds])
RES['smc'] = corpus('SMC', build_smc())
json.dump(RES, open('x1c_vol.json', 'w'), indent=1)
