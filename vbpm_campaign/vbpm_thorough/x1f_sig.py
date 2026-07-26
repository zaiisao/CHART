"""Significance of the audio gain on SMC: per-song paired comparison + song-level bootstrap
of pooled R2 difference, hist vs hist+audio-causal and hist vs hist+audio-full."""
import numpy as np, json
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
z = np.load('smc_feats.npz', allow_pickle=True)
Xf, Xc, Xh = z['Xf'].astype(np.float64), z['Xc'].astype(np.float64), z['Xh'].astype(np.float64)
stem = z['G_stem']; e = z['G_u'] - z['G_u_prev']
stems = sorted(set(stem)); fold_of = {s: i % 5 for i, s in enumerate(stems)}
fold = np.array([fold_of[s] for s in stem])
gbr = lambda: GBR(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=0)
def oof(X):
    p = np.zeros(len(e))
    for f in range(5):
        p[fold == f] = gbr().fit(X[fold != f], e[fold != f]).predict(X[fold == f])
    return p
P = {'hist': oof(Xh), 'hist+causal': oof(np.hstack([Xh, Xc])), 'hist+full': oof(np.hstack([Xh, Xf])),
     'causal': oof(Xc), 'full': oof(Xf)}
rng = np.random.default_rng(0)
song_idx = [np.where(stem == s)[0] for s in stems]
OUT = {}
for a, b in [('hist', 'hist+causal'), ('hist', 'hist+full'), ('hist', 'causal')]:
    sse_a = np.array([((e[ii]-P[a][ii])**2).sum() for ii in song_idx])
    sse_b = np.array([((e[ii]-P[b][ii])**2).sum() for ii in song_idx])
    sst = np.array([((e[ii]-e.mean())**2).sum() for ii in song_idx])
    dR2 = (sse_a.sum()-sse_b.sum())/sst.sum()
    boots = []
    for _ in range(4000):
        j = rng.integers(0, len(stems), len(stems))
        boots.append((sse_a[j].sum()-sse_b[j].sum())/sst[j].sum())
    boots = np.array(boots)
    lo_, hi_ = np.percentile(boots, [2.5, 97.5])
    win = float(np.mean(sse_b < sse_a))
    p_neg = float(np.mean(boots <= 0))
    OUT[f'{b} vs {a}'] = dict(dR2=float(dR2), ci=[float(lo_), float(hi_)], frac_songs_better=win, p_boot=p_neg)
    print(f'{b:12s} vs {a:5s}: dR2={dR2:+.4f}  95%CI [{lo_:+.4f},{hi_:+.4f}]  '
          f'songs improved {win:.3f}  P(dR2<=0) {p_neg:.4f}', flush=True)
json.dump(OUT, open('x1f_sig.json', 'w'), indent=1)
