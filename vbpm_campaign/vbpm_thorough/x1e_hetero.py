"""X1(e): V1(b-iii) analogue ON SMC -- heteroscedastic / mean-conditioned Student-t(nu=2)
transition, exact discretized 1-ms-bin log-mass, held-out BY SONG (5-fold outer CV,
inner 4-fold GroupKFold for OOF calibration -- fully honest, test songs never touched).
mu = u_prev + c (+ w*mhat) ;  s = exp(a + b*ghat) ; ghat/mhat = GBM predictors.
"""
import sys, json, math, os
import numpy as np
sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
from core import prep, gather, logmass
from feats import song_feats
from histfeat import hist_feats
from smc_data import build_smc

NU = 2.0
gbm = lambda: GBR(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=0)
FC = 'smc_feats.npz'

if os.path.exists(FC):
    z = np.load(FC, allow_pickle=True)
    Xf, Xc, Xh = z['Xf'], z['Xc'], z['Xh']
    G = {k[2:]: z[k] for k in z.files if k.startswith('G_')}
    G['n'] = len(G['u'])
else:
    D = prep(build_smc())
    Xf_, Xc_, P = [], [], []
    for d in D:
        Xa, ks = song_feats(d, 300, 'full')
        Xb, _ = song_feats(d, 300, 'causal')
        if not len(ks): continue
        Xf_.append(Xa); Xc_.append(Xb); P += [(d, k) for k in ks]
    Xf = np.concatenate(Xf_, 0).astype(np.float64)
    Xc = np.concatenate(Xc_, 0).astype(np.float64)
    Xh, _ = hist_feats(D)
    G = gather(P)
    np.savez_compressed(FC, Xf=Xf, Xc=Xc, Xh=Xh,
                        **{f'G_{k}': v for k, v in G.items() if isinstance(v, np.ndarray)})

e = G['u'] - G['u_prev']; le = np.log(np.abs(e) + 1e-4)
up, lo, hi = G['u_prev'], G['ulo'], G['uhi']
stems = sorted(set(G['stem'])); fold_of = {s: i % 5 for i, s in enumerate(stems)}
fold = np.array([fold_of[s] for s in G['stem']])
print(f'pairs {G["n"]} songs {len(stems)}')

def fit_base(m):
    def nll(th):
        c, ls = th
        return -logmass(lo[m], hi[m], up[m]+c, math.exp(ls), 't', NU).mean()
    return minimize(nll, [0.0, math.log(0.06)], method='Nelder-Mead',
                    options=dict(maxiter=4000, xatol=1e-8, fatol=1e-10)).x

SETS = {'hist': Xh, 'audio-causal': Xc, 'audio-full': Xf, 'hist+audio-full': np.hstack([Xh, Xf])}
OUT = {}
ll_base = np.zeros(G['n'])
for f in range(5):
    tr, te = fold != f, fold == f
    c, ls = fit_base(tr)
    ll_base[te] = logmass(lo[te], hi[te], up[te]+c, math.exp(ls), 't', NU)
OUT['baseline'] = float(ll_base.mean())
print(f'baseline t2 fixed-scale heldout-by-song: {ll_base.mean():+.4f}')

for nm, X in SETS.items():
    ll_s = np.zeros(G['n']); ll_m = np.zeros(G['n']); ll_ms = np.zeros(G['n'])
    for f in range(5):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        # inner OOF on training folds
        g_oof = np.zeros(len(tr)); m_oof = np.zeros(len(tr))
        gkf = GroupKFold(n_splits=4)
        for itr, iva in gkf.split(X[tr], e[tr], groups=G['stem'][tr]):
            g_oof[iva] = gbm().fit(X[tr][itr], le[tr][itr]).predict(X[tr][iva])
            m_oof[iva] = gbm().fit(X[tr][itr], e[tr][itr]).predict(X[tr][iva])
        g_te = gbm().fit(X[tr], le[tr]).predict(X[te])
        m_te = gbm().fit(X[tr], e[tr]).predict(X[te])
        lo_t, hi_t, up_t = lo[tr], hi[tr], up[tr]
        lo_e, hi_e, up_e = lo[te], hi[te], up[te]
        c0, ls0 = fit_base(fold != f)
        def nll_s(th):
            c, a, b = th
            return -logmass(lo_t, hi_t, up_t+c, np.exp(a+b*g_oof), 't', NU).mean()
        r = minimize(nll_s, [c0, ls0, 0.0], method='Nelder-Mead',
                     options=dict(maxiter=6000, xatol=1e-8, fatol=1e-10)).x
        ll_s[te] = logmass(lo_e, hi_e, up_e+r[0], np.exp(r[1]+r[2]*g_te), 't', NU)
        def nll_m(th):
            c, w, ls = th
            return -logmass(lo_t, hi_t, up_t+c+w*m_oof, math.exp(ls), 't', NU).mean()
        rm = minimize(nll_m, [c0, 1.0, ls0], method='Nelder-Mead',
                      options=dict(maxiter=6000, xatol=1e-8, fatol=1e-10)).x
        ll_m[te] = logmass(lo_e, hi_e, up_e+rm[0]+rm[1]*m_te, math.exp(rm[2]), 't', NU)
        def nll_ms(th):
            c, w, a, b = th
            return -logmass(lo_t, hi_t, up_t+c+w*m_oof, np.exp(a+b*g_oof), 't', NU).mean()
        rms = minimize(nll_ms, [rm[0], rm[1], r[1], r[2]], method='Nelder-Mead',
                       options=dict(maxiter=8000, xatol=1e-8, fatol=1e-10)).x
        ll_ms[te] = logmass(lo_e, hi_e, up_e+rms[0]+rms[1]*m_te, np.exp(rms[2]+rms[3]*g_te), 't', NU)
    OUT[nm] = dict(scale=float(ll_s.mean()), mean=float(ll_m.mean()), mean_scale=float(ll_ms.mean()),
                   d_scale=float(ll_s.mean()-ll_base.mean()), d_mean=float(ll_m.mean()-ll_base.mean()),
                   d_mean_scale=float(ll_ms.mean()-ll_base.mean()))
    print(f'{nm:18s} scale {ll_s.mean():+.4f} (d {ll_s.mean()-ll_base.mean():+.4f})  '
          f'mean {ll_m.mean():+.4f} (d {ll_m.mean()-ll_base.mean():+.4f})  '
          f'mean+scale {ll_ms.mean():+.4f} (d {ll_ms.mean()-ll_base.mean():+.4f})', flush=True)
json.dump(OUT, open('x1e_hetero.json', 'w'), indent=1)
