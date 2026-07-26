"""X1(b) THE KEY: history-alone vs audio-causal vs audio-full vs hist+audio prediction of
per-beat tempo increments on SMC, held-out BY SONG (5-fold deterministic GroupKFold).
Also: scale (log|e|) prediction, change-point classification, and main-corpus->SMC transfer.
Feature builders reused verbatim: vbpm_premise/feats.py (audio), histfeat.py (history).
"""
import sys, json, math
import numpy as np
sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.ensemble import HistGradientBoostingClassifier as GBC
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from core import prep, gather
from feats import song_feats
from histfeat import hist_feats
from smc_data import build_smc, FPS

z = np.load('smc_feats.npz', allow_pickle=True)
Xf = z['Xf'].astype(np.float64); Xc = z['Xc'].astype(np.float64); Xh = z['Xh'].astype(np.float64)
G = {k[2:]: z[k] for k in z.files if k.startswith('G_')}
G['n'] = len(G['u'])
e = G['u'] - G['u_prev']
stems = sorted(set(G['stem']))
fold_of = {s: i % 5 for i, s in enumerate(stems)}
fold = np.array([fold_of[s] for s in G['stem']])
print(flush=True) if False else print(f'SMC songs {len(stems)}  pairs {G["n"]}  Xfull {Xf.shape} Xcausal {Xc.shape} Xhist {Xh.shape}')

SETS = {'hist(audio-blind)': Xh, 'audio-causal': Xc, 'audio-full': Xf,
        'hist+audio-causal': np.hstack([Xh, Xc]), 'hist+audio-full': np.hstack([Xh, Xf])}

def cv_predict(model_fn, X, y):
    p = np.zeros(len(y))
    for f in range(5):
        m = model_fn().fit(X[fold != f], y[fold != f])
        p[fold == f] = m.predict(X[fold == f])
    return p

def cv_predict_proba(model_fn, X, y):
    p = np.zeros(len(y))
    for f in range(5):
        m = model_fn().fit(X[fold != f], y[fold != f])
        p[fold == f] = m.predict_proba(X[fold == f])[:, 1]
    return p

gbr = lambda: GBR(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=0)
gbc = lambda: GBC(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=0)
ridge = lambda: RidgeCV(alphas=np.logspace(-2, 4, 20))

OUT = {}
print('\n== MEAN prediction of increment e (held-out BY SONG, pooled R2) ==')
for nm, X in SETS.items():
    for mnm, fn in [('ridge', ridge), ('GBR', gbr)]:
        p = cv_predict(fn, X, e)
        r2 = 1 - ((e - p)**2).sum() / ((e - e.mean())**2).sum()
        mae = float(np.abs(e - p).mean())
        OUT[f'mean|{nm}|{mnm}'] = dict(R2=float(r2), mae=mae)
        print(flush=True) if False else print(f'  {nm:22s} {mnm:6s} R2={r2:+.4f}  MAE={mae:.5f}  (persistence MAE {np.abs(e).mean():.5f})')

print('\n== SCALE prediction log|e| (heteroscedastic; held-out BY SONG) ==')
le = np.log(np.abs(e) + 1e-4)
for nm, X in SETS.items():
    p = cv_predict(gbr, X, le)
    r2 = 1 - ((le - p)**2).sum() / ((le - le.mean())**2).sum()
    OUT[f'scale|{nm}'] = dict(R2=float(r2))
    print(flush=True) if False else print(f'  {nm:22s} R2={r2:+.4f}')

print('\n== CHANGE-POINT classification |e|>thr (held-out BY SONG, AUC) ==')
for thr in (0.05, 0.10):
    y = (np.abs(e) > thr).astype(int)
    print(flush=True) if False else print(f'  thr={thr}: positives {y.mean():.3f}')
    for nm, X in SETS.items():
        p = cv_predict_proba(gbc, X, y)
        auc = roc_auc_score(y, p)
        OUT[f'chg{thr}|{nm}'] = dict(auc=float(auc), pos_rate=float(y.mean()))
        print(flush=True) if False else print(f'    {nm:22s} AUC={auc:.4f}')

# ---- ms interpretation (next-beat placement), GBR audio-full vs persistence
p_full = cv_predict(gbr, SETS['audio-full'], e)
p_hist = cv_predict(gbr, SETS['hist(audio-blind)'], e)
Ip = (2*math.pi/(G['meter']*FPS))/np.exp(G['u_prev'])
In = (2*math.pi/(G['meter']*FPS))/np.exp(G['u'])
for tag, p in [('persistence', np.zeros_like(e)), ('hist', p_hist), ('audio-full', p_full)]:
    Ihat = Ip*np.exp(p)
    err = np.abs(In - Ihat)*1000
    OUT[f'ms|{tag}'] = dict(mean=float(err.mean()), median=float(np.median(err)),
                            p95=float(np.percentile(err, 95)))
    print(flush=True) if False else print(f'  next-beat |err| ms [{tag:11s}]: mean {err.mean():6.2f} median {np.median(err):6.2f} p95 {np.percentile(err,95):7.2f}')

# ---- TRANSFER: models trained on MAIN corpus (train fold) -> predict SMC
print('\n== TRANSFER main-corpus-trained GBM -> SMC (same feature builders) ==')
def L(mode, sp):
    d = np.load(f'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise/feat_{mode}_{sp}.npz', allow_pickle=True)
    G_ = {k: d[k] for k in d.files if k != 'X'}
    return d['X'].astype(np.float64), G_
def H(sp):
    return np.load(f'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise/hist_{sp}.npz', allow_pickle=True)['X'].astype(np.float64)
Xf_t, Gt = L('full', 'train'); Xc_t, _ = L('causal', 'train'); Xh_t = H('train')
et = Gt['u'] - Gt['u_prev']
TSETS = {'hist(audio-blind)': (Xh_t, Xh), 'audio-causal': (Xc_t, Xc), 'audio-full': (Xf_t, Xf),
         'hist+audio-causal': (np.hstack([Xh_t, Xc_t]), np.hstack([Xh, Xc])),
         'hist+audio-full': (np.hstack([Xh_t, Xf_t]), np.hstack([Xh, Xf]))}
for nm, (A_, B_) in TSETS.items():
    m = gbr().fit(A_, et)
    p = m.predict(B_)
    r2 = 1 - ((e - p)**2).sum() / ((e - e.mean())**2).sum()
    OUT[f'transfer|{nm}'] = dict(R2=float(r2), mae=float(np.abs(e - p).mean()))
    print(flush=True) if False else print(f'  {nm:22s} R2={r2:+.4f}  MAE={np.abs(e-p).mean():.5f}')

json.dump(OUT, open('x1b_pred.json', 'w'), indent=1)
print('DONE', flush=True)
