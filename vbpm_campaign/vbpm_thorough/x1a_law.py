"""X1(a): P1 replication on SMC -- tempo-increment law, held-out ll/step.
Families: gauss/laplace/t(2,3,5) x RW/OU, discretized 1-ms-bin likelihood (core.py verbatim).
Held-out = 5-fold CV BY SONG (deterministic). Plus transfer: main-corpus params scored on SMC.
"""
import sys, json, math
import numpy as np
sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough')
from core import prep, pairs, gather, fit_rw, score_rw
from smc_data import build_smc

D = prep(build_smc())
stems = sorted(d['stem'] for d in D)
fold_of = {s: i % 5 for i, s in enumerate(stems)}
P_all = pairs(D, 'all')
G_all = gather(P_all)
print(f'SMC songs {len(D)}  prediction pairs {G_all["n"]}')

FAMS = [('gauss', 0), ('laplace', 0), ('t', 2.0), ('t', 3.0), ('t', 5.0)]
res = {}
for fam, nu in FAMS:
    for ou in (False, True):
        nm = f"{fam}{'' if not nu else nu}{'_OU' if ou else '_RW'}"
        ll_ho = np.zeros(G_all['n']); filled = np.zeros(G_all['n'], bool)
        ths = []
        for f in range(5):
            tr_m = np.array([fold_of[s] != f for s in G_all['stem']])
            te_m = ~tr_m
            Gtr = {k: (v[tr_m] if isinstance(v, np.ndarray) else v) for k, v in G_all.items()}
            Gte = {k: (v[te_m] if isinstance(v, np.ndarray) else v) for k, v in G_all.items()}
            M = fit_rw(Gtr, fam, nu, ou)
            ll_ho[te_m] = score_rw(M, Gte); filled[te_m] = True
            ths.append([float(x) for x in M['th']])
        assert filled.all()
        M_full = fit_rw(G_all, fam, nu, ou)
        tr_ll = float(score_rw(M_full, G_all).mean())
        res[nm] = dict(cv_heldout=float(ll_ho.mean()), train_full=tr_ll,
                       th_full=[float(x) for x in M_full['th']], th_folds=ths)
        print(f'{nm:14s} CV-heldout {ll_ho.mean():+.4f}  full-fit train {tr_ll:+.4f} '
              f'params {np.round(M_full["th"], 4)}')

# transfer: main-corpus fitted params -> SMC
main = json.load(open('/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise/step1.json'))
print('\nTRANSFER main-corpus params -> all SMC pairs (ll/step):')
for nm, r in main.items():
    fam = 'gauss' if nm.startswith('gauss') else ('laplace' if nm.startswith('laplace') else 't')
    nu = float(nm[1:4]) if fam == 't' else 0
    M = dict(th=np.array(r['th']), ou=nm.endswith('_OU'), fam=fam, nu=nu)
    ll = float(score_rw(M, G_all).mean())
    res.setdefault('transfer', {})[nm] = ll
    print(f'  {nm:14s} main-eval {r["eval2"]:+.4f}  -> SMC {ll:+.4f}')
json.dump(res, open('x1a_law.json', 'w'), indent=1)
