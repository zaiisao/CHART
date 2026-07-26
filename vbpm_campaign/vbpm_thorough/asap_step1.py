"""X2(a): tempo-increment law on ASAP, mirror of vbpm_premise/step1_family.py,
plus tempo-DRIFT magnitude (ASAP vs steady corpus)."""
import sys, numpy as np, math, json
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough')
from core import prep, pairs, gather, fit_rw, score_rw
from asap_data import build_asap
from data import build

Dtr = prep(build_asap('train')); Dev = prep(build_asap('eval'))
Gtr = gather(pairs(Dtr,'all'))
Gev2 = gather(pairs(Dev,'second'))
print('ASAP train pairs', Gtr['n'], 'eval-2ndhalf pairs', Gev2['n'], 'songs', len(set(Gev2['stem'])), flush=True)
res={}
for fam,nu in [('gauss',0),('laplace',0),('t',2.0),('t',3.0),('t',5.0)]:
    for ou in (False,True):
        M = fit_rw(Gtr, fam, nu, ou)
        tr = score_rw(M,Gtr).mean(); ev = score_rw(M,Gev2).mean()
        nm=f"{fam}{'' if not nu else nu}{'_OU' if ou else '_RW'}"
        res[nm]=dict(train=float(tr), eval2=float(ev), th=[float(x) for x in M['th']])
        print(f"{nm:14s} train {tr:+.4f} evalHO {ev:+.4f} params {np.round(M['th'],4)}", flush=True)
json.dump(res, open('asap_step1.json','w'), indent=1)

# ---- tempo-DRIFT magnitude: ASAP vs steady corpus ------------------------------
def drift_stats(D, name):
    out=[]
    for d in D:
        e=d['e']; I=d['I']; u=d['u']
        if len(e)<8: continue
        dur = d['beats'][-1]-d['beats'][0]
        out.append(dict(mabs_e=float(np.mean(np.abs(e))),
                        e_per_sec=float(np.sum(np.abs(e))/max(dur,1e-6)),
                        sd_u=float(np.std(u)), rng_u=float(u.max()-u.min()),
                        ac1=float(np.corrcoef(e[:-1],e[1:])[0,1]) if len(e)>10 else np.nan))
    agg={k: float(np.nanmedian([o[k] for o in out])) for k in out[0]}
    q90={k: float(np.nanquantile([o[k] for o in out],0.9)) for k in out[0]}
    print(f"{name:14s} n_songs={len(out)} median per-song: mean|e|={agg['mabs_e']:.4f} "
          f"sum|e|/sec={agg['e_per_sec']:.4f} sd(u)={agg['sd_u']:.4f} range(u)={agg['rng_u']:.4f} "
          f"lag1-autocorr(e)={agg['ac1']:+.3f}   [q90 mean|e|={q90['mabs_e']:.4f} sd(u)={q90['sd_u']:.4f}]", flush=True)
    return dict(median=agg, q90=q90, n=len(out))

DR={}
DR['asap_eval']  = drift_stats(Dev, 'ASAP eval')
DR['asap_train'] = drift_stats(Dtr, 'ASAP train')
Sev = prep(build('eval')); Str = prep(build('train'))
DR['steady_eval']  = drift_stats(Sev, 'steady eval')
DR['steady_train'] = drift_stats(Str, 'steady train')
json.dump(DR, open('asap_drift.json','w'), indent=1)
print('DONE')
