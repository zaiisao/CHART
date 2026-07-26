"""Aggregate the 10 trained anchored runs into the final lambda-curve table."""
import json, math, glob, os
import numpy as np
HERE="/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough"
def row(tag):
    p=f"{HERE}/{tag}.json"
    if not os.path.exists(p): return None
    d=json.load(open(p))
    th=d.get("train_hist") or {}
    out={}
    for k in ("path","pf_meter_path"):
        s=d.get(k)
        if s: out[k]=s
    return dict(tag=tag, hist=th, res=d)
def show(tag,lam,law):
    r=row(tag)
    if r is None:
        print(f"{law:7s} lam={lam:<6} MISSING"); return None
    h=r["hist"]; s=r["res"].get("pf_meter_path") or r["res"].get("path")
    print(f"{law:7s} lam={str(lam):<6} beatF={s['beat_F']:.4f} blind={s['blind_best_offset']:.4f} "
          f"MARGIN={s['margin_over_blind']:+.4f} dbF={s['downbeat_F']:.4f} n_ratio={s['n_ratio']:.3f} "
          f"frac_neg={s['frac_neg']:.3f} | train-end: ANC={h.get('anchor',float('nan')):.2f} "
          f"ANCph/fr={h.get('anchor_phase_pf',float('nan')):.4f} g_psi={h.get('gamma_psi',float('nan')):.2e} "
          f"s_lv={h.get('s_lv',float('nan')):.2e} dof={h.get('tempo_dof',float('nan')):.2f} "
          f"fneg_prior={h.get('frac_neg_prior',float('nan')):.4f} sat={h.get('sat_frac',float('nan')):.2f}")
    return s
print("=== TRAINED anchored VAE, deploy PF K=300 a=1.0, 79 songs, sup emission (pf_meter_path) ===")
for law,pre in (("gauss","tg"),("student","ts")):
    for lam in (0,0.03,0.3,3,"inf"):
        tag=f"{pre}_inf" if lam=="inf" else f"{pre}_lam{lam}"
        show(tag,lam,law)
# per-song paired diff: best finite vs inf, each law
from scipy import stats
for law,pre in (("gauss","tg"),("student","ts")):
    best=None
    for lam in (0,0.03,0.3,3):
        r=row(f"{pre}_lam{lam}")
        if r is None: continue
        f=(r["res"].get("pf_meter_path") or {}).get("beat_F",float("nan"))
        if best is None or f>best[1]: best=(lam,f,r)
    ri=row(f"{pre}_inf")
    if best and ri:
        pb={x["stem"]:x["beat_F"] for x in best[2]["res"]["rows"]}
        pi={x["stem"]:x["beat_F"] for x in ri["res"]["rows"]}
        st=sorted(set(pb)&set(pi))
        d=np.array([pb[s]-pi[s] for s in st])
        rng=np.random.default_rng(0)
        bs=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(10000)])
        lo,hi=np.percentile(bs,[2.5,97.5])
        w=stats.wilcoxon(d)
        print(f"{law}: best finite lam={best[0]} vs inf: paired diff {d.mean():+.4f} "
              f"CI95=[{lo:+.4f},{hi:+.4f}] wilcoxon p={w.pvalue:.3f} n={len(d)}  "
              f"(NOTE rows are 'path' readout not pf_meter_path)")
