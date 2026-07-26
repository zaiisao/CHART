"""Context: frozen activation head peak-pick on SMC, with the MANDATORY density-matched
blind control (code verbatim from vbpm_final/run_exp2.py / vbpm_arms build_act_head)."""
import sys, json, math
import numpy as np
sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough')
from vbpm.evaluate import beats_from_activation, f_measure, metronome
from smc_data import build_smc, FPS

def blind_grid_controls(ref, T, n_est, n_off=12):
    dur = T / FPS
    if n_est < 2 or len(ref) < 2:
        return float("nan"), float("nan")
    per = dur / n_est
    base = np.arange(n_est) * per
    f0 = f_measure(ref, base)
    best = max(f_measure(ref, base + k * per / n_off) for k in range(n_off))
    return float(f0), float(max(best, f0))

rows = []
for d in build_smc():
    ref = d['beats']; T = d['T']
    est = beats_from_activation(d['act'][:, 0], FPS, thr=0.5, min_dist_sec=0.15)
    b0, bb = blind_grid_controls(ref, T, len(est))
    rows.append(dict(stem=d['stem'], beat_F=f_measure(ref, est), n_est=len(est), n_true=len(ref),
                     blind0=b0, blind_best=bb, metronome_F=f_measure(ref, metronome(T, FPS))))
M = lambda k: float(np.nanmean([r[k] for r in rows]))
n_ratio = sum(r['n_est'] for r in rows)/sum(r['n_true'] for r in rows)
s = dict(n_songs=len(rows), beat_F=M('beat_F'), n_ratio=n_ratio,
         blind_same_density=M('blind0'), blind_best_offset=M('blind_best'),
         margin_over_blind=M('beat_F')-M('blind_best'), metronome=M('metronome_F'))
print(json.dumps(s, indent=1))
json.dump(dict(summary=s, rows=rows), open('x1d_head.json', 'w'), indent=1)
