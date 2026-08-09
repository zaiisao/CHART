"""LOAD-BEARING TEST: does the exact activation likelihood carry tempo information at all?

Per song, replace the initial-state prior with a one-hot on EACH of the 36 tempo bins and
record the NLL profile. If the annotated bin ranks near the top, the objective DOES prefer
the true tempo (and the prior head / 1/T leverage is to blame for the nulls). If the rank is
~uniform, the objective is tempo-flat and no conditioning input could ever be rewarded.
Also run at a SHORT crop (T=150) to test the 1/T-suppression fix directly.
"""
import sys, numpy as np, torch
from pathlib import Path
M = Path("/home/sogang/jaehoon/VBPM/rungs_lab/mert"); sys.path.insert(0, str(M)); sys.path.insert(0, str(M.parent))
from mert_r4_model import R4Conditioned, UNIFORM_FLOOR

DEV = "cuda:1"; FPS = 44100 / 1024
CROP = int(sys.argv[1]) if len(sys.argv) > 1 else 1400
NSONG = int(sys.argv[2]) if len(sys.argv) > 2 else 60
c = torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt", weights_only=False)
mean, std = c["feat_mean"].to(DEV), c["feat_std"].to(DEV)
mm, ms = c["mert_mean"].to(DEV), c["mert_std"].to(DEV)
ck = torch.load(M / "runs/mertr4_mertfull_bestsel.pt", weights_only=False)
m = R4Conditioned(fps=FPS, input_mode="featsmert", device=DEV, input_dim=ck["input_dim"])
m.load_state_dict(ck["model"]); m.eval()
bpm = 60.0 * FPS / (m._min_interval + np.arange(m.num_tempi))

def mll(a, ti, po):
    dens = m.chassis.log_class_densities(a); lp, lk, _ = m.head_outputs(ti)
    li = m.conditioned_log_inits(po)
    per = [dp.forward_log_likelihood(i, lk, dens, state_to_class=s)
           for dp, i, s in zip(m.chassis.dynamic_programs, li, m.chassis.state_to_classes)]
    return float(torch.logsumexp(torch.stack(per), 0)) / a.shape[0]

ranks, spans, tops = [], [], []
with torch.no_grad():
    for e in c["val_entries"][:NSONG]:
        s = e["stem"]; a = c["val_acts"][s]; f = c["val_feats"][s].astype(np.float32); mt = c["val_mert"][s].astype(np.float32)
        L = a.shape[0]
        if L > CROP:
            st = (L - CROP) // 2; a, f, mt = a[st:st+CROP], f[st:st+CROP], mt[st:st+CROP]
        a = torch.from_numpy(a).to(DEV)
        ti = torch.cat([(torch.from_numpy(f).to(DEV) - mean) / std, (torch.from_numpy(mt).to(DEV) - mm) / ms], 1)
        ibi = np.diff(e["beat_times"]); ibi = ibi[ibi > 1e-3]
        if len(ibi) < 2: continue
        j_true = int(np.abs(bpm - 60.0 / np.median(ibi)).argmin())
        prof = np.zeros(m.num_tempi)
        for j in range(m.num_tempi):
            p = torch.full((m.num_tempi,), UNIFORM_FLOOR / m.num_tempi, device=DEV)
            p[j] += 1 - UNIFORM_FLOOR
            prof[j] = mll(a, ti, p.log())
        order = np.argsort(-prof)                      # best (highest LL) first
        ranks.append(int(np.where(order == j_true)[0][0]) + 1)
        spans.append(float(prof.max() - prof.min()))
        tops.append(int(order[0] == j_true))
ranks = np.array(ranks)
print(f"CROP={CROP} n={len(ranks)} bins={m.num_tempi}")
print(f"  annotated-bin rank: median {np.median(ranks):.0f}  mean {ranks.mean():.1f}  (uniform expectation {(m.num_tempi+1)/2:.1f})")
print(f"  top-1 {np.mean(tops):.1%}  top-3 {np.mean(ranks<=3):.1%}  top-5 {np.mean(ranks<=5):.1%}  (chance top-3 {3/m.num_tempi:.1%})")
print(f"  NLL span across bins: median {np.median(spans):.5f} nats/frame  (log36/T = {np.log(36)/CROP:.5f})")
