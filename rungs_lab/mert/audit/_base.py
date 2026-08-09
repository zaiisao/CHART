"""Extended octave-decision layer race (~600 songs: race 200 + era corpus 300 + era smc 217).

Candidates are label-free (BT beat-activation ACF best lag + its stronger octave partner);
label = which candidate the annotation matches. Balanced accuracy, 5-fold CV, per-layer.
Embeddings: song-level mean pool -- race npy (50 fps) and era .pt caches (86.13 fps) mix freely.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path("/home/sogang/jaehoon/VBPM/rungs_lab/mert")
LAB = Path("/home/sogang/jaehoon/VBPM/rungs_lab")
sys.path.insert(0, str(LAB))
from data.songs import iter_songs
from smc_data import load_smc

RACE = Path("/disk4/jaehoon/VBPM_cache/mert/race")
ERA = Path("/disk1/jaehoon/vbpm_mert_layers")

# ---- tempi ----
tempo_of, dataset_of = {}, {}
for s in iter_songs():
    bt, _ = s.beats()
    if len(bt) >= 8:
        tempo_of[s.stem] = 60.0 / float(np.median(np.diff(bt)))
        dataset_of[s.stem] = s.dataset
for e in load_smc():
    if len(e["beat_times"]) >= 8:
        tempo_of[e["stem"]] = 60.0 / float(np.median(np.diff(e["beat_times"])))
        dataset_of[e["stem"]] = "smc"

# ---- acts ----
acts = {}
for fn in ("bt_acts_beat.npz", "bt_acts_beat_v2.npz"):
    d = np.load(RACE / fn)
    FPS = float(d["fps"])
    for k in d.files:
        if k != "fps":
            acts[k] = d[k]

# ---- embeddings ----
emb = {}
for p in RACE.glob("*.npy"):
    emb[p.stem] = np.load(p).astype(np.float32).mean(axis=1)
for sub in ("corpus", "smc"):
    for p in (ERA / sub).glob("*.pt"):
        if p.stem in emb:
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        emb[p.stem] = d["layers"].float().mean(dim=1).numpy()

def acf_candidates(a, fps):
    a = a - a.mean()
    n = len(a)
    ac = np.correlate(a, a, "full")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(fps * 60 / 300), min(int(fps * 60 / 40), n - 1)
    lag = lo + int(np.argmax(ac[lo:hi + 1]))
    t1 = 60.0 * fps / lag
    part = {}
    for r in (0.5, 2.0):
        pl = int(round(lag * r))
        if lo <= pl <= hi:
            part[r] = ac[pl]
    r = max(part, key=part.get)
    return t1, 60.0 * fps / int(round(lag * r))

rows, X_all = [], []
for stem, a in acts.items():
    if stem not in tempo_of or stem not in emb:
        continue
    t1, t2 = acf_candidates(a, FPS)
    lo, hi = min(t1, t2), max(t1, t2)
    ta = tempo_of[stem]
    d_lo, d_hi = abs(np.log2(ta / lo)), abs(np.log2(ta / hi))
    if min(d_lo, d_hi) > np.log2(1.19):
        continue
    rows.append(dict(stem=stem, dataset=dataset_of[stem], lo=lo, hi=hi,
                     y=int(d_hi < d_lo), acf_pick_high=int(t1 == hi)))
    X_all.append(emb[stem])
X_all = np.stack(X_all)                                   # [n, 13, 768]
y = np.array([r["y"] for r in rows])
pick = np.array([r["acf_pick_high"] for r in rows])
gm = np.log2([np.sqrt(r["lo"] * r["hi"]) for r in rows])[:, None]
ds = np.array([r["dataset"] for r in rows])
acf_bal = balanced_accuracy_score(y, pick)
print(f"{len(rows)} trials  minority(n)={min((y==0).sum(), (y==1).sum())}  "
      f"truth=higher {y.mean():.2f}  ACF-pick balanced {acf_bal:.3f}", flush=True)
for d in np.unique(ds):
    m = ds == d
    print(f"  {d}: n={m.sum()} truth=higher {y[m].mean():.2f} "
          f"ACF-bal {balanced_accuracy_score(y[m], pick[m]):.3f}", flush=True)

def cv_balacc(X, yv, C=0.05, folds=5, seed=0):
    pred = np.zeros(len(yv))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, yv):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(
            sc.transform(X[tr]), yv[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return float(balanced_accuracy_score(yv, pred)), pred

base_gm, _ = cv_balacc(gm, y, C=1.0)
print(f"tempo-only baseline balanced {base_gm:.3f}", flush=True)
# --- CONFOUND TESTS ---
import numpy as np
uds = np.unique(ds)
oh = np.stack([(ds == d).astype(float) for d in uds], 1)
b1, p1 = cv_balacc(np.hstack([oh, gm]), y, C=1.0)
print(f"DATASET-ONEHOT + tempo baseline balanced {b1:.3f}", flush=True)
for d in uds:
    m = ds == d
    print(f"   ds={d} n={m.sum()} truth=higher {y[m].mean():.2f} onehot-pred-balacc {balanced_accuracy_score(y[m], p1[m]):.3f}")
L = 7
X = np.hstack([X_all[:, L], gm])
# within-dataset CV (no cross-dataset shortcut)
for d in uds:
    m = ds == d
    
    if m.sum()<30 or min((y[m]==0).sum(),(y[m]==1).sum())<10: print(f"   WITHIN-ds {d} SKIP degenerate: n={m.sum()} minority={min((y[m]==0).sum(),(y[m]==1).sum())}"); continue
    b, _ = cv_balacc(X[m], y[m])
    bt, _ = cv_balacc(gm[m], y[m], C=1.0)
    print(f"   WITHIN-ds {d} n={m.sum()} L7 balacc {b:.3f}  tempo-only {bt:.3f}")
# leave-one-dataset-out
for d in uds:
    te = ds == d; tr = ~te
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2: continue
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
    print(f"   LODO held-out {d} L7 balacc {balanced_accuracy_score(y[te], clf.predict(sc.transform(X[te]))):.3f}")
