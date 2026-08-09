"""Layer race: per-layer linear probes on song-level mean-pooled MERT features.

(a) tempo: ridge -> log2(annotated tempo), 5-fold CV, pooled Spearman.
(b) octave decision: ACF of the BT beat activation gives two label-free octave candidates
    (best ACF lag + its stronger octave partner); logistic probe [MERT-mean ; log2 gm(pair)]
    picks lower vs higher; CV accuracy vs the ACF-pick and tempo-only baselines.
(c) genre: gtzan 10-class logistic, stratified 4-fold CV accuracy.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

RACE = Path("/disk4/jaehoon/VBPM_cache/mert/race")
manifest = json.load(open(RACE / "manifest.json"))
acts = np.load(RACE / "bt_acts_beat.npz")
FPS = float(acts["fps"])

# ---- song-level embeddings [n, 13, 768] ----
songs, embs = [], []
for m in manifest:
    p = RACE / f"{m['stem']}.npy"
    if not p.exists():
        continue
    f = np.load(p).astype(np.float32)          # [13, T, 768]
    embs.append(f.mean(axis=1))                # [13, 768]
    songs.append(m)
E = np.stack(embs)
print(f"{len(songs)} songs embedded", flush=True)

# ---- octave candidates from activation ACF (label-free) ----
def acf_candidates(a, fps):
    a = a - a.mean()
    n = len(a)
    ac = np.correlate(a, a, "full")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(fps * 60 / 300), int(fps * 60 / 40)         # 300..40 bpm
    hi = min(hi, n - 1)
    lag = lo + int(np.argmax(ac[lo:hi + 1]))
    t1 = 60.0 * fps / lag
    part = {}
    for r in (0.5, 2.0):
        pl = int(round(lag * r))
        if lo <= pl <= hi:
            part[r] = ac[pl]
    r = max(part, key=part.get)
    t2 = 60.0 * fps / int(round(lag * r))
    return t1, t2

oct_rows = []                                  # (song_idx, t_low, t_high, y_high, acf_pick_high)
for i, m in enumerate(songs):
    if m["stem"] not in acts:
        continue
    t1, t2 = acf_candidates(acts[m["stem"]], FPS)
    lo, hi = min(t1, t2), max(t1, t2)
    ta = m["tempo_ann"]
    d_lo, d_hi = abs(np.log2(ta / lo)), abs(np.log2(ta / hi))
    if min(d_lo, d_hi) > np.log2(1.19):        # truth not in candidate set -> invalid trial
        continue
    oct_rows.append((i, lo, hi, int(d_hi < d_lo), int(t1 == hi)))
oct_rows = np.array(oct_rows, dtype=np.float64)
acf_bal = 0.5 * (np.mean(oct_rows[oct_rows[:,3]==1,4]==1) + np.mean(oct_rows[oct_rows[:,3]==0,4]==0))
print(f"octave trials: {len(oct_rows)}, truth=higher in {oct_rows[:,3].mean():.2f}, "
      f"ACF-pick balanced acc {acf_bal:.3f}", flush=True)

# tempo-only octave baseline (no MERT): logistic on log2 gm
gm = np.log2(np.sqrt(oct_rows[:, 1] * oct_rows[:, 2]))[:, None]
yo = oct_rows[:, 3].astype(int)
from sklearn.metrics import balanced_accuracy_score
def cv_logistic(X, y, folds=5, seed=0):
    """Pooled-CV BALANCED accuracy (class-weighted logistic): the minority class -- ACF picked
    the wrong octave -- is the failure mode under test, so plain accuracy is a class-prior trap."""
    pred, kf = np.zeros(len(y)), StratifiedKFold(folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced').fit(
            sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return float(balanced_accuracy_score(y, pred))
base_gm = cv_logistic(gm, yo)
print(f"tempo-only octave baseline: {base_gm:.3f}", flush=True)

# ---- per-layer probes ----
y_tempo = np.log2([m["tempo_ann"] for m in songs])
g_idx = [i for i, m in enumerate(songs) if m["genre"]]
y_genre = np.array([songs[i]["genre"] for i in g_idx])
oi = oct_rows[:, 0].astype(int)

results = []
for l in range(13):
    X = E[:, l]
    # (a) tempo
    preds = np.zeros(len(songs))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        sc = StandardScaler().fit(X[tr])
        r = Ridge(alpha=100.0).fit(sc.transform(X[tr]), y_tempo[tr])
        preds[te] = r.predict(sc.transform(X[te]))
    rho = spearmanr(preds, y_tempo).statistic
    # (b) octave
    Xo = np.hstack([X[oi], gm])
    acc_oct = cv_logistic(Xo, yo)
    # (c) genre
    Xg = X[g_idx]
    accs = []
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=0).split(Xg, y_genre):
        sc = StandardScaler().fit(Xg[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xg[tr]), y_genre[tr])
        accs.append(clf.score(sc.transform(Xg[te]), y_genre[te]))
    acc_gen = float(np.mean(accs))
    results.append(dict(layer=l, tempo_spearman=float(rho), octave_acc=acc_oct,
                        genre_acc=acc_gen))
    print(f"L{l:2d}  tempo rho {rho:.3f}  octave {acc_oct:.3f}  genre {acc_gen:.3f}", flush=True)

json.dump(dict(n_songs=len(songs), n_octave=len(oct_rows),
               baseline_acf_pick_balanced=float(acf_bal),
               baseline_tempo_only=base_gm, layers=results),
          open(Path(__file__).parent / "results_layer_race.json", "w"), indent=1)
print("DONE", flush=True)
