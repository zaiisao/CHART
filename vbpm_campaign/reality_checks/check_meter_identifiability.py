#!/usr/bin/env python
"""
Reality check: METER IDENTIFIABILITY (ELBO_for_DBN.md sec 3, 5.1, 5.4).

Paper model: the decoder emission (sec 5.4) is a BEAT-ONLY Bernoulli
b_t = sigma(NN(z_t, h)).  It emits a single binary BEAT indicator -- there is
NO downbeat / beat-in-bar channel in the generative likelihood.  Meter m_t is a
Categorical latent that is therefore only inferable INDIRECTLY, through whatever
structure the beat process carries.  Phase is explicitly decoupled from meter
("meter is inferred from the phase trajectory", sec 3).

THE ASSUMPTION UNDER TEST -- 'meter identifiability':
    beats-per-bar can be recovered from the BEAT PROCESS.  Because the emission
    is beat-only, the only observable the likelihood ever sees is the sequence
    of beat times.  So the sharp question is:

        From beat TIMING alone (inter-beat intervals; NO downbeat labels, NO
        audio), how identifiable is beats-per-bar?

If it is NOT recoverable from beat timing, the paper's beat-only emission gives
the meter latent nothing to latch onto on real data: q(m|b,h) can only draw
meter information from h (audio), never from the modelled beat likelihood, so the
generative meter mechanism is faithfully-vacuous.

METHOD
  truth : per-song beats-per-bar (bpb) from downbeat labels col1 (modal bar
          length between consecutive downbeats).  Keep 'clean' songs whose modal
          meter covers >= 90% of bars, and meter classes with enough support.
  features (TIMING ONLY, from col0 beat times -- col1 is NEVER shown to them):
      - coeff of variation of IBI
      - autocorrelation of normalized IBI at lags 1..8
      - period-M "agogic" strength eta^2_M for M=2..8 : fraction of IBI variance
        explained by grouping beats by (position mod M).  A genuine bar-level
        timing signature (downbeat lengthening/shortening) shows up here.
      - FFT magnitude of normalized IBI at period M=2..8
  oracle : RandomForest, stratified 5-fold CV WITHIN each dataset (and pooled),
           predict bpb from timing features.  Report accuracy & balanced accuracy
           vs MAJORITY-class chance.  This is the identifiability upper bound a
           trained inference net could reach from the beat likelihood alone.
  unsupervised: pred = argmax_M eta^2_M over the observed meter set -- does the raw
           timing periodicity even point at the right meter without any labels?
  ceiling (WITH downbeats): meter from col1 is exact (=1.000). Reported as contrast
           -- this is the information the beat-only emission THROWS AWAY.
"""
import os, glob, collections
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = "/home/sogang/jaehoon/VBPM/dataset_store/beat_this_annotations"
DATASETS_WITH_METER = ["ballroom", "beatles", "asap", "hjdb", "gtzan",
                       "hainsworth", "rwc"]
DATASETS_NO_METER = ["smc"]
MAXLAG = 8
RNG = 0


# ----------------------------- truth from downbeats -------------------------
def song_meter(path):
    """Return (bpb, frac_modal, n_beats) or None if no clean meter."""
    arr = np.loadtxt(path, ndmin=2)
    if arr.shape[1] < 2 or arr.shape[0] < 4:
        return None, arr
    idx = arr[:, 1].astype(int)
    db = np.where(idx == 1)[0]
    if len(db) < 2:
        return None, arr
    bl = np.diff(db)
    bl = bl[(bl >= 2) & (bl <= 12)]          # plausible bar lengths only
    if len(bl) == 0:
        return None, arr
    mode = collections.Counter(bl).most_common(1)[0][0]
    frac = collections.Counter(bl)[mode] / len(bl)
    return (int(mode), float(frac), arr.shape[0]), arr


# ----------------------------- timing-only features -------------------------
def autocorr(x, lag):
    if lag >= len(x):
        return 0.0
    a, b = x[:-lag], x[lag:]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def eta2_period(r, M):
    """Fraction of IBI variance explained by grouping index by (i mod M).
    Phase-invariant measure of period-M (bar-level) timing structure."""
    n = len(r)
    if n < 2 * M:
        return 0.0
    tot = r - r.mean()
    ss_tot = float((tot ** 2).sum())
    if ss_tot < 1e-12:
        return 0.0
    ss_between = 0.0
    for k in range(M):
        g = r[k::M]
        if len(g) == 0:
            continue
        ss_between += len(g) * (g.mean() - r.mean()) ** 2
    return float(ss_between / ss_tot)


def fft_mag(r, M):
    n = len(r)
    if n < 2 * M:
        return 0.0
    x = r - r.mean()
    freqs = np.fft.rfftfreq(n)
    mag = np.abs(np.fft.rfft(x))
    target = 1.0 / M
    j = int(np.argmin(np.abs(freqs - target)))
    denom = mag.sum() + 1e-12
    return float(mag[j] / denom)


def timing_features(arr):
    t = arr[:, 0]
    t = np.sort(t)
    ibi = np.diff(t)
    ibi = ibi[ibi > 1e-4]
    if len(ibi) < 2 * MAXLAG:
        return None
    med = np.median(ibi)
    r = ibi / (med + 1e-12)                  # tempo-normalized IBI
    feats = {}
    feats["cv"] = float(ibi.std() / (ibi.mean() + 1e-12))
    for lag in range(1, MAXLAG + 1):
        feats[f"ac{lag}"] = autocorr(r, lag)
    for M in range(2, MAXLAG + 1):
        feats[f"eta{M}"] = eta2_period(r, M)
        feats[f"fft{M}"] = fft_mag(r, M)
    return feats


# ----------------------------- dataset loading ------------------------------
def load_dataset(name):
    files = sorted(glob.glob(os.path.join(ROOT, name, "annotations", "beats", "*.beats")))
    rows = []
    for f in files:
        m, arr = song_meter(f)
        if m is None:
            continue
        bpb, frac, nb = m
        if frac < 0.90:                      # keep clean single-meter songs
            continue
        ft = timing_features(arr)
        if ft is None:
            continue
        rows.append((bpb, ft))
    return rows, len(files)


def to_xy(rows, keep_classes):
    keys = None
    X, y = [], []
    for bpb, ft in rows:
        if bpb not in keep_classes:
            continue
        if keys is None:
            keys = sorted(ft.keys())
        X.append([ft[k] for k in keys])
        y.append(bpb)
    return np.array(X), np.array(y), keys


# ----------------------------- oracle evaluation ----------------------------
def cv_oracle(X, y, seed=RNG):
    classes, counts = np.unique(y, return_counts=True)
    maj = counts.max() / counts.sum()
    n_splits = min(5, counts.min())
    if n_splits < 2 or len(classes) < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yp = np.empty_like(y)
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=300, random_state=seed,
                                     class_weight="balanced", n_jobs=-1)
        clf.fit(X[tr], y[tr])
        yp[te] = clf.predict(X[te])
    acc = accuracy_score(y, yp)
    bacc = balanced_accuracy_score(y, yp)
    return dict(acc=acc, bacc=bacc, maj=maj, n=len(y),
                classes=dict(zip(classes.tolist(), counts.tolist())),
                bacc_chance=1.0 / len(classes))


def unsup_periodicity(rows, keep_classes):
    """Label-free: predict meter = argmax_M eta^2_M over candidate meters."""
    cands = sorted(keep_classes)
    correct = tot = 0
    conf = collections.Counter()
    for bpb, ft in rows:
        if bpb not in keep_classes:
            continue
        scores = {M: ft.get(f"eta{M}", 0.0) for M in cands}
        pred = max(scores, key=scores.get)
        correct += int(pred == bpb)
        tot += 1
        conf[(bpb, pred)] += 1
    return (correct / tot if tot else float("nan")), tot, conf


# ----------------------------- main -----------------------------------------
def main():
    print("=" * 82)
    print("METER IDENTIFIABILITY reality check")
    print("(ELBO_for_DBN.md 5.4 beat-only Bernoulli emission -> meter must be")
    print(" inferred INDIRECTLY. Is beats-per-bar recoverable from beat TIMING alone?)")
    print("=" * 82)

    all_rows = []
    per_ds = {}
    for d in DATASETS_WITH_METER:
        rows, nfiles = load_dataset(d)
        per_ds[d] = rows
        all_rows.extend([(d, r) for r in rows])
        dist = collections.Counter(bpb for bpb, _ in rows)
        print(f"\n[{d}] files={nfiles}  clean-meter songs used={len(rows)}  "
              f"bpb dist={dict(sorted(dist.items()))}")

    # choose classes with enough pooled support (>=25 songs) for a fair oracle
    pooled_dist = collections.Counter(bpb for _, (bpb, _) in all_rows)
    keep = {c for c, n in pooled_dist.items() if n >= 25}
    print("\n" + "-" * 82)
    print(f"Pooled bpb distribution: {dict(sorted(pooled_dist.items()))}")
    print(f"Meter classes kept for the oracle (>=25 songs each): {sorted(keep)}")
    print("-" * 82)

    # ---------------- per-dataset oracle (timing-only) ----------------
    print("\n### PER-DATASET oracle: RandomForest on TIMING features, 5-fold CV")
    print("    (acc vs majority chance; balanced-acc vs uniform chance)")
    per_ds_results = {}
    for d in DATASETS_WITH_METER:
        rows = per_ds[d]
        # dataset-local classes with >=2 examples and present in global keep
        local = collections.Counter(bpb for bpb, _ in rows)
        cls = {c for c in local if local[c] >= 5 and c in keep}
        if len(cls) < 2:
            print(f"  [{d:11s}] single-meter corpus ({dict(sorted(local.items()))}) "
                  f"-> meter class DEGENERATE, oracle N/A (chance=1.000)")
            per_ds_results[d] = None
            continue
        X, y, _ = to_xy(rows, cls)
        res = cv_oracle(X, y)
        per_ds_results[d] = res
        if res is None:
            print(f"  [{d:11s}] insufficient per-class support for CV")
            continue
        usup, ut, _ = unsup_periodicity(rows, cls)
        gain = res["acc"] - res["maj"]
        print(f"  [{d:11s}] n={res['n']:4d} classes={res['classes']}  "
              f"acc={res['acc']:.3f} (chance {res['maj']:.3f}, +{gain:+.3f})  "
              f"bacc={res['bacc']:.3f} (chance {res['bacc_chance']:.3f})  "
              f"unsup-eta2={usup:.3f}")

    # ---------------- pooled oracle ----------------
    print("\n### POOLED oracle (all meter datasets together)")
    pooled_rows = [r for _, r in all_rows]
    X, y, keys = to_xy(pooled_rows, keep)
    res = cv_oracle(X, y)
    usup, ut, conf = unsup_periodicity(pooled_rows, keep)
    print(f"  n={res['n']}  classes(bpb:count)={res['classes']}")
    print(f"  MAJORITY-class chance      : {res['maj']:.4f}")
    print(f"  Oracle accuracy (timing)   : {res['acc']:.4f}   (gain over chance {res['acc']-res['maj']:+.4f})")
    print(f"  Oracle balanced-accuracy   : {res['bacc']:.4f}   (uniform chance {res['bacc_chance']:.4f})")
    print(f"  Unsupervised argmax eta^2  : {usup:.4f}   (label-free periodicity vote)")

    # feature importance to see WHAT (if anything) carries meter
    clf = RandomForestClassifier(n_estimators=400, random_state=RNG,
                                 class_weight="balanced", n_jobs=-1).fit(X, y)
    imp = sorted(zip(keys, clf.feature_importances_), key=lambda x: -x[1])[:6]
    print(f"  Top timing features        : " +
          ", ".join(f"{k}={v:.3f}" for k, v in imp))

    # ---------------- how strong is the bar-level timing signal at all? ------
    print("\n### DECISIVE STATISTIC: is there ANY period-M timing structure?")
    print("    eta^2_M = fraction of IBI variance explained by bar-position grouping.")
    print("    If ~0, beat timing is metronomic and carries no meter -> emission-blind.")
    for M in (2, 3, 4):
        vals = np.array([ft.get(f"eta{M}", 0.0) for _, (_, ft) in all_rows])
        print(f"    M={M}: mean eta^2={vals.mean():.4f}  median={np.median(vals):.4f}  "
              f"p90={np.percentile(vals,90):.4f}")
    # for the TRUE meter of each song, how much variance does its own period explain?
    own = np.array([ft.get(f"eta{bpb}", 0.0) for _, (bpb, ft) in all_rows
                    if bpb in keep])
    print(f"    eta^2 at each song's TRUE meter: mean={own.mean():.4f} "
          f"median={np.median(own):.4f}  (variance of IBI a bar-level agogic "
          f"accent would explain)")

    # 3-vs-4 head-to-head: the discrimination the meter latent most needs
    three_four = {c for c in (3, 4) if c in keep}
    if len(three_four) == 2:
        X2, y2, _ = to_xy(pooled_rows, three_four)
        r2 = cv_oracle(X2, y2)
        print(f"\n### 3-vs-4 head-to-head (the core meter distinction)")
        print(f"    n={r2['n']} {r2['classes']}  acc={r2['acc']:.4f} "
              f"(chance {r2['maj']:.4f})  bacc={r2['bacc']:.4f}")

    # ---------------- ceiling WITH downbeats ----------------
    print("\n### CONTRAST -- meter WITH downbeat labels (col1)")
    print("    Meter from col1 = count beats between consecutive downbeats = EXACT.")
    print("    Recoverability WITH downbeats = 1.000 by construction.")
    print("    -> The gap (1.000 - oracle-acc) is the meter information the paper's")
    print("       BEAT-ONLY emission structurally discards.")

    # ---------------- SMC: no annotation at all ----------------
    print("\n### SMC (target-style corpus): NO downbeat annotation")
    for d in DATASETS_NO_METER:
        files = sorted(glob.glob(os.path.join(ROOT, d, "annotations", "beats", "*.beats")))
        has = sum(1 for f in files if np.loadtxt(f, ndmin=2).shape[1] >= 2)
        print(f"    [{d}] {len(files)} files, {has} with col1 -> meter has NO ground "
              f"truth AND (per above) is not recoverable from timing.")

    print("\n" + "=" * 82)
    print("SUMMARY")
    print("=" * 82)
    print(f"  Beat-timing oracle acc {res['acc']:.3f} vs majority {res['maj']:.3f} "
          f"(gain {res['acc']-res['maj']:+.3f}); balanced {res['bacc']:.3f} "
          f"vs {res['bacc_chance']:.3f}.")
    print(f"  Bar-level agogic signal eta^2 at true meter ~ {own.mean():.3f} "
          f"(near-zero => metronomic beats).")


if __name__ == "__main__":
    main()
