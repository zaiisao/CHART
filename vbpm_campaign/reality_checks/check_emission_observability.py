"""EMISSION_OBSERVABILITY reality check (ELBO_for_DBN.md 5.4: p(b_t | z_t, h)).

The paper's decoder reads the audio h and must emit the beat AT THE PHASE:
    b_hat_t = sigma( NN_theta(z_t, h_{1:T}) ).
For this emission assumption to be satisfied the WORLD must actually place an
observable acoustic cue (an onset) at each annotated beat. If beats are instead
carried by non-acoustic phrasing (expressive rubato, syncopation, implied pulse),
then h does NOT carry the beat and p(b_t|z_t,h) is misspecified: no decoder can
read a cue that is not in the signal, and the beat must be supplied by the
transition PRIOR (tempo continuity) instead.

We measure, per corpus, the beat-vs-background ONSET CONTRAST from the raw
librosa onset-strength envelope -- a frontend-agnostic proxy for "does the audio
provide an observable beat cue at the annotated phase".

Primary statistic: ROC-AUC of onset-strength discriminating beat frames from
background frames (0.5 = beat invisible in the audio; 1.0 = perfectly observable).
The SMC drift-beat finding was AUC ~0.76 (present but weak); we extend it.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
np.random.seed(0)

sys.path.insert(0, "/home/sogang/jaehoon/VBPM")
import librosa  # noqa: E402
from data.songs import iter_songs  # noqa: E402

SR = 22050
HOP = 512
FRAME_SEC = HOP / SR
TOL_FRAMES = 3            # +-3 frames ~ +-70 ms (standard beat tolerance)
MAX_PER_DS = 100
MAX_DUR = 60.0

SMC_AUDIO = Path("/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Audio")
SMC_BEATS = Path("/home/sogang/jaehoon/VBPM/dataset_store/beat_this_annotations/smc/annotations/beats")


def auc(pos, neg):
    """ROC-AUC via Mann-Whitney U with average ranks for ties."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def analyse_song(audio_path, beat_times, downbeat_times=None):
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True, duration=MAX_DUR)
    if y.size < SR:
        return None
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    env = env / (env.max() + 1e-9)
    n = len(env)
    bt = beat_times[(beat_times >= 0) & (beat_times < MAX_DUR)]
    if len(bt) < 4:
        return None
    lo = max(0, int(bt.min() / FRAME_SEC) - TOL_FRAMES)
    hi = min(n, int(bt.max() / FRAME_SEC) + TOL_FRAMES + 1)
    if hi - lo < 20:
        return None
    beat_frames = np.round(bt / FRAME_SEC).astype(int)
    beat_frames = beat_frames[(beat_frames >= lo) & (beat_frames < hi)]
    if len(beat_frames) < 4:
        return None
    # per-beat CUE = strongest onset within +-TOL of the annotated beat
    cue = np.array([env[max(0, f - TOL_FRAMES):min(n, f + TOL_FRAMES + 1)].max()
                    for f in beat_frames])
    # BACKGROUND = frames >=(TOL+2) from every beat, within the annotated span
    near = np.zeros(n, bool)
    for f in beat_frames:
        near[max(0, f - TOL_FRAMES - 2):min(n, f + TOL_FRAMES + 3)] = True
    span = np.zeros(n, bool)
    span[lo:hi] = True
    bg = env[span & ~near]
    if len(cue) < 4 or len(bg) < 4:
        return None
    bg75 = np.percentile(bg, 75)
    peakhit = float(np.mean(cue > bg75))     # beat clearly above the between-beat floor
    mu_p, mu_n = cue.mean(), bg.mean()
    # DOWNBEAT emission: is the downbeat acoustically distinct from ordinary beats?
    db_auc = np.nan
    if downbeat_times is not None and len(downbeat_times) >= 3:
        dbt = downbeat_times[(downbeat_times >= 0) & (downbeat_times < MAX_DUR)]
        dbf = set(np.round(dbt / FRAME_SEC).astype(int).tolist())
        db_cue, ob_cue = [], []
        for f, c in zip(beat_frames, cue):
            (db_cue if f in dbf or (f - 1) in dbf or (f + 1) in dbf else ob_cue).append(c)
        if len(db_cue) >= 3 and len(ob_cue) >= 3:
            db_auc = auc(db_cue, ob_cue)   # >0.5 => downbeat louder than other beats
    sd = np.sqrt(0.5 * (cue.var() + bg.var())) + 1e-9
    return dict(auc=auc(cue, bg),
                dprime=(mu_p - mu_n) / sd,
                ratio=mu_p / (mu_n + 1e-9),
                peakhit=peakhit,
                db_auc=db_auc,
                nbeat=len(beat_frames))


def sample(items, k):
    if len(items) <= k:
        return items
    idx = np.random.choice(len(items), k, replace=False)
    return [items[i] for i in sorted(idx)]


def collect_local(dataset):
    songs = iter_songs(datasets=[dataset])
    rows = []
    for s in sample(songs, MAX_PER_DS):
        bt, db = s.beats()
        try:
            r = analyse_song(s.audio_path, bt, db)
        except Exception:
            r = None
        if r:
            rows.append(r)
    return rows


def collect_smc():
    wavs = sorted(SMC_AUDIO.glob("*.wav"))
    rows = []
    for w in sample(wavs, MAX_PER_DS):
        bf = SMC_BEATS / f"{w.stem.lower()}.beats"
        if not bf.exists():
            continue
        bt = np.loadtxt(bf, ndmin=2)[:, 0]
        try:
            r = analyse_song(w, bt)
        except Exception:
            r = None
        if r:
            rows.append(r)
    return rows


def summarise(name, rows):
    if not rows:
        print(f"{name:12s}  NO DATA")
        return None
    a = np.array([r["auc"] for r in rows])
    d = np.array([r["dprime"] for r in rows])
    ra = np.array([r["ratio"] for r in rows])
    ph = np.array([r["peakhit"] for r in rows])
    dba = np.array([r["db_auc"] for r in rows], float)
    dba = dba[~np.isnan(dba)]
    dbstr = f"  db_vs_beat_AUC={dba.mean():.3f}(n={len(dba)})" if len(dba) else "  db_vs_beat_AUC=--(no col1)"
    print(f"{name:12s} n={len(rows):3d}  AUC={a.mean():.3f}+-{a.std():.3f} "
          f"[{np.percentile(a,10):.2f},{np.percentile(a,90):.2f}]  "
          f"d'={d.mean():+.2f}  ratio={ra.mean():.2f}x  peakhit={ph.mean():.2f}{dbstr}")
    return dict(name=name, n=len(rows), auc=a.mean(), auc_sd=a.std(),
                dprime=d.mean(), ratio=ra.mean(), peakhit=ph.mean())


if __name__ == "__main__":
    print(f"onset-contrast emission observability | sr={SR} hop={HOP} "
          f"tol=+-{TOL_FRAMES}f(~{TOL_FRAMES*FRAME_SEC*1000:.0f}ms) "
          f"<= {MAX_PER_DS} songs/ds, <= {MAX_DUR:.0f}s each\n")
    datasets = ["ballroom", "gtzan", "hjdb", "beatles", "hainsworth"]
    results = []
    for ds in datasets:
        results.append(summarise(ds, collect_local(ds)))
    results.append(summarise("smc", collect_smc()))
    results = [r for r in results if r]

    print()
    all_auc = np.concatenate([[r["auc"]] * r["n"] for r in results])
    print(f"POOLED  song-weighted mean AUC = {all_auc.mean():.3f}  "
          f"(range {min(r['auc'] for r in results):.3f}"
          f"..{max(r['auc'] for r in results):.3f})")

    print("\nobservability gradient (AUC, high=beat audible, low=beat implied):")
    for r in sorted(results, key=lambda x: -x["auc"]):
        verdict = ("STRONG  " if r["auc"] >= 0.85 else
                   "MODERATE" if r["auc"] >= 0.72 else
                   "WEAK    ")
        print(f"  {r['name']:12s} {r['auc']:.3f}  {verdict}")
