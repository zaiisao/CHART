#!/usr/bin/env python
"""
Reality check: DRIFT_RESIDUAL  (VBPM assumption: DETERMINISTIC PHASE ADVANCE)

Paper 3/5.2: beat phase advances by its mean  mu^p_phi,t = phi_{t-1} + phidot_{t-1}
with only small iid von Mises noise (kappa learned). Tempo phidot follows a per-frame
log-normal random walk (5.3). Net modeling claim: a smoothly-evolving tempo is the ONLY
driver of beat timing; everything left over is small, unstructured, iid noise.

For each song:
  - beat times t_1..t_N (col0 of the .beats file)
  - fit a SMOOTH tempo trajectory (Savitzky-Golay low-pass, poly2, win9): what an integrated
    random-walk tempo can represent (slow accel/decel).
  - RESIDUAL = actual beat time - smooth-tempo prediction == the drift/expressive part the
    deterministic-phase-advance model must dump into von Mises noise.

Statistics per dataset (steady: ballroom/hjdb vs expressive: asap/smc + breadth):
  R2_smooth : variance of beat deviation a smooth tempo explains (over constant-tempo baseline)
  resid_ms  : std of smooth-residual in ms == drift magnitude the model cannot capture
  resid_frac: residual var / constant-tempo-detrended beat var == variance RW-tempo FAILS to capture
  ac1_resid : lag-1 autocorr of residual. Model assumes iid (~0); nonzero => structured expressive timing
  ac1_dIBI  : lag-1 autocorr of IBI increments (2nd diff of beat time). Free random walk => ~0;
              strong negative => error-correcting expressive timing => RW misspecified
  kurt      : excess kurtosis of standardized residual (Gaussian=0); heavy tails => vM noise too thin
"""
import glob, os, warnings
import numpy as np
from scipy.signal import savgol_filter
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = "/home/sogang/jaehoon/VBPM/dataset_store/beat_this_annotations"
DATASETS = ["ballroom", "hjdb", "gtzan", "rwc", "beatles", "hainsworth", "asap", "smc"]
STEADY = {"ballroom", "hjdb"}
EXPRESSIVE = {"asap", "smc"}
MIN_BEATS = 20
SG_WIN = 9
SG_POLY = 2


def load_beats(path):
    a = np.loadtxt(path, ndmin=2)
    if a.size == 0:
        return None
    t = np.sort(a[:, 0].astype(float))
    t = t[np.isfinite(t)]
    t = t[np.concatenate([[True], np.diff(t) > 1e-6])]
    return t


def analyze_song(t):
    N = len(t)
    if N < MIN_BEATS:
        return None
    idx = np.arange(N, dtype=float)
    ibi = np.diff(t)
    med_ibi = np.median(ibi)
    if med_ibi <= 0:
        return None
    # --- octave / skipped-beat audit: IBIs far from the LOCAL median are gaps,
    #     not tempo dynamics. Flag their share so we know the residual is genuine
    #     expressive microtiming and not a handful of dropped-beat spikes.
    from numpy.lib.stride_tricks import sliding_window_view as _swv
    if len(ibi) >= 5:
        loc = np.median(_swv(np.pad(ibi, 2, mode="edge"), 5), axis=1)
    else:
        loc = np.full_like(ibi, med_ibi)
    ratio = ibi / loc
    gap_frac = float(np.mean((ratio > 1.5) | (ratio < 0.66)))

    b, a = np.polyfit(idx, t, 1)
    resid_const = t - (a + b * idx)
    win = min(SG_WIN, N if N % 2 == 1 else N - 1)
    if win < SG_POLY + 2:
        return None
    smooth = savgol_filter(t, window_length=win, polyorder=SG_POLY)
    resid_smooth = t - smooth
    v_const = np.var(resid_const)
    v_smooth = np.var(resid_smooth)
    if v_const < 1e-9:
        return None
    r2_smooth = 1.0 - v_smooth / v_const
    resid_frac = v_smooth / v_const
    resid_ms = np.std(resid_smooth) * 1000.0
    resid_rel = np.std(resid_smooth) / med_ibi
    # robust (MAD-based) residual scale: immune to a few gap/outlier beats.
    # If robust ~= std the residual is broad microtiming, not spikes.
    mad = np.median(np.abs(resid_smooth - np.median(resid_smooth)))
    resid_ms_rob = 1.4826 * mad * 1000.0
    resid_rel_rob = 1.4826 * mad / med_ibi

    def ac1(x):
        x = x - x.mean()
        d = np.dot(x, x)
        if d < 1e-12 or len(x) < 3:
            return np.nan
        return float(np.dot(x[:-1], x[1:]) / d)

    ac1_resid = ac1(resid_smooth)
    dibi = np.diff(ibi)
    ac1_dibi = ac1(dibi) if len(dibi) >= 4 else np.nan
    s = np.std(resid_smooth)
    kurt = float(stats.kurtosis(resid_smooth / s, fisher=True)) if s > 1e-9 else np.nan

    # --- Variance-ratio test (Lo-MacKinlay) on log-IBI: is tempo a random walk?
    #     RW  => VR(q)=1 (increments white, variance grows linearly).
    #     VR<1 => mean-reverting / bounded (tempo returns toward a level) -> RW too diffusive.
    #     VR>1 => persistent drift beyond a single step.
    logibi = np.log(ibi)
    d1 = np.diff(logibi)
    vr2 = np.nan
    if len(logibi) >= 6 and np.var(d1) > 1e-12:
        dq = logibi[2:] - logibi[:-2]
        vr2 = float(np.var(dq) / (2.0 * np.var(d1)))

    return dict(N=N, r2_smooth=r2_smooth, resid_frac=resid_frac, resid_ms=resid_ms,
                resid_rel=resid_rel, resid_ms_rob=resid_ms_rob, resid_rel_rob=resid_rel_rob,
                gap_frac=gap_frac, ac1_resid=ac1_resid, ac1_dibi=ac1_dibi, kurt=kurt, vr2=vr2)


def med(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)])
    return float(np.median(x)) if len(x) else float("nan")


def summarize(rows):
    keys = ["r2_smooth", "resid_frac", "resid_ms", "resid_rel", "resid_ms_rob",
            "resid_rel_rob", "gap_frac", "ac1_resid", "ac1_dibi", "kurt", "vr2"]
    out = {k: med([r[k] for r in rows]) for k in keys}
    out["n_songs"] = len(rows)
    out["med_N"] = med([r["N"] for r in rows])
    out["vr2_below1"] = float(np.mean([r["vr2"] < 0.9 for r in rows
                                       if np.isfinite(r["vr2"])])) if rows else float("nan")
    return out


def collect(ds):
    files = sorted(glob.glob(os.path.join(ROOT, ds, "annotations", "beats", "*.beats")))
    rows = []
    for f in files:
        t = load_beats(f)
        if t is None:
            continue
        r = analyze_song(t)
        if r is not None:
            rows.append(r)
    return rows


def main():
    hdr = (f"{'dataset':<12} {'songs':>5} {'medN':>5} {'resid_ms':>8} {'rob_ms':>7} "
           f"{'resid%bt':>8} {'rob%bt':>7} {'gap%':>6} {'ac1_res':>8} {'ac1_dIBI':>8} "
           f"{'kurt':>6} {'VR2':>6} {'VR2<1':>6}")
    print(hdr)
    print("-" * len(hdr))
    per, pooled, cache = {}, [], {}
    for ds in DATASETS:
        rows = collect(ds)
        cache[ds] = rows
        if not rows:
            print(f"{ds:<12} (no usable songs)")
            continue
        per[ds] = summarize(rows)
        pooled.extend(rows)
        s = per[ds]
        tag = "STEADY" if ds in STEADY else ("EXPRESS" if ds in EXPRESSIVE else "")
        print(f"{ds:<12} {s['n_songs']:>5} {s['med_N']:>5.0f} {s['resid_ms']:>8.1f} "
              f"{s['resid_ms_rob']:>7.1f} {s['resid_rel']*100:>7.1f}% {s['resid_rel_rob']*100:>6.1f}% "
              f"{s['gap_frac']*100:>5.1f}% {s['ac1_resid']:>8.3f} {s['ac1_dibi']:>8.3f} "
              f"{s['kurt']:>6.1f} {s['vr2']:>6.3f} {s['vr2_below1']*100:>5.0f}%  {tag}")
    ps = summarize(pooled)
    print("-" * len(hdr))
    print(f"{'POOLED':<12} {ps['n_songs']:>5} {ps['med_N']:>5.0f} {ps['resid_ms']:>8.1f} "
          f"{ps['resid_ms_rob']:>7.1f} {ps['resid_rel']*100:>7.1f}% {ps['resid_rel_rob']*100:>6.1f}% "
          f"{ps['gap_frac']*100:>5.1f}% {ps['ac1_resid']:>8.3f} {ps['ac1_dibi']:>8.3f} "
          f"{ps['kurt']:>6.1f} {ps['vr2']:>6.3f} {ps['vr2_below1']*100:>5.0f}%")

    gs = summarize([r for ds in STEADY for r in cache.get(ds, [])])
    ge = summarize([r for ds in EXPRESSIVE for r in cache.get(ds, [])])
    print("\n=== STEADY (ballroom+hjdb) vs EXPRESSIVE (asap+smc) ===")
    for nm, g in [("STEADY", gs), ("EXPRESSIVE", ge)]:
        print(f"  {nm:<11} songs={g['n_songs']:>4}  resid_ms={g['resid_ms']:>6.1f} "
              f"(robust {g['resid_ms_rob']:>5.1f})  resid%beat={g['resid_rel']*100:>5.1f}% "
              f"(robust {g['resid_rel_rob']*100:>4.1f}%)  gap%={g['gap_frac']*100:>4.1f}  "
              f"kurt={g['kurt']:>5.1f}  VR2={g['vr2']:>.3f}")
    if np.isfinite(gs["resid_ms"]) and gs["resid_ms"] > 0:
        print(f"  --> expressive drift residual is {ge['resid_ms']/gs['resid_ms']:.1f}x steady (std), "
              f"{ge['resid_ms_rob']/gs['resid_ms_rob']:.1f}x steady (robust/MAD).")

    print(f"\nHEADLINE: pooled median residual = {ps['resid_ms']:.1f} ms "
          f"({ps['resid_rel']*100:.1f}% of a beat); smooth tempo leaves {ps['resid_frac']*100:.0f}% "
          f"of constant-tempo-detrended beat variance unexplained.")
    print(f"         residual lag-1 autocorr (pooled) = {ps['ac1_resid']:.3f} "
          f"(0=>iid noise as model assumes; <0=>structured expressive timing).")
    print(f"         IBI-increment lag-1 autocorr (pooled) = {ps['ac1_dibi']:.3f} "
          f"(0=>free random walk as paper assumes; strongly <0=>misspecified).")


if __name__ == "__main__":
    main()
