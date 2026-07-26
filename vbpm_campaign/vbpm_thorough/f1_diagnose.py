"""STRAND F1 -- diagnose the PF-vs-peak-pick inversion (0.7505 vs 0.8115).

Decomposition:
  (a) METER   : PF free (read-out with own meter = champion) vs PF free read-out GT meter
                vs PF with meter CLAMPED to GT (one-hot prior, p_switch=0).
                Subsets: meter-correct vs meter-wrong songs.
  (b) TIMING  : signed offset of PF beats vs nearest GT beat and vs nearest activation
                peak; same for peak-pick; SNAP experiment (move each PF beat to the
                nearest activation peak within 70ms, rescore).
  (c) SCATTER : per-song F(PF)-F(peak) with meter/octave/contrast/duration covariates.
  (d) CONTINUITY: mir_eval CMLc/CMLt/AMLc/AMLt (5s trim per mir_eval convention).

Every aggregate beat_F carries n_est/n_true and the density-matched blind control.
Nothing outside vbpm_thorough/ is modified.
"""
from __future__ import annotations
import json, math, sys, time
import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")
from emission import (PhaseEmission, load_act, load_split, obs_contrast, song_phase,
                      METERS, TWO_PI, FPS, _estimate_meter)
from pf import particle_filter
from vbpm.evaluate import (beats_from_barphase, downbeats_from_barphase,
                           beats_from_activation, f_measure)
import mir_eval

# ---- MANDATORY density-matched blind control (verbatim from vbpm_final/run_exp2.py) ----
def blind_grid_controls(ref, T, n_est, n_off=12):
    dur = T / FPS
    if n_est < 2 or len(ref) < 2:
        return float("nan"), float("nan")
    per = dur / n_est
    base = np.arange(n_est) * per
    f0 = f_measure(ref, base)
    best = max(f_measure(ref, base + k * per / n_off) for k in range(n_off))
    return float(f0), float(max(best, f0))


def continuity(ref, est):
    if len(ref) < 2 or len(est) < 2:
        return dict(CMLc=0.0, CMLt=0.0, AMLc=0.0, AMLt=0.0)
    r = mir_eval.beat.trim_beats(np.asarray(ref, float))
    e = mir_eval.beat.trim_beats(np.asarray(est, float))
    if len(r) < 2 or len(e) < 2:
        return dict(CMLc=0.0, CMLt=0.0, AMLc=0.0, AMLt=0.0)
    c = mir_eval.beat.continuity(r, e)
    return dict(CMLc=float(c[0]), CMLt=float(c[1]), AMLc=float(c[2]), AMLt=float(c[3]))


def offsets_to(est, targets, win=0.35):
    """signed offset (est - nearest target) for each est beat, |off|<=win kept."""
    if len(est) == 0 or len(targets) == 0:
        return np.array([])
    t = np.asarray(targets, float)
    j = np.searchsorted(t, est)
    j = np.clip(j, 1, len(t) - 1)
    lo, hi = t[j - 1], t[np.clip(j, 0, len(t) - 1)]
    off = np.where(np.abs(est - lo) <= np.abs(est - hi), est - lo, est - hi)
    return off[np.abs(off) <= win]


def snap_to_peaks(est, peaks, win=0.07, min_dist=0.10):
    """Move each est beat to the nearest activation peak within `win`; dedupe."""
    if len(est) == 0:
        return est
    out = []
    pk = np.asarray(peaks, float)
    for e in est:
        if len(pk):
            d = np.abs(pk - e)
            i = int(d.argmin())
            out.append(pk[i] if d[i] <= win else e)
        else:
            out.append(e)
    out = np.unique(np.round(np.asarray(out), 4))
    keep = [out[0]]
    for x in out[1:]:
        if x - keep[-1] >= min_dist:
            keep.append(x)
    return np.asarray(keep)


def octave_ratio(est, ref):
    """median-IBI ratio est/true (1.0 = same tempo; 2 = est twice as fast)."""
    if len(est) < 3 or len(ref) < 3:
        return float("nan")
    return float(np.median(np.diff(ref)) / np.median(np.diff(est)))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=0)
    ap.add_argument("--tag", default="f1_diag")
    a = ap.parse_args()

    K, ALPHA, SIGMA, SPHI, PSW, SEED = 600, 0.25, 0.05, 0.03, 0.005, 1234
    tr = load_split("train"); at = load_act("train")
    ev, ae = load_split("eval"), load_act("eval")
    if a.n_eval:
        ev = ev[:a.n_eval]
    print(f"train {len(tr)}  eval {len(ev)}", flush=True)

    t0 = time.time()
    emis = PhaseEmission(bins_per_beat=24, likelihood="gauss", smooth=0.0).fit(
        tr, at, phase_mode="downbeat")
    print(f"emission fitted ({time.time()-t0:.1f}s)", flush=True)

    prior = np.zeros(5)
    for s in tr:
        m = _estimate_meter(s["beats"], s["downs"])
        if m in METERS:
            prior[m] += 1

    rows = []
    t0 = time.time()
    for i, s in enumerate(ev):
        act = ae.get(s["stem"])
        if act is None:
            continue
        T = min(len(act), s["T"])
        ref = s["beats"][s["beats"] < T / FPS]
        dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 3:
            continue
        m_gt = _estimate_meter(s["beats"], s["downs"])
        LL = emis.padded_table(act[:T])

        # --- run 1: FREE PF (identical to FINAL_eval champion) ---
        out = particle_filter(LL, emis.nb, K=K, alpha=ALPHA, sigma_lt=SIGMA,
                              sigma_phi=SPHI, p_switch=PSW, meter_prior=prior,
                              fps=FPS, seed=SEED + i)
        m_pf = int(np.bincount(out["meter_path"]).argmax())
        est_free = beats_from_barphase(out["phase_path"], m_pf, FPS)      # champion
        est_gtm  = beats_from_barphase(out["phase_path"], m_gt, FPS)      # GT-meter read-out

        # --- run 2: meter CLAMPED to GT ---
        oh = np.zeros(5); oh[m_gt] = 1.0
        out_c = particle_filter(LL, emis.nb, K=K, alpha=ALPHA, sigma_lt=SIGMA,
                                sigma_phi=SPHI, p_switch=0.0, meter_prior=oh,
                                fps=FPS, seed=SEED + i)
        est_clamp = beats_from_barphase(out_c["phase_path"], m_gt, FPS)

        # --- peak-pick baseline + activation peaks for snapping/offsets ---
        peaks = beats_from_activation(act[:T, 0], FPS)
        est_pk = peaks

        # --- snap experiments ---
        est_free_sn  = snap_to_peaks(est_free, peaks)
        est_clamp_sn = snap_to_peaks(est_clamp, peaks)

        # per-song contrast
        c1, _ = obs_contrast(emis, [s], {s["stem"]: act}, phase_mode="downbeat")

        variants = dict(pf_free=est_free, pf_gtmeter=est_gtm, pf_clamp=est_clamp,
                        pf_free_snap=est_free_sn, pf_clamp_snap=est_clamp_sn,
                        peakpick=est_pk)
        r = dict(stem=s["stem"], dataset=s["dataset"], T=T, dur=T / FPS,
                 n_true=len(ref), n_true_db=len(dref), m_gt=m_gt, m_pf=m_pf,
                 meter_ok=int(m_pf == m_gt), contrast=c1,
                 ess=out["ess"], ess_clamp=out_c["ess"])
        for k, e in variants.items():
            b0, bb = blind_grid_controls(ref, T, len(e))
            r[k] = dict(F=f_measure(ref, e), n_est=len(e), blind0=b0, blind_best=bb,
                        oct=octave_ratio(e, ref), **continuity(ref, e))
        # downbeats (champion + peak-pick db channel), for context
        dest_free = downbeats_from_barphase(out["phase_path"], FPS)
        dest_pk = beats_from_activation(act[:T, 1], FPS, min_dist_sec=0.30)
        r["db_pf_F"] = f_measure(dref, dest_free) if len(dref) >= 2 else float("nan")
        r["db_pk_F"] = f_measure(dref, dest_pk) if len(dref) >= 2 else float("nan")

        # timing offsets (seconds): est vs GT, est vs activation peaks
        r["off_pf_gt"]  = offsets_to(est_free, ref).tolist()
        r["off_pk_gt"]  = offsets_to(est_pk, ref).tolist()
        r["off_pf_pk"]  = offsets_to(est_free, peaks).tolist()
        rows.append(r)
        if (len(rows) % 20) == 0:
            print(f"  {len(rows)} songs ({time.time()-t0:.0f}s)", flush=True)

    print(f"PF done {len(rows)} songs ({time.time()-t0:.0f}s)", flush=True)
    json.dump(rows, open(f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough/{a.tag}_rows.json", "w"),
              default=float)

    # ---------------- aggregate ----------------
    def agg(key):
        F = np.mean([r[key]["F"] for r in rows])
        ne = sum(r[key]["n_est"] for r in rows); nt = sum(r["n_true"] for r in rows)
        bb = np.nanmean([r[key]["blind_best"] for r in rows])
        cml = np.mean([r[key]["CMLt"] for r in rows])
        aml = np.mean([r[key]["AMLt"] for r in rows])
        cmc = np.mean([r[key]["CMLc"] for r in rows])
        return dict(beat_F=float(F), n_est=ne, n_true=nt, n_ratio=ne / nt,
                    blind_best=float(bb), margin=float(F - bb),
                    CMLc=float(cmc), CMLt=float(cml), AMLt=float(aml))

    summary = {k: agg(k) for k in ("pf_free", "pf_gtmeter", "pf_clamp",
                                   "pf_free_snap", "pf_clamp_snap", "peakpick")}
    for k, v in summary.items():
        print(f"[{k:14s}] F={v['beat_F']:.4f} n_est/n_true={v['n_est']}/{v['n_true']} "
              f"(ratio {v['n_ratio']:.3f}) blind_best={v['blind_best']:.4f} "
              f"MARGIN={v['margin']:+.4f} | CMLc={v['CMLc']:.3f} CMLt={v['CMLt']:.3f} "
              f"AMLt={v['AMLt']:.3f}", flush=True)

    json.dump(summary, open(f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough/{a.tag}_summary.json", "w"),
              indent=1, default=float)
    print("WROTE", a.tag, flush=True)


if __name__ == "__main__":
    main()
