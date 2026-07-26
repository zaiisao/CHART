"""ARTIFACT CONTROLS for the A+B variant.

A high beat_F from a deploy path proves nothing on its own.  Three ways to fake it:

  (1) DENSITY.  mir_eval F rises with the number of estimates.  An audio-BLIND isochronous
      grid emitting the same count scores surprisingly well.  -> "blind_grid" floor.
  (2) NOT-AUDIO-LOCKED.  A path may respond to audio *statistics* without being aligned.
      -> "time-roll" control: roll h circularly by +1000 frames, keep the labels.  A truly
      audio-locked read-out must LOSE most of its margin over the floor.
  (3) CHAOS.  The mandatory shift test measures max|circular difference| between the
      trajectories for h and h-shifted-by-25-frames.  A stochastic particle filter reruns
      differently anyway, so the same statistic is computed for two different SEEDS on the
      SAME audio ("seed control").  A shift response only counts if it is (a) far above the
      seed control and, decisively, (b) accompanied by the estimated beats MOVING BY THE
      SHIFT (median matched offset ~ +0.50 s) and by F recovering when scored against the
      shifted reference.

Usage:  python verify.py --ckpt <final.pt> --regime dirac|mert [--n 20] [--K 300]
"""
import sys, os, json, math, time, argparse
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/ab")

import numpy as np
import torch

from vbpm.model import BarPointerVAE
from vbpm.elbo import free_run
from vbpm.evaluate import beats_from_barphase, downbeats_from_barphase, f_measure, _estimate_meter, metronome
from model_ab import BarPointerVAE_AB
from elbo_ab import free_run_ab, particle_filter
import common as C

DEV = "cuda:0"
FPS = C.FPS
TWO_PI = C.TWO_PI


# --------------------------------------------------------------------------- helpers
def blind_grid_F(ref, n_est, dur, n_rep=25, rng=None):
    """Chance floor: audio-blind isochronous grid with exactly n_est beats over `dur` s."""
    if n_est < 1 or len(ref) < 2:
        return float("nan")
    rng = rng or np.random.default_rng(0)
    period = dur / n_est
    vals = []
    for _ in range(n_rep):
        off = rng.uniform(0.0, period)
        est = off + np.arange(n_est) * period
        vals.append(f_measure(ref, est[est < dur]))
    return float(np.mean(vals))


@torch.no_grad()
def phase_of(kind, model, h, K, temper, seed):
    torch.manual_seed(seed)
    if kind == "baseline_freerun":
        return free_run(model, h)["phase_mu"][0].float().cpu().numpy()
    if kind == "A_freerun":
        return free_run_ab(model, h, use_corr=True)["phase_mu"][0].float().cpu().numpy()
    if kind == "noA_freerun":
        return free_run_ab(model, h, use_corr=False)["phase_mu"][0].float().cpu().numpy()
    if kind in ("B_filter", "AB_filter"):
        r = particle_filter(model, h, K=K, use_corr=(kind == "AB_filter"), temper=temper)
        return r["phase_mean"][0].float().cpu().numpy()      # circular weighted mean read-out
    if kind in ("B_filter_path", "AB_filter_path"):
        r = particle_filter(model, h, K=K, use_corr=kind.startswith("AB"), temper=temper)
        return r["phase_path"][0].float().cpu().numpy()
    raise ValueError(kind)


def build_h(regime, song, T, merge, shift=0, roll=0):
    if regime == "dirac":
        h = C.dirac_h(song["beats"], song["downs"], 0, T, shift_frames=shift,
                      rng=np.random.default_rng(0))
        if roll:
            h = np.roll(h, roll, axis=0)
        return torch.from_numpy(h).unsqueeze(0).to(DEV)
    f = song["feats"][:, :T, :].astype(np.float32)
    if shift:
        f = np.roll(f, shift, axis=1)
    if roll:
        f = np.roll(f, roll, axis=1)
    return merge(torch.from_numpy(f).unsqueeze(0).to(DEV))


def med_offset(est_a, est_b):
    """Median (est_b - nearest est_a) : how far the estimate MOVED, in seconds."""
    if len(est_a) < 2 or len(est_b) < 2:
        return float("nan")
    d = []
    for t in est_b:
        i = int(np.argmin(np.abs(est_a - t)))
        d.append(t - est_a[i])
    return float(np.median(d))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--regime", choices=["dirac", "mert"], required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--temper", type=float, default=1.0)
    ap.add_argument("--max_frames", type=int, default=1600)
    ap.add_argument("--roll", type=int, default=1000)
    ap.add_argument("--shift", type=int, default=25)
    ap.add_argument("--kinds", default="A_freerun,noA_freerun,B_filter,AB_filter")
    ap.add_argument("--baseline", type=int, default=0)
    ap.add_argument("--max_phase_corr", type=float, default=0.30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location=DEV)
    h_dim = 8 if a.regime == "dirac" else 768
    merge = None
    if a.baseline:
        model = BarPointerVAE(h_dim=h_dim, hidden=128, num_meters=4).to(DEV)
    else:
        model = BarPointerVAE_AB(h_dim=h_dim, hidden=128, num_meters=4,
                                 max_phase_corr=a.max_phase_corr).to(DEV)
    model.load_state_dict(ck["model"]); model.eval()
    if a.regime == "mert":
        merge = C.LayerMerge().to(DEV); merge.load_state_dict(ck["merge"]); merge.eval()
    print(f"loaded {a.ckpt} | regime={a.regime} max_phase_corr={getattr(model,'max_phase_corr','n/a')}", flush=True)

    songs = C.load_split("eval", cap=a.n, with_feats=(a.regime == "mert"))
    kinds = a.kinds.split(",")
    rec = {k: {n: [] for n in ("beat_F", "db_F", "floor", "roll_F", "roll_floor",
                               "n_est", "n_true", "bars_adv", "bars_true", "phi_rng",
                               "shift_maxcirc", "seed_maxcirc", "shift_offset",
                               "shift_F_vs_orig", "shift_F_vs_shifted", "ibi",
                               "shift_beatphase")} for k in kinds}
    t0 = time.time()
    for si, s in enumerate(songs):
        T = min(s["T"], a.max_frames)
        dur = T / FPS
        ref = s["beats"][s["beats"] < dur]
        dref = s["downs"][s["downs"] < dur]
        if len(ref) < 2:
            continue
        m = _estimate_meter(ref, dref)
        h0 = build_h(a.regime, s, T, merge)
        hR = build_h(a.regime, s, T, merge, roll=a.roll)
        hS = build_h(a.regime, s, T, merge, shift=a.shift)
        for k in kinds:
            p0 = phase_of(k, model, h0, a.K, a.temper, seed=1234)
            est0 = beats_from_barphase(p0, m, FPS)
            d = np.diff(p0); d = (d + math.pi) % TWO_PI - math.pi
            rec[k]["beat_F"].append(f_measure(ref, est0))
            rec[k]["db_F"].append(f_measure(dref, downbeats_from_barphase(p0, FPS)) if len(dref) >= 2 else np.nan)
            rec[k]["floor"].append(blind_grid_F(ref, len(est0), dur))
            rec[k]["n_est"].append(len(est0)); rec[k]["n_true"].append(len(ref))
            rec[k]["bars_adv"].append(float(d.sum() / TWO_PI)); rec[k]["bars_true"].append(len(ref) / m)
            rec[k]["phi_rng"].append(float(p0.max() - p0.min()))
            # --- control (2) time-roll
            pR = phase_of(k, model, hR, a.K, a.temper, seed=1234)
            estR = beats_from_barphase(pR, m, FPS)
            rec[k]["roll_F"].append(f_measure(ref, estR))
            rec[k]["roll_floor"].append(blind_grid_F(ref, len(estR), dur))
            # --- control (3) shift
            pS = phase_of(k, model, hS, a.K, a.temper, seed=1234)
            estS = beats_from_barphase(pS, m, FPS)
            rec[k]["shift_maxcirc"].append(float(C.circ_absdiff(p0, pS).max()))
            pSeed = phase_of(k, model, h0, a.K, a.temper, seed=999)
            rec[k]["seed_maxcirc"].append(float(C.circ_absdiff(p0, pSeed).max()))
            rec[k]["shift_offset"].append(med_offset(est0, estS))
            rec[k]["shift_F_vs_orig"].append(f_measure(ref, estS))
            ref_shift = ref + a.shift / FPS
            rec[k]["shift_F_vs_shifted"].append(f_measure(ref_shift[ref_shift < dur], estS))
            ibi = float(np.median(np.diff(ref)))
            rec[k]["ibi"].append(ibi)
            # how far from a whole beat the +25-frame shift is (0 or 1 => the test is vacuous)
            rec[k]["shift_beatphase"].append(float(((a.shift / FPS) / ibi) % 1.0))
        if (si + 1) % 5 == 0:
            print(f"  {si+1}/{len(songs)} songs  {time.time()-t0:.0f}s", flush=True)

    summ = {}
    for k in kinds:
        r = rec[k]
        g = lambda n: float(np.nanmean(r[n])) if len(r[n]) else float("nan")
        summ[k] = dict(
            beat_F=g("beat_F"), db_F=g("db_F"), blind_grid_floor=g("floor"),
            margin_over_floor=g("beat_F") - g("floor"),
            density=float(np.sum(r["n_est"])) / max(np.sum(r["n_true"]), 1),
            bars_adv=g("bars_adv"), bars_true=g("bars_true"), phi_range=g("phi_rng"),
            roll_beat_F=g("roll_F"), roll_floor=g("roll_floor"),
            roll_margin=g("roll_F") - g("roll_floor"),
            shift_maxcirc=g("shift_maxcirc"), seed_maxcirc=g("seed_maxcirc"),
            shift_offset_sec=float(np.nanmedian(r["shift_offset"])),
            shift_F_vs_orig=g("shift_F_vs_orig"), shift_F_vs_shifted=g("shift_F_vs_shifted"),
            median_ibi=float(np.nanmedian(r["ibi"])) if r["ibi"] else float("nan"),
            n_songs=len(r["beat_F"]), per_song=r)
    metro = float(np.mean([f_measure(s["beats"][s["beats"] < min(s["T"], a.max_frames) / FPS],
                                     metronome(min(s["T"], a.max_frames), FPS)) for s in songs]))
    summ["_metronome_F"] = metro
    summ["_args"] = vars(a)

    print("\n" + "=" * 118)
    print(f"{'path':<16}{'beat_F':>8}{'floor':>8}{'marg':>7}{'dens':>7}{'barsAdv':>9}{'barsT':>7}"
          f"{'rollF':>7}{'rollMg':>8}{'shiftDelta':>11}{'seedDelta':>10}{'shiftOff':>9}{'F|orig':>8}{'F|shift':>8}")
    for k in kinds:
        v = summ[k]
        print(f"{k:<16}{v['beat_F']:>8.3f}{v['blind_grid_floor']:>8.3f}{v['margin_over_floor']:>7.3f}"
              f"{v['density']:>7.2f}{v['bars_adv']:>9.2f}{v['bars_true']:>7.2f}"
              f"{v['roll_beat_F']:>7.3f}{v['roll_margin']:>8.3f}{v['shift_maxcirc']:>11.3f}"
              f"{v['seed_maxcirc']:>10.3f}{v['shift_offset_sec']:>9.3f}"
              f"{v['shift_F_vs_orig']:>8.3f}{v['shift_F_vs_shifted']:>8.3f}")
    print(f"{'metronome120':<16}{metro:>8.3f}")
    print("=" * 118)
    out = a.out or (os.path.splitext(a.ckpt)[0] + f".verify_{a.regime}.json")
    json.dump(summ, open(out, "w"), indent=1)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
