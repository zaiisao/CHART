"""EXPERIMENT 3 -- shared scoring harness (independent of the E2 driver files).

Every E3 method is scored through EXACTLY this code so the accounting table is
apples-to-apples.  Nothing in vbpm/, vbpm_fix/, vbpm_arms/ is modified.

MANDATORY controls (task rule):
  * n_est / n_true                (beats and downbeats)
  * density-matched blind grid    (uniform grid with the SAME number of events)
  * best-phase-offset blind grid  (sweep of 12 offsets)  -> MARGIN = F - blind_best
  * obs_contrast                  (true-phase vs 11 wrong-offset likelihood ratio)
  * fraction of negative phase increments
"""
from __future__ import annotations

import math
import sys

import numpy as np

for _p in ("/home/sogang/jaehoon/VBPM_reintegration",
           "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
           "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms",
           "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vbpm.evaluate import (  # noqa: E402
    beats_from_barphase, downbeats_from_barphase, beats_from_activation,
    metronome, f_measure, _estimate_meter,
)

FPS = 50.0
TWO_PI = 2.0 * math.pi
METERS = (2, 3, 4)
FINAL = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final"


def blind_grid_controls(ref, T, n_est, n_off=12):
    """Uniform grid with the SAME number of events the method emitted.
    Returns (F at offset 0, best-of-12-offsets F)."""
    dur = T / FPS
    if n_est < 2 or len(ref) < 2:
        return float("nan"), float("nan")
    per = dur / n_est
    base = np.arange(n_est) * per
    f0 = f_measure(ref, base)
    best = max(f_measure(ref, base + k * per / n_off) for k in range(n_off))
    return float(f0), float(max(best, f0))


def phase_diag(ph):
    d = (np.diff(np.asarray(ph, float)) + math.pi) % TWO_PI - math.pi
    if len(d) == 0:
        return dict(frac_neg=float("nan"), mean_adv=float("nan"),
                    jitter=float("nan"), jitter_over_adv=float("nan"))
    adv = float(d.mean())
    return dict(frac_neg=float(np.mean(d < 0)), mean_adv=adv, jitter=float(d.std()),
                jitter_over_adv=float(d.std() / max(abs(adv), 1e-9)))


def score_traj(ph, m, ref, dref, T):
    est = beats_from_barphase(ph, m, FPS)
    dest = downbeats_from_barphase(ph, FPS)
    b0, bb = blind_grid_controls(ref, T, len(est))
    d0, db = blind_grid_controls(dref, T, len(dest))
    return dict(beat_F=f_measure(ref, est),
                db_F=f_measure(dref, dest) if len(dref) >= 2 else float("nan"),
                n_est=len(est), n_est_db=len(dest), blind0=b0, blind_best=bb,
                blind_db0=d0, blind_db_best=db, **phase_diag(ph))


def score_events(est, dest, ref, dref, T):
    b0, bb = blind_grid_controls(ref, T, len(est))
    d0, db = blind_grid_controls(dref, T, len(dest))
    return dict(beat_F=f_measure(ref, est),
                db_F=f_measure(dref, dest) if len(dref) >= 2 else float("nan"),
                n_est=len(est), n_est_db=len(dest), blind0=b0, blind_best=bb,
                blind_db0=d0, blind_db_best=db, frac_neg=float("nan"),
                mean_adv=float("nan"), jitter=float("nan"),
                jitter_over_adv=float("nan"))


def summarize(rows, name):
    def M(k):
        v = [r[k] for r in rows
             if isinstance(r.get(k), float) and not math.isnan(r[k])]
        return float(np.mean(v)) if v else float("nan")
    ne = sum(r["n_est"] for r in rows); nt = sum(r["n_true"] for r in rows)
    ned = sum(r["n_est_db"] for r in rows); ntd = sum(r["n_true_db"] for r in rows)
    bf, bb = M("beat_F"), M("blind_best")
    dfm, dbb = M("db_F"), M("blind_db_best")
    return dict(name=name, beat_F=bf, downbeat_F=dfm,
                n_ratio=ne / max(nt, 1), n_ratio_db=ned / max(ntd, 1),
                blind_same_density=M("blind0"), blind_best_offset=bb,
                margin_over_blind=bf - bb, blind_db_best=dbb,
                blind_db_same=M("blind_db0"), margin_db_over_blind=dfm - dbb,
                frac_neg=M("frac_neg"), jitter_over_adv=M("jitter_over_adv"),
                obs_contrast=M("obs_contrast"), ess=M("ess"),
                meter_acc=M("meter_ok"), n_songs=len(rows))


def pr(d):
    print(f"  [{d['name']:40s}] beat_F={d['beat_F']:.4f} db_F={d['downbeat_F']:.4f} "
          f"n_ratio={d['n_ratio']:.3f} blind0={d['blind_same_density']:.4f} "
          f"blindbest={d['blind_best_offset']:.4f} MARGIN={d['margin_over_blind']:+.4f} | "
          f"db_blind={d['blind_db_best']:.4f} MARGIN_db={d['margin_db_over_blind']:+.4f} "
          f"n_ratio_db={d['n_ratio_db']:.2f} | frac_neg={d['frac_neg']:.3f} "
          f"jit/adv={d['jitter_over_adv']:.2f} ESS={d['ess']:.0f} "
          f"contrast={d['obs_contrast']:.3g} meter_acc={d['meter_acc']:.2f}", flush=True)


def autocorr_period(a, lo=14, hi=55):
    """Dominant lag (frames) of the beat-activation autocorrelation, 55-215 BPM band."""
    x = np.asarray(a, float)
    x = x - x.mean()
    if x.std() < 1e-9:
        return float((lo + hi) // 2)
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return float((lo + hi) // 2)
    return float(lo + int(np.argmax(ac[lo:hi + 1])))
