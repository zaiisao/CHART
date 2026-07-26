"""E3 ACCOUNTING -- the no-VAE controls.

Ablation (i) of the task: supervised emission + inference, but with a FIXED
(non-learned) bar-pointer transition and NO VAE anywhere.  Three rigidity levels:

  (i-a) RIGID-AUTOCORR : constant tempo per song taken from the autocorrelation of the
        beat activation; the only free parameters are the bar-phase offset and the meter,
        both chosen by EXACT MAP over the supervised emission.  No dynamics at all.
  (i-b) RIGID-MAP      : same, but the constant tempo is also chosen by MAP over a grid
        (55-215 BPM).  Still a rigid metronome; strictly stronger than (i-a).
  (i-c) SIMPLE-PF      : hand-set bar-pointer transition (deterministic pointer advance +
        random-walk log-tempo + bar-gated meter switch), bootstrap PF.  Non-learned.

Plus the reference bars (iii) frozen activation head peak-pick, 120-BPM metronome, and
the oracle true-bar-phase ceiling.  Everything scored through e3_common.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")
import e3_common as C                                                 # noqa: E402
from e3_common import (FPS, TWO_PI, METERS, FINAL, _estimate_meter,    # noqa: E402
                       beats_from_activation, metronome, score_traj,
                       score_events, summarize, pr, autocorr_period)
from e3_emission import PhaseEmission, load_act, load_split, obs_contrast, song_phase  # noqa
from e3_pf import particle_filter as simple_pf                        # noqa: E402


# --------------------------------------------------------------- rigid MAP decode
def rigid_decode(tab_by_m, nb, periods_by_m):
    """Exact MAP over {meter} x {constant bar period} x {bar-phase offset at bin resolution}.

    tab_by_m : dict m -> [T, nb[m]] log p(o_t | bin, m)
    periods_by_m : dict m -> iterable of candidate BAR periods in frames
    Returns (phase[T], m_best, bar_period_best, best_score).
    """
    best = (-np.inf, None, None)
    for m, tab in tab_by_m.items():
        T, nbm = tab.shape
        t = np.arange(T)
        for P in periods_by_m[m]:
            b0 = ((t / P) % 1.0 * nbm).astype(np.int64)
            idx = (b0[:, None] + np.arange(nbm)[None, :]) % nbm       # [T, nbm]
            S = np.take_along_axis(tab, idx, axis=1).sum(0)           # [nbm] score per offset
            k = int(np.argmax(S))
            if S[k] > best[0]:
                best = (float(S[k]), m, float(P), k)
    _, m, P, k = best
    T = tab_by_m[m].shape[0]
    nbm = nb[m]
    ph = ((np.arange(T) / P) + k / nbm) % 1.0 * TWO_PI
    return ph, m, P, best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lik", default="gauss", choices=["bern", "gauss"])
    ap.add_argument("--bpb", type=int, default=24)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--split", default="eval", choices=["eval", "train"])
    ap.add_argument("--n_eval", type=int, default=0)
    ap.add_argument("--K", type=int, default=600)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--sigma_lt", type=float, default=0.05)
    ap.add_argument("--sigma_phi", type=float, default=0.03)
    ap.add_argument("--p_switch", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="e3_abl")
    a = ap.parse_args()

    tr = load_split("train"); at = load_act("train")
    if a.split == "train":
        ev, ae = tr, at
    else:
        ev, ae = load_split("eval"), load_act("eval")
    if a.n_eval:
        ev = ev[:a.n_eval]
    print(f"train {len(tr)}  eval({a.split}) {len(ev)}", flush=True)

    emis = PhaseEmission(bins_per_beat=a.bpb, likelihood=a.lik,
                         smooth=a.smooth).fit(tr, at, phase_mode="downbeat")
    c_ev, per_c = obs_contrast(emis, ev, ae, phase_mode="downbeat")
    c_tr, _ = obs_contrast(emis, tr, at, phase_mode="downbeat")
    print(f"SUPERVISED EMISSION lik={a.lik} bpb={a.bpb} songs/meter={emis.n_used} "
          f"obs_contrast eval={c_ev:.4f} train={c_tr:.4f}", flush=True)

    prior = np.zeros(5)
    for s in tr:
        m = _estimate_meter(s["beats"], s["downs"])
        if m in METERS:
            prior[m] += 1

    # candidate constant tempi for (i-b): 48 log-spaced beat periods, 55-215 BPM
    beat_periods = np.exp(np.linspace(math.log(14.0), math.log(55.0), 48))

    rows = {k: [] for k in ("rigid_autocorr", "rigid_autocorr_own", "rigid_map",
                            "rigid_map_own", "simple_pf_mean", "simple_pf_map",
                            "simple_pf_path", "simple_pf_path_own", "act_peak",
                            "metronome120", "oracle_phase")}
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
        a2 = act[:T]
        tab = emis.loglik_table(a2)
        base = dict(stem=s["stem"], dataset=s["dataset"], T=T, n_true=len(ref),
                    n_true_db=len(dref), ess=float("nan"),
                    obs_contrast=per_c[i] if i < len(per_c) else float("nan"),
                    meter_ok=float("nan"))

        # ---- (i-a) rigid, tempo from autocorrelation of the beat activation ----
        P_ac = autocorr_period(a2[:, 0])
        ph, m_a, P_a, _ = rigid_decode(tab, emis.nb, {m: [m * P_ac] for m in METERS})
        rows["rigid_autocorr"].append({**base, "meter_ok": float(m_a == m_gt),
                                       **score_traj(ph, m_gt, ref, dref, T)})
        rows["rigid_autocorr_own"].append({**base, "meter_ok": float(m_a == m_gt),
                                           **score_traj(ph, m_a, ref, dref, T)})

        # ---- (i-b) rigid, tempo by MAP over a grid ----
        ph, m_b, P_b, _ = rigid_decode(tab, emis.nb,
                                       {m: m * beat_periods for m in METERS})
        rows["rigid_map"].append({**base, "meter_ok": float(m_b == m_gt),
                                  **score_traj(ph, m_gt, ref, dref, T)})
        rows["rigid_map_own"].append({**base, "meter_ok": float(m_b == m_gt),
                                      **score_traj(ph, m_b, ref, dref, T)})

        # ---- (i-c) hand-set bar-pointer transition + bootstrap PF ----
        LL = emis.padded_table(a2)
        out = simple_pf(LL, emis.nb, K=a.K, alpha=a.alpha, sigma_lt=a.sigma_lt,
                        sigma_phi=a.sigma_phi, p_switch=a.p_switch,
                        meter_prior=prior, fps=FPS, seed=a.seed + i)
        m_pf = int(np.bincount(out["meter_path"]).argmax())
        pfb = {**base, "ess": out["ess"], "meter_ok": float(m_pf == m_gt)}
        for k, ph2 in (("simple_pf_mean", out["phase_mean"]),
                       ("simple_pf_map", out["phase_map"]),
                       ("simple_pf_path", out["phase_path"])):
            rows[k].append({**pfb, **score_traj(ph2, m_gt, ref, dref, T)})
        rows["simple_pf_path_own"].append({**pfb, **score_traj(out["phase_path"], m_pf,
                                                               ref, dref, T)})

        # ---- (iii) reference bars ----
        e_b = beats_from_activation(a2[:, 0], FPS)
        e_d = beats_from_activation(a2[:, 1], FPS, min_dist_sec=0.30)
        rows["act_peak"].append({**base, **score_events(e_b, e_d, ref, dref, T)})
        rows["metronome120"].append({**base, **score_events(metronome(T, FPS),
                                                            np.array([]), ref, dref, T)})
        pho = song_phase(s, "downbeat")
        if pho is not None:
            rows["oracle_phase"].append({**base, **score_traj(pho[:T], m_gt, ref, dref, T)})
        if i % 20 == 0:
            print(f"  {i}/{len(ev)}  {time.time()-t0:.0f}s", flush=True)

    res = {"config": vars(a), "emission": {"contrast_eval": c_ev, "contrast_train": c_tr,
                                           "n_used": {str(k): v for k, v in emis.n_used.items()}}}
    names = {"rigid_autocorr": "(i-a) RIGID autocorr-tempo, no VAE",
             "rigid_autocorr_own": "(i-a) RIGID autocorr [own meter]",
             "rigid_map": "(i-b) RIGID MAP-tempo, no VAE",
             "rigid_map_own": "(i-b) RIGID MAP-tempo [own meter]",
             "simple_pf_path_own": "(i-c) SIMPLE-PF [path, own meter]",
             "simple_pf_mean": "(i-c) SIMPLE-PF hand transition [mean]",
             "simple_pf_map": "(i-c) SIMPLE-PF hand transition [map]",
             "simple_pf_path": "(i-c) SIMPLE-PF hand transition [path]",
             "act_peak": "(iii) act-head peak-pick",
             "metronome120": "metronome-120 floor",
             "oracle_phase": "oracle true bar phase (ceiling)"}
    print("", flush=True)
    for k, rr in rows.items():
        if rr:
            d = summarize(rr, names[k])
            pr(d)
            res.setdefault("summary", {})[k] = d
    res["rows"] = rows
    json.dump(res, open(f"{FINAL}/{a.tag}.json", "w"), indent=1, default=float)
    print("WROTE", a.tag + ".json", flush=True)


if __name__ == "__main__":
    main()
