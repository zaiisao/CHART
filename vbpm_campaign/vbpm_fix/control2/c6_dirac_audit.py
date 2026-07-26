"""DIRAC-REGIME audit of the trained variants: shift-equivariance + the copy ceiling.

In the Dirac regime the input h IS the label track (impulses at true beat/downbeat frames),
so the only meaningful questions are:
  * does the deploy path FOLLOW its input when the impulses move (condition C)?
  * how far is the score from the trivial copy ceiling (1.000 activation / 0.960 bar-phase)?

Conditions (identical seeds): A aligned; B impulses delayed +R frames, labels fixed;
C impulses delayed +R AND labels shifted +R/fps. All scoring trimmed to
t in [R/fps + 0.5 s, T/fps).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/control2")

from cc import (FPS, load_split, agg, ratio, banner, metronome, f_measure, _estimate_meter,
                beats_from_barphase, downbeats_from_barphase, ideal_barphase)

DEV = "cuda:0"
ap = argparse.ArgumentParser()
ap.add_argument("--roll", type=int, default=250)
ap.add_argument("--cap", type=int, default=1600)
ap.add_argument("--K", type=int, default=400)
ap.add_argument("--n_per_ds", type=int, default=10)
ap.add_argument("--out", default="/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/control2/c6_out.json")
A = ap.parse_args()
R = A.roll
LO = R / FPS + 0.5
H_DIM = 8

ev_all = load_split("eval")
sub = []
for ds in ["ballroom", "beatles", "hainsworth"]:
    sub += [s for s in ev_all if s["dataset"] == ds][:A.n_per_ds]
print(f"dirac audit: {len(sub)} songs, cap={A.cap}, roll={R} ({R/FPS:.1f}s), K={A.K}", flush=True)


def dirac_h(s, n, shift=0, seed=1):
    r = np.random.default_rng(seed)
    h = r.standard_normal((n, H_DIM)).astype(np.float32) * 0.01
    for t in s["beats"]:
        i = int(round(t * FPS)) + shift
        if 0 <= i < n:
            h[i, 0] += 1.0
    for t in s["downs"]:
        i = int(round(t * FPS)) + shift
        if 0 <= i < n:
            h[i, 1] += 1.0
    return h


def trim(t, lo, hi):
    t = np.asarray(t, float)
    return t[(t >= lo) & (t < hi)]


def score_traj(ph, s, T, cond):
    hi = T / FPS
    ref0 = s["beats"][s["beats"] < hi]
    dref0 = s["downs"][s["downs"] < hi]
    m = _estimate_meter(ref0, dref0)
    off = R / FPS if cond == "C" else 0.0
    ref, dref = trim(s["beats"] + off, LO, hi), trim(s["downs"] + off, LO, hi)
    est = trim(beats_from_barphase(np.asarray(ph), m, FPS), LO, hi)
    dest = trim(downbeats_from_barphase(np.asarray(ph), FPS), LO, hi)
    if len(ref) < 2:
        return None
    return dict(beat_F=f_measure(ref, est),
                downbeat_F=f_measure(dref, dest) if len(dref) >= 2 else float("nan"),
                n_est=len(est), n_true=len(ref),
                metronome_F=f_measure(ref, trim(metronome(T, FPS), LO, hi)), dataset=s["dataset"])


def blind_grid_at_density(dens, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for s in sub:
        T = min(s["T"], A.cap)
        hi = T / FPS
        ref = trim(s["beats"], LO, hi)
        if len(ref) < 2:
            continue
        n = max(int(round(len(ref) * dens)), 2)
        step = (hi - LO) / n
        out.append(f_measure(ref, trim(LO + np.arange(n) * step + rng.random() * step, LO, hi)))
    return float(np.mean(out))


results = {}


def run_system(name, phase_fn):
    banner(f"SYSTEM: {name}")
    per = {}
    for cond, sh in [("A", 0), ("B", R), ("C", R)]:
        rows = []
        t0 = time.time()
        for i, s in enumerate(sub):
            T = min(s["T"], A.cap)
            torch.manual_seed(1234 + i)
            r = score_traj(phase_fn(s, T, sh), s, T, cond)
            if r:
                rows.append(r)
        lab = {"A": "A aligned", "B": f"B impulses +{R}, labels fixed",
               "C": f"C impulses +{R}, labels shifted +{R/FPS:.1f}s"}[cond]
        print(f"  {lab:44s} beat_F={agg(rows,'beat_F'):.3f} db_F={agg(rows,'downbeat_F'):.3f} "
              f"n_est/n_true={ratio(rows):.3f} metro={agg(rows,'metronome_F'):.3f} N={len(rows)} "
              f"({time.time()-t0:.0f}s)", flush=True)
        per[cond] = dict(beat_F=agg(rows, "beat_F"), db_F=agg(rows, "downbeat_F"),
                         ratio=ratio(rows), metronome=agg(rows, "metronome_F"), N=len(rows))
        if cond == "A":
            per["blind_grid_at_own_density"] = blind_grid_at_density(ratio(rows))
            print(f"  {'FLOOR blind grid at its own density':44s} "
                  f"beat_F={per['blind_grid_at_own_density']:.3f}", flush=True)
    a, b, c = per["A"]["beat_F"], per["B"]["beat_F"], per["C"]["beat_F"]
    if a <= max(per["A"]["metronome"], per["blind_grid_at_own_density"]) + 0.02:
        v = "AT/BELOW FLOOR (density explains it) -- and the Dirac copy ceiling is 1.000"
    elif abs(a - b) < 0.02 and abs(a - c) < 0.02:
        v = "INPUT-BLIND -- invariant to moving the impulses that ARE the input"
    elif (a - b) > 0.05 and (c - b) > 0.05:
        v = "FOLLOWS ITS INPUT (still an ORACLE input: copy ceiling 1.000 / 0.960)"
    else:
        v = "AMBIGUOUS"
    print(f"  -> A={a:.3f} B={b:.3f} C={c:.3f} :: {v}", flush=True)
    per["verdict"] = v
    results[name] = per
    json.dump(results, open(A.out, "w"), indent=1)


# ---- reference: the copy ceiling under THIS trimmed protocol ----
banner("REFERENCE: Dirac copy ceilings under the same trimmed protocol")
rows = []
for s in sub:
    T = min(s["T"], A.cap)
    hi = T / FPS
    h = dirac_h(s, T)
    ph = ideal_barphase(np.where(h[:, 1] > 0.5)[0] / FPS, T)
    if ph is None:
        continue
    r = score_traj(ph, s, T, "A")
    if r:
        rows.append(r)
print(f"  bar-phase rebuilt from the Dirac downbeat channel: beat_F={agg(rows,'beat_F'):.3f} "
      f"db_F={agg(rows,'downbeat_F'):.3f} n_est/n_true={ratio(rows):.3f}", flush=True)
results["copy_ceiling_barphase"] = dict(beat_F=agg(rows, "beat_F"), db_F=agg(rows, "downbeat_F"))

# ---- variant-B Dirac model (PF + free_run) ----
VB = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/varB_dirac.pt"
if os.path.exists(VB):
    from vbpm.elbo import free_run
    from vbpm_fix.variant_b import BarPointerVAE_B, dirac_obs, particle_filter

    ck = torch.load(VB, map_location=DEV)
    sd = ck["model"] if "model" in ck else ck
    vb = BarPointerVAE_B(h_dim=H_DIM, hidden=128, num_meters=4, obs_dim=2, obs_type="bern").to(DEV)
    vb.load_state_dict(sd)
    vb.eval()

    def _h(s, T, sh):
        return torch.from_numpy(dirac_h(s, T, sh)).unsqueeze(0).to(DEV)

    @torch.no_grad()
    def vb_free(s, T, sh):
        return free_run(vb, _h(s, T, sh))["phase_mu"][0, :T].cpu().numpy()

    @torch.no_grad()
    def vb_pf(s, T, sh):
        h = _h(s, T, sh)
        return particle_filter(vb, h, dirac_obs(h), K=A.K)["phase_mean"].numpy()

    run_system("variant-B DIRAC free_run (open loop)", vb_free)
    run_system(f"variant-B DIRAC particle filter (K={A.K})", vb_pf)
else:
    print("!! varB_dirac.pt missing", flush=True)

print("\nWROTE", A.out, flush=True)
