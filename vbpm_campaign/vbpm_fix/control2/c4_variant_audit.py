"""(4) TIME-ROLL / SHIFT-EQUIVARIANCE CONTROL applied to the actual claimed fix.

Three conditions per system, SAME rng seed, stratified 30-song eval subset, cap 1600:

  A  aligned                                  -> the claimed score
  B  features rolled +R frames, labels fixed  -> must COLLAPSE to the floor if the score
                                                 came from the audio
  C  features rolled +R, labels ALSO shifted +R/fps -> must RECOVER to ~A if the deploy path
                                                 actually follows the audio (shift-equivariance)

B alone cannot distinguish "audio-blind" from "leaking": an audio-blind constant grid is
invariant (B == A) and a leaking evaluator also keeps B high. C separates them:
  A high, B low,  C high  -> genuinely tracks the audio (CLEAN)
  A ~ B ~ C               -> AUDIO-BLIND (score is a property of the read-out, not the audio)
  A high, B high, C low   -> leak

All scoring is trimmed to t in [R/fps + 0.5 s, T/fps) in every condition so the circular
wrap-around region is excluded and A/B/C are directly comparable.

Systems audited: whatever trained checkpoints exist in vbpm_fix/ (variant-B MERT free_run
and its particle filter), plus MY conv probe as a POSITIVE CONTROL (a system known to use
the audio must pass, otherwise the control has no power).
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/control2")

from cc import (FPS, load_split, agg, ratio, banner, metronome, f_measure, _estimate_meter,
                beats_from_barphase, downbeats_from_barphase, beats_from_activation)

DEV = "cuda:0"
ap = argparse.ArgumentParser()
ap.add_argument("--roll", type=int, default=250)          # 5 s
ap.add_argument("--cap", type=int, default=1600)
ap.add_argument("--K", type=int, default=300)
ap.add_argument("--n_per_ds", type=int, default=10)
ap.add_argument("--out", default="/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/control2/c4_out.json")
A = ap.parse_args()
R = A.roll
LO = R / FPS + 0.5

ev_all = load_split("eval", with_feats=True)
sub = []
for ds in ["ballroom", "beatles", "hainsworth"]:
    sub += [s for s in ev_all if s["dataset"] == ds][:A.n_per_ds]
print(f"stratified subset: {len(sub)} songs "
      f"({ {d: sum(1 for s in sub if s['dataset']==d) for d in ['ballroom','beatles','hainsworth']} }) "
      f"cap={A.cap} roll={R} frames ({R/FPS:.1f}s)", flush=True)


def trim(times, lo, hi):
    t = np.asarray(times, float)
    return t[(t >= lo) & (t < hi)]


def score_traj(ph, s, T, cond):
    """Official read-out; labels shifted by +R/fps in condition C; all trimmed identically."""
    hi = T / FPS
    ref = s["beats"][s["beats"] < hi]
    dref = s["downs"][s["downs"] < hi]
    m = _estimate_meter(ref, dref)                       # meter from the UNSHIFTED gt (same gift)
    if cond == "C":
        ref = s["beats"] + R / FPS
        dref = s["downs"] + R / FPS
    est = beats_from_barphase(np.asarray(ph), m, FPS)
    dest = downbeats_from_barphase(np.asarray(ph), FPS)
    ref, dref = trim(ref, LO, hi), trim(dref, LO, hi)
    est, dest = trim(est, LO, hi), trim(dest, LO, hi)
    if len(ref) < 2:
        return None
    return dict(beat_F=f_measure(ref, est),
                downbeat_F=f_measure(dref, dest) if len(dref) >= 2 else float("nan"),
                n_est=len(est), n_true=len(ref),
                metronome_F=f_measure(ref, trim(metronome(T, FPS), LO, hi)),
                dataset=s["dataset"])


def score_act(prob, s, T, cond):
    hi = T / FPS
    ref = s["beats"] + (R / FPS if cond == "C" else 0.0)
    dref = s["downs"] + (R / FPS if cond == "C" else 0.0)
    est = beats_from_activation(np.asarray(prob), FPS, thr=0.5)
    ref, dref = trim(ref, LO, hi), trim(dref, LO, hi)
    est = trim(est, LO, hi)
    if len(ref) < 2:
        return None
    return dict(beat_F=f_measure(ref, est), downbeat_F=float("nan"),
                n_est=len(est), n_true=len(ref),
                metronome_F=f_measure(ref, trim(metronome(T, FPS), LO, hi)),
                dataset=s["dataset"])


def blind_grid_at_density(rows, seed=0):
    """Constant-spacing grid matched to the system's OWN emission density, random phase."""
    rng = np.random.default_rng(seed)
    out = []
    for s in sub:
        T = min(s["T"], A.cap)
        hi = T / FPS
        ref = trim(s["beats"], LO, hi)
        if len(ref) < 2:
            continue
        dens = ratio(rows)
        n = max(int(round(len(ref) * dens)), 2)
        step = (hi - LO) / n
        est = LO + np.arange(n) * step + rng.random() * step
        out.append(f_measure(ref, trim(est, LO, hi)))
    return float(np.mean(out))


def run_system(name, phase_fn, scorer, results):
    banner(f"SYSTEM: {name}")
    per = {}
    for cond, roll in [("A", 0), ("B", R), ("C", R)]:
        rows = []
        t0 = time.time()
        for i, s in enumerate(sub):
            T = min(s["T"], A.cap)
            torch.manual_seed(1234 + i)                 # identical draw in every condition
            out = phase_fn(s, T, roll)
            r = scorer(out, s, T, cond)
            if r:
                rows.append(r)
        lab = {"A": "A aligned", "B": f"B feats rolled +{R}, labels fixed",
               "C": f"C feats rolled +{R}, labels shifted +{R/FPS:.1f}s"}[cond]
        print(f"  {lab:44s} beat_F={agg(rows,'beat_F'):.3f} db_F={agg(rows,'downbeat_F'):.3f} "
              f"n_est/n_true={ratio(rows):.3f} metro={agg(rows,'metronome_F'):.3f} "
              f"N={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
        per[cond] = dict(beat_F=agg(rows, "beat_F"), db_F=agg(rows, "downbeat_F"),
                         ratio=ratio(rows), metronome=agg(rows, "metronome_F"), N=len(rows))
        if cond == "A":
            per["blind_grid_at_own_density"] = blind_grid_at_density(rows)
            print(f"  {'FLOOR blind grid at its own density':44s} "
                  f"beat_F={per['blind_grid_at_own_density']:.3f}", flush=True)
    a, b, c = per["A"]["beat_F"], per["B"]["beat_F"], per["C"]["beat_F"]
    floor = per["A"]["metronome"]
    if a <= max(floor, per["blind_grid_at_own_density"]) + 0.02:
        v = "AT/BELOW FLOOR -- no tracking claim survives (score explained by density alone)"
    elif abs(a - b) < 0.02 and abs(a - c) < 0.02:
        v = "AUDIO-BLIND -- score invariant to a 5 s audio shift in BOTH directions"
    elif (a - b) > 0.05 and (c - b) > 0.05:
        v = "CLEAN / TRACKING -- collapses when misaligned, recovers when labels follow the audio"
    elif (a - b) < 0.02 and c < a - 0.05:
        v = "*** LEAK SUSPECTED -- stays high when misaligned, drops when labels follow ***"
    else:
        v = "AMBIGUOUS"
    print(f"  -> A={a:.3f} B={b:.3f} C={c:.3f} floor={floor:.3f} "
          f"density-floor={per['blind_grid_at_own_density']:.3f} :: {v}", flush=True)
    per["verdict"] = v
    results[name] = per
    json.dump(results, open(A.out, "w"), indent=1)


results = {}

# ---------------------------------------------------------------- POSITIVE CONTROL: conv probe
class Probe(nn.Module):
    """Same architecture as c3_mert_baselines.Probe (copied, not imported: that module
    trains on import)."""

    def __init__(self, kind="conv"):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(13))
        self.head = nn.Sequential(nn.Conv1d(768, 128, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(128, 1, 1))
        self.kind = kind

    def forward(self, feats):
        m = torch.einsum("l,bltf->btf", torch.softmax(self.layer_logits, 0), feats)
        return self.head(m.transpose(1, 2)).squeeze(1)

import os
CKP = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/control2/probe_conv.pt"
if os.path.exists(CKP):
    probe = Probe("conv").to(DEV)
    probe.load_state_dict(torch.load(CKP, map_location=DEV))
    probe.eval()

    @torch.no_grad()
    def probe_fn(s, T, roll):
        f = torch.from_numpy(s["feats"][:, :T, :].astype(np.float32)).unsqueeze(0).to(DEV)
        if roll:
            f = torch.roll(f, roll, dims=2)
        return torch.sigmoid(probe(f))[0].cpu().numpy()

    run_system("conv probe + peak-pick (POSITIVE CONTROL)", probe_fn, score_act, results)
else:
    print("!! conv probe checkpoint not ready -- positive control skipped", flush=True)

# ---------------------------------------------------------------- UNFIXED VBPM (the "0.31")
UB = "/home/sogang/jaehoon/VBPM_reintegration/runs/mert_vbpm/best.pt"
if os.path.exists(UB):
    from vbpm.model import BarPointerVAE
    from vbpm.elbo import free_run as vanilla_free_run

    class _Merge(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_logits = nn.Parameter(torch.zeros(13))

        def forward(self, f):
            return torch.einsum("l,bltf->btf", torch.softmax(self.layer_logits, 0), f)

    ck0 = torch.load(UB, map_location=DEV)
    mg0 = _Merge().to(DEV)
    mg0.load_state_dict(ck0["merge"])
    mg0.eval()
    m0 = BarPointerVAE(h_dim=768, hidden=128, num_meters=4).to(DEV)
    m0.load_state_dict(ck0["model"])
    m0.eval()

    @torch.no_grad()
    def unfixed_fn(s, T, roll):
        f = torch.from_numpy(s["feats"][:, :T, :].astype(np.float32)).unsqueeze(0).to(DEV)
        if roll:
            f = torch.roll(f, roll, dims=2)
        return vanilla_free_run(m0, mg0(f))["phase_mu"][0, :T].cpu().numpy()

    run_system("UNFIXED VBPM free_run (the quoted 0.31)", unfixed_fn, score_traj, results)

# ---------------------------------------------------------------- variant B (the claimed fix)
VB = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/varB_mert.pt"
if os.path.exists(VB):
    from vbpm.elbo import free_run
    from vbpm_fix.variant_b import BarPointerVAE_B, particle_filter
    from vbpm_fix.run_mert import FixedProj, LayerMerge, OBS_DIM

    ck = torch.load(VB, map_location=DEV)
    proj = FixedProj(ck["pca_mean"], ck["pca_comps"]).to(DEV)
    merge = LayerMerge().to(DEV)
    merge.load_state_dict(ck["merge"])
    merge.eval()
    vb = BarPointerVAE_B(h_dim=768, hidden=128, num_meters=4, obs_dim=OBS_DIM,
                         obs_type="gauss").to(DEV)
    vb.load_state_dict(ck["model"])
    vb.eval()

    def feats(s, T, roll):
        f = torch.from_numpy(s["feats"][:, :T, :].astype(np.float32)).unsqueeze(0).to(DEV)
        if roll:
            f = torch.roll(f, roll, dims=2)
        return f

    @torch.no_grad()
    def vb_free(s, T, roll):
        return free_run(vb, merge(feats(s, T, roll)))["phase_mu"][0, :T].cpu().numpy()

    @torch.no_grad()
    def vb_pf(s, T, roll):
        f = feats(s, T, roll)
        return particle_filter(vb, merge(f), proj(f), K=A.K)["phase_mean"].numpy()

    run_system("variant-B free_run (open loop)", vb_free, score_traj, results)
    run_system(f"variant-B PARTICLE FILTER (K={A.K})", vb_pf, score_traj, results)
else:
    print("!! varB_mert.pt not found", flush=True)

print("\nWROTE", A.out, flush=True)
