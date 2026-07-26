"""EXP1 stage (3): DEPLOY-only, from the saved exp1_*.pt checkpoints.

Same PF + MANDATORY controls as exp1_cut_tempo.py, plus two extra diagnostics that
disambiguate 'phase revived as the BAR POINTER' from 'phase revived as some other code':
  contrast_bar   : likelihood ratio p(o|z at IDEAL BAR phase) / p(o|z at 11 rotations)
  contrast_beat  : same but phi = IDEAL BEAT phase (2*pi per BEAT, not per bar)
  lock_bar / lock_beat : circular resultant length R of (phi_PF - phi_ideal), 0=unlocked
  phidot_ratio   : PF mean phase advance / the song's TRUE bar-advance rate
"""
from __future__ import annotations
import argparse, json, math, sys, time
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")

import variant_b as VB
from vbpm.evaluate import (beats_from_barphase, downbeats_from_barphase, f_measure,
                           _estimate_meter, metronome)
from audit_common import load_split, ideal_barphase, banner, FPS
from common import smooth_phase
from arm_ii import blind_grid_controls, phase_diag, summarize, pr
from exp1_cut_tempo import CutModel, build_obs_cache, VIEWS

DEV, OUT = "cuda:0", "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
TWO_PI = 2 * math.pi


def _R(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.hypot(np.cos(d).mean(), np.sin(d).mean()))


@torch.no_grad()
def contrast(model, obs_t, phi, m, lt, T, n_off=12):
    dv = obs_t.device
    mt = F.one_hot(torch.tensor([m - 1] * T, device=dv), model.K).float()
    ltv = torch.full((T,), lt, device=dv)
    ph = torch.from_numpy(phi).float().to(dv)
    ll = float(model.obs_logp(model.z_features(mt, ph, ltv), obs_t).mean())
    offs = [float(model.obs_logp(model.z_features(mt, (ph + TWO_PI * k / n_off) % TWO_PI, ltv),
                                 obs_t).mean()) for k in range(1, n_off)]
    return float(math.exp(min(ll - float(np.mean(offs)), 60.0)))


@torch.no_grad()
def eval_pf(model, songs, obs_cache, K, alpha, smooth=5, seed=1234):
    rows = []
    for i, s in enumerate(songs):
        T = s["T"]
        ref = s["beats"][s["beats"] < T / FPS]
        dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 3:
            continue
        obs = torch.from_numpy(obs_cache[s["stem"]][:T]).unsqueeze(0).to(DEV)
        torch.manual_seed(seed + i)
        out = VB.particle_filter(model, obs, obs, K=K, alpha=alpha)
        m = _estimate_meter(ref, dref)
        row = dict(stem=s["stem"], dataset=s["dataset"], T=T, meter=int(m),
                   n_true=int(len(ref)), n_true_db=int(len(dref)), ess=float(out["ess"]),
                   metronome_F=f_measure(ref, metronome(T, FPS)))
        ph_bar = ideal_barphase(dref, T, FPS, mode="extrap") if len(dref) >= 2 else None
        ph_beat = ideal_barphase(ref, T, FPS, mode="extrap") if len(ref) >= 2 else None
        if ph_bar is not None:
            bar_f = float(np.median(np.diff(dref))) * FPS
            row["obs_contrast"] = contrast(model, obs[0], ph_bar, m,
                                           math.log(TWO_PI / max(bar_f, 1e-6)), T)
            row["true_phidot"] = TWO_PI / bar_f
        if ph_beat is not None:
            beat_f = float(np.median(np.diff(ref))) * FPS
            row["contrast_beat"] = contrast(model, obs[0], ph_beat, m,
                                            math.log(TWO_PI / max(beat_f, 1e-6)), T)
        for tag, phv in (("mean", out["phase_mean"].numpy()),
                         ("map", out["phase_map"].numpy()),
                         ("smooth", smooth_phase(out["phase_mean"].numpy(), smooth))):
            est = beats_from_barphase(phv, m, FPS)
            dest = downbeats_from_barphase(phv, FPS)
            b0, bb = blind_grid_controls(ref, T, len(est))
            d0, db = blind_grid_controls(dref, T, len(dest)) if len(dref) >= 2 else (np.nan, np.nan)
            pd = phase_diag(phv)
            row.update({
                f"{tag}|beat_F": f_measure(ref, est),
                f"{tag}|db_F": f_measure(dref, dest) if len(dref) >= 2 else float("nan"),
                f"{tag}|n_est": int(len(est)), f"{tag}|n_est_db": int(len(dest)),
                f"{tag}|blind0": b0, f"{tag}|blind_best": bb,
                f"{tag}|blind_db0": d0, f"{tag}|blind_db_best": db,
                f"{tag}|frac_neg": pd["frac_neg"], f"{tag}|mean_adv": pd["mean_adv"],
                f"{tag}|jitter": pd["jitter"], f"{tag}|jit_adv": pd["jitter_over_adv"],
                f"{tag}|lock_bar": _R(phv, ph_bar) if ph_bar is not None else float("nan"),
                f"{tag}|lock_beat": _R(phv, ph_beat) if ph_beat is not None else float("nan"),
                f"{tag}|phidot_ratio": (pd["mean_adv"] / row["true_phidot"]
                                        if "true_phidot" in row else float("nan"))})
        rows.append(row)
    return rows


def M(rows, k):
    v = [r[k] for r in rows if isinstance(r.get(k), float) and not math.isnan(r[k])]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True, choices=list(VIEWS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Ks", type=int, nargs="+", default=[300, 600])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--n_eval", type=int, default=40)
    a = ap.parse_args()
    tag = f"{a.view}_s{a.seed}"
    banner(f"EXP1 DEPLOY  view={a.view} seed={a.seed} dims={VIEWS[a.view]}")
    ev = load_split("eval", cap=(a.n_eval or None))
    assert all(s["fold"] == 0 for s in ev)
    obs_ev = build_obs_cache(ev, f"{ARMS}/act_eval.npz")
    ck = torch.load(f"{OUT}/exp1_{tag}.pt", map_location=DEV)
    model = CutModel(a.view, h_dim=2, hidden=ck["config"]["hidden"], num_meters=4,
                     obs_dim=2, obs_type="bern").to(DEV)
    model.load_state_dict(ck["model"]); model.eval()
    res = {"view": a.view, "seed": a.seed, "pf": {}}
    for K in a.Ks:
        for al in a.alphas:
            t1 = time.time()
            rows = eval_pf(model, ev, obs_ev, K, al)
            print(f"  PF K={K} alpha={al} ({time.time()-t1:.0f}s, {len(rows)} songs)  "
                  f"contrast_bar={M(rows,'obs_contrast'):.4f} "
                  f"contrast_beat={M(rows,'contrast_beat'):.4f}", flush=True)
            for rd in ("mean", "map", "smooth"):
                s = summarize(rows, rd)
                s["contrast_beat"] = M(rows, "contrast_beat")
                s["lock_bar"] = M(rows, f"{rd}|lock_bar")
                s["lock_beat"] = M(rows, f"{rd}|lock_beat")
                s["phidot_ratio"] = M(rows, f"{rd}|phidot_ratio")
                pr(s)
                print(f"             lock_bar={s['lock_bar']:.4f} lock_beat={s['lock_beat']:.4f} "
                      f"phidot/true={s['phidot_ratio']:.3f} contrast_beat={s['contrast_beat']:.4f}",
                      flush=True)
                res["pf"][f"K{K}_a{al}_{rd}"] = s
            res.setdefault("rows", {})[f"K{K}_a{al}"] = rows
            json.dump(res, open(f"{OUT}/exp1dep_{tag}.json", "w"), indent=1, default=float)
    json.dump(res, open(f"{OUT}/exp1dep_{tag}.json", "w"), indent=1, default=float)
    print(f"WROTE {OUT}/exp1dep_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
