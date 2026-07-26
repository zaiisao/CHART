"""EVIDENCE-OFF CONTROL (alpha = 0).

Comparing the particle filter to vbpm.free_run confounds three changes at once:
evidence, the K-particle machinery, and the read-out.  Running the SAME particle filter
with the observation term switched off (alpha=0 => w is uniform, no resampling pressure)
holds machinery and read-out fixed and varies ONLY "does evidence reach the state".
Any gain that survives alpha=0 -> alpha=alpha* is attributable to FILTERING.

Also re-reports the density-matched phase-blind floor at each setting.
"""
import argparse, json, math, sys
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.elbo import free_run
from vbpm.evaluate import metronome, f_measure, _estimate_meter
from vbpm_fix.varB_pf.vb import BarPointerVAE_B, MertFront, particle_filter
from vbpm_fix.varB_pf import common as C

FPS = C.FPS
DEV = "cuda:0"


@torch.no_grad()
def run(mode, ckpt, alphas, n_eval, cap, K, n_harm, hidden, seed=0):
    ev = C.load("eval", n_eval, with_feats=(mode == "mert"))
    if mode == "dirac":
        model = BarPointerVAE_B(h_dim=C.H_DIM_DIRAC, hidden=hidden, num_meters=4,
                                obs_mode="bern", obs_dim=2, n_harm=n_harm).to(DEV)
        model.load_state_dict(torch.load(ckpt, map_location=DEV)); front = None
    else:
        sd = torch.load(ckpt, map_location=DEV)
        front = MertFront().to(DEV); front.load_state_dict(sd["front"])
        model = BarPointerVAE_B(h_dim=768, hidden=hidden, num_meters=4,
                                obs_mode="gauss", obs_dim=768, n_harm=n_harm).to(DEV)
        model.load_state_dict(sd["model"])
    model.eval()
    if front is not None:
        front.eval()

    out = {}
    for al in alphas:
        res = {k: [] for k in ["pf_anc", "pf_anc_mono", "pf_circ", "pf_circ_mono"]}
        bpm_r = []
        for i, s in enumerate(ev):
            T = min(s["T"] if mode == "dirac" else s["feats"].shape[1], cap)
            ref = s["beats"][s["beats"] < T / FPS]; dref = s["downs"][s["downs"] < T / FPS]
            if len(ref) < 2:
                continue
            m = _estimate_meter(ref, dref)
            if mode == "dirac":
                h = torch.from_numpy(C.dirac_h(s["beats"], s["downs"], 0, T,
                                               np.random.default_rng(1000 + i))).unsqueeze(0).to(DEV)
                obs = C.dirac_obs(h)
            else:
                f = torch.from_numpy(s["feats"][:, :T, :].astype(np.float32)).unsqueeze(0).to(DEV)
                h = front(f); obs = h
            torch.manual_seed(seed)
            pf = particle_filter(model, h, obs, K=K, alpha=al)
            for nm, key, mono in [("pf_anc", "anc", 0), ("pf_circ", "circ", 0),
                                  ("pf_anc_mono", "anc", 1), ("pf_circ_mono", "circ", 1)]:
                p = C.monotonise(pf[key]) if mono else pf[key]
                res[nm].append(C.score_phase(p, ref, dref, m, T, tag_seed=i))
            tb = 60.0 / np.median(np.diff(ref)) if len(ref) > 2 else np.nan
            bpm_r.append(60.0 * FPS * m * math.exp(float(np.median(pf["anc_log_tempo"])))
                         / (2 * math.pi) / tb)
        o = {k: C.summarize(v) for k, v in res.items()}
        o["pf_bpm_ratio"] = float(np.median(bpm_r))
        out[str(al)] = o
        print(f"alpha={al}: " + "  ".join(
            f"{k} F={o[k]['beat_F']:.3f}/blind={o[k]['blind_floor']:.3f}/nr={o[k]['n_ratio']:.2f}"
            for k in ["pf_anc_mono", "pf_circ_mono"]) + f"  bpm_ratio={o['pf_bpm_ratio']:.2f}",
            flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dirac")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alphas", default="0,1.0")
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--cap", type=int, default=1600)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--n_harm", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = run(a.mode, a.ckpt, [float(x) for x in a.alphas.split(",")],
            a.n_eval, a.cap, a.K, a.n_harm, a.hidden)
    json.dump(r, open(a.out, "w"), indent=2)
    print("wrote", a.out)
