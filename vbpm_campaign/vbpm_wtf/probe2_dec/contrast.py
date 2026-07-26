"""PROBE 2c -- the claim-(B) measurement itself.

Recompute obs_contrast at full precision with the arm's own code path, and add the
ANALOGOUS measurement on the log-tempo channel (roll the posterior log-tempo trace in
time instead of rotating the phase).  If phase-contrast ~= 1 but tempo-contrast >> 1, the
emission is not blind -- it is keyed to a channel obs_contrast never perturbs.
"""
from __future__ import annotations
import argparse, json, math, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms")

import variant_b as VB                                       # noqa: E402
from vbpm.evaluate import _estimate_meter                     # noqa: E402
from audit_common import load_split, ideal_barphase, FPS      # noqa: E402
from common import targets                                    # noqa: E402
from ablate import LayerMerge, posterior_Z                    # noqa: E402

DEV = "cuda:0"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
OUT = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf/probe2_dec"
TWO_PI = 2 * math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["i", "ii"])
    ap.add_argument("--n_songs", type=int, default=40)
    ap.add_argument("--frames", type=int, default=2000)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tag = f"{a.arm}_bern"
    ck = torch.load(f"{ARMS}/arm_i_{tag}.pt", map_location="cpu")
    model = VB.BarPointerVAE_B(h_dim=768 if a.arm == "i" else 2, hidden=128,
                               num_meters=4, obs_dim=2, obs_type="bern").to(DEV)
    model.load_state_dict(ck["model"]); model.eval()
    merge = LayerMerge().to(DEV); merge.load_state_dict(ck["merge"]); merge.eval()

    ev = load_split("eval", with_feats=False)[:a.n_songs]
    act = np.load(f"{ARMS}/act_eval.npz", allow_pickle=True)
    ph_c, lt_c, both_c = [], [], []
    for s in ev:
        T = min(s["T"], a.frames)
        downs = s["downs"][s["downs"] < T / FPS]
        ref = s["beats"][s["beats"] < T / FPS]
        if len(downs) < 3:
            continue
        phi = ideal_barphase(downs, T, FPS, mode="extrap")
        if phi is None:
            continue
        A = np.clip(np.asarray(act[s["stem"] + "|act"], np.float32)[:T], 1e-4, 1 - 1e-4)
        obs_t = torch.from_numpy(A).to(DEV)
        m = _estimate_meter(ref, downs)
        bar_frames = float(np.median(np.diff(downs))) * FPS
        lt = math.log(TWO_PI / max(bar_frames, 1e-6))
        mt = F.one_hot(torch.tensor([m - 1] * T, device=DEV), model.K).float()
        ltv = torch.full((T,), lt, device=DEV)
        ph = torch.from_numpy(phi).float().to(DEV)

        # (A) EXACT reproduction of arm_i.obs_contrast_song -- rotate phase only
        ll_true = float(model.obs_logp(model.z_features(mt, ph, ltv), obs_t).mean())
        offs = [float(model.obs_logp(model.z_features(mt, (ph + TWO_PI * k / 12) % TWO_PI,
                                                      ltv), obs_t).mean()) for k in range(1, 12)]
        ph_c.append(ll_true - float(np.mean(offs)))

        # (B) same protocol on the LOG-TEMPO channel: the posterior's own log-tempo trace,
        #     true alignment vs 11 time-rolled versions (same marginal, wrong timing)
        b_np, _ = targets(s["beats"], s["downs"], 0, T)
        b = torch.from_numpy(b_np).unsqueeze(0).to(DEV)
        obs_b = obs_t.unsqueeze(0)
        h = obs_b
        if a.arm == "i":
            d = np.load(s["path"], allow_pickle=True)
            h = merge(torch.from_numpy(np.asarray(d["feats"][:, :T, :], np.float32)
                                       ).unsqueeze(0).to(DEV))
            del d
        Z, PHp, LTp, MTp, _ = posterior_Z(model, h, b, 0.3)
        ltq = LTp[0]
        ll_t = float(model.obs_logp(model.z_features(mt, ph, ltq), obs_t).mean())
        offs2 = [float(model.obs_logp(model.z_features(mt, ph, torch.roll(ltq, int(T * k / 12))),
                                      obs_t).mean()) for k in range(1, 12)]
        lt_c.append(ll_t - float(np.mean(offs2)))
        # (C) posterior z as-is vs time-rolled posterior z (phase AND tempo together)
        ll_z = float(model.obs_logp(Z[0], obs_t).mean())
        offs3 = [float(model.obs_logp(torch.roll(Z[0], int(T * k / 12), dims=0), obs_t).mean())
                 for k in range(1, 12)]
        both_c.append(ll_z - float(np.mean(offs3)))

    out = dict(arm=a.arm, n_songs=len(ph_c),
               phase_contrast_logratio=float(np.mean(ph_c)),
               phase_contrast_ratio=float(np.exp(np.mean(ph_c))),
               logtempo_contrast_logratio=float(np.mean(lt_c)),
               logtempo_contrast_ratio=float(np.exp(min(np.mean(lt_c), 60))),
               full_z_contrast_logratio=float(np.mean(both_c)),
               full_z_contrast_ratio=float(np.exp(min(np.mean(both_c), 60))))
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{OUT}/contrast_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
