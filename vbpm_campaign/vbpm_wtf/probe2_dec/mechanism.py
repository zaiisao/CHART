"""PROBE 2b -- mechanism: the log-tempo Morse channel, and the exact phase-flatness of BOTH decoders.

1. response surface of decoder / emission over (log_tempo x phase)  -> is phase EXACTLY flat?
2. posterior log_tempo distribution vs the deploy prior support band
3. 1-bit "log_tempo > thr" reference predictor  -> how much of rec_beat it explains
4. rec_beat when log_tempo is clamped to the deploy band (deploy-realistic latent)
5. my own obs_contrast (phase) AND the analogous tempo-contrast
"""
from __future__ import annotations
import argparse, json, math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms")

import variant_b as VB                                        # noqa: E402
from vbpm.distributions import TWO_PI                          # noqa: E402
from audit_common import load_split                            # noqa: E402
from common import targets                                     # noqa: E402
from ablate import LayerMerge, posterior_Z                     # noqa: E402

DEV = "cuda:0"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
OUT = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf/probe2_dec"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["i", "ii"])
    ap.add_argument("--crops_per_song", type=int, default=4)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--split", default="eval")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tag = f"{a.arm}_bern"
    ck = torch.load(f"{ARMS}/arm_i_{tag}.pt", map_location="cpu")
    model = VB.BarPointerVAE_B(h_dim=768 if a.arm == "i" else 2, hidden=128,
                               num_meters=4, obs_dim=2, obs_type="bern").to(DEV)
    model.load_state_dict(ck["model"]); model.eval()
    merge = LayerMerge().to(DEV); merge.load_state_dict(ck["merge"]); merge.eval()
    out = {"arm": a.arm, "split": a.split}

    # ---------- 1. response surface over (log_tempo x phase), meter = 4/4 ----------
    lts = torch.linspace(-6.0, 1.0, 29, device=DEV)
    phs = torch.linspace(0, TWO_PI, 33, device=DEV)[:-1]
    LT, PH = torch.meshgrid(lts, phs, indexing="ij")
    m = torch.zeros(LT.numel(), 4, device=DEV); m[:, 3] = 1.0
    zf = model.z_features(m, PH.reshape(-1), LT.reshape(-1))
    dec = model.decoder(zf).reshape(len(lts), len(phs), 2)
    emi = model.h_dec(zf).reshape(len(lts), len(phs), 2)
    surf = []
    for i, lt in enumerate(lts.tolist()):
        surf.append(dict(
            log_tempo=round(lt, 3),
            beat_p_mean=round(float(torch.sigmoid(dec[i, :, 0]).mean()), 6),
            beat_logit_range_over_phase=round(float(dec[i, :, 0].max() - dec[i, :, 0].min()), 8),
            emis_beat_logit_mean=round(float(emi[i, :, 0].mean()), 5),
            emis_beat_logit_range_over_phase=round(float(emi[i, :, 0].max() - emi[i, :, 0].min()), 8),
            emis_db_logit_range_over_phase=round(float(emi[i, :, 1].max() - emi[i, :, 1].min()), 8)))
    out["surface"] = surf
    out["max_beat_logit_range_over_phase_any_tempo"] = round(
        float((dec[:, :, 0].max(1).values - dec[:, :, 0].min(1).values).max()), 8)
    out["max_emis_logit_range_over_phase_any_tempo"] = round(
        float((emi[:, :, 0].max(1).values - emi[:, :, 0].min(1).values).max()), 8)
    # first-layer weight columns for cos/sin vs tempo, and their pre-activation contribution
    W = model.decoder[0].weight.detach()
    Wh = model.h_dec[0].weight.detach()
    out["dec_W0_col_absmean"] = [round(float(x), 5) for x in W.abs().mean(0).cpu()]
    out["emis_W0_col_absmean"] = [round(float(x), 5) for x in Wh.abs().mean(0).cpu()]

    # ---------- data ----------
    ev = load_split(a.split, with_feats=False)
    act = np.load(f"{ARMS}/act_{'eval' if a.split == 'eval' else 'train'}.npz",
                  allow_pickle=True)
    rng = np.random.default_rng(11)
    T = a.frames
    LTs, Bs, DBs, recs = [], [], [], {}
    for s in ev:
        d = np.load(s["path"], allow_pickle=True)
        feats = d["feats"] if a.arm == "i" else None
        A = np.clip(np.asarray(act[s["stem"] + "|act"], np.float32), 1e-4, 1 - 1e-4)
        if s["T"] <= T + 1:
            continue
        starts = rng.integers(0, s["T"] - T, size=a.crops_per_song)
        bb, dd, oo, ff = [], [], [], []
        for st in starts:
            bt, dt = targets(s["beats"], s["downs"], int(st), T)
            bb.append(bt); dd.append(dt); oo.append(A[st:st + T])
            if feats is not None:
                ff.append(np.asarray(feats[:, st:st + T, :], np.float32))
        b = torch.from_numpy(np.stack(bb)).to(DEV)
        db = torch.from_numpy(np.stack(dd)).to(DEV)
        obs = torch.from_numpy(np.stack(oo)).to(DEV)
        h = merge(torch.from_numpy(np.stack(ff)).to(DEV)) if a.arm == "i" else obs
        Z, PHt, LTt, MT, RHO = posterior_Z(model, h, b, a.temp)
        LTs.append(LTt.cpu().numpy()); Bs.append(b.cpu().numpy()); DBs.append(db.cpu().numpy())
        # rec with log_tempo clamped into the DEPLOY prior support band
        for nm, lo, hi in (("deployband", -3.55, -2.18), ("wide", -5.0, -1.0)):
            Zc = Z.clone(); Zc[..., 2] = Z[..., 2].clamp(lo, hi)
            lg = model.decoder(Zc)
            r = F.binary_cross_entropy_with_logits(lg[..., 0], b, reduction="none").sum(1)
            ro = -model.obs_logp(Zc.reshape(-1, 7), obs.reshape(-1, 2)).reshape(b.shape).sum(1)
            recs.setdefault(f"rec_beat|lt_clamp_{nm}", []).append(r.cpu().numpy())
            recs.setdefault(f"rec_obs|lt_clamp_{nm}", []).append(ro.cpu().numpy())
        lg = model.decoder(Z)
        recs.setdefault("rec_beat|none", []).append(
            F.binary_cross_entropy_with_logits(lg[..., 0], b, reduction="none").sum(1).cpu().numpy())
        recs.setdefault("rec_obs|none", []).append(
            (-model.obs_logp(Z.reshape(-1, 7), obs.reshape(-1, 2)).reshape(b.shape).sum(1)).cpu().numpy())
        del feats, d
    for k, v in recs.items():
        out[k] = float(np.concatenate(v).mean())

    LTf = np.concatenate(LTs); Bf = np.concatenate(Bs)
    q = np.quantile(LTf, [0.01, 0.1, 0.5, 0.9, 0.95, 0.99])
    out["logtempo_quantiles_1_10_50_90_95_99"] = [round(float(x), 3) for x in q]
    out["frac_logtempo_in_deploy_band"] = float(np.mean((LTf > -3.55) & (LTf < -2.18)))
    out["frac_logtempo_above_-2.0"] = float(np.mean(LTf > -2.0))
    out["frac_beatframes_logtempo_above_-2.0"] = float(np.mean(LTf[Bf > 0.5] > -2.0))
    out["frac_nonbeatframes_logtempo_above_-2.0"] = float(np.mean(LTf[Bf < 0.5] > -2.0))
    # implied phase advance rad/frame -> "BPM" at m=4
    out["implied_bpm_at_beat_frames"] = float(
        np.median(np.exp(LTf[Bf > 0.5])) / TWO_PI * 50.0 * 4 * 60.0)
    out["implied_bpm_median_all"] = float(np.median(np.exp(LTf)) / TWO_PI * 50.0 * 4 * 60.0)

    # ---------- 3. one-bit log_tempo threshold predictor ----------
    best = None
    for thr in np.quantile(LTf, np.linspace(0.5, 0.999, 60)):
        hi = LTf > thr
        p1 = max(Bf[hi].mean() if hi.any() else 0.0, 1e-9)
        p0 = max(Bf[~hi].mean(), 1e-9)
        p1 = min(p1, 1 - 1e-9); p0 = min(p0, 1 - 1e-9)
        pr = np.where(hi, p1, p0)
        bce = -(Bf * np.log(pr) + (1 - Bf) * np.log(1 - pr)).sum(1).mean()
        if best is None or bce < best[0]:
            best = (float(bce), float(thr), float(p1), float(p0), float(hi.mean()))
    out["onebit_logtempo_BCE_beat"], out["onebit_thr"], out["onebit_p_hi"], \
        out["onebit_p_lo"], out["onebit_frac_hi"] = best

    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{OUT}/mechanism_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
