"""Post-hoc mechanism diagnostics for an E3 checkpoint.

Answers "what did the VAE's transition actually learn?":
  * wrapped-Cauchy phase concentration rho on eval audio (rho -> 1 = a sharp pointer;
    rho ~ 0.5 = the init, i.e. a per-frame phase that jumps almost uniformly)
  * implied phase scale gamma = -log(rho) rad  (median per-frame phase spread)
  * tempo level / deviation innovation scales, AR coefficients, Student-t dof
  * correlation of log_tempo with the beat targets = the SIDE-CHANNEL probe (posterior
    pass, teacher-forced), and the same for cos/sin of the bar phase.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")
import e3_common as C                                              # noqa: E402
from e3_common import FPS, TWO_PI, FINAL, _estimate_meter          # noqa: E402
from e3_emission import PhaseEmission, load_act                    # noqa: E402
from e3_model import FrozenPhaseEmission, E3VAE, elbo_e3           # noqa: E402
from e3_vae import LayerMerge, build_obs, sample_batch, circ_R     # noqa: E402
from audit_common import load_split, ideal_barphase                # noqa: E402
from common import targets                                        # noqa: E402

DEV = "cuda:0"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--batches", type=int, default=8)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    train = load_split("train", with_feats=True)
    ev = load_split("eval", with_feats=True, cap=a.n)
    at = load_act("train")
    NM = cfg.get("num_meters", 3); MET0 = 2 if NM == 3 else 1
    emis_t = None
    if cfg["emission"] == "frozen":
        emis = PhaseEmission(bins_per_beat=cfg["bpb"], likelihood=cfg["lik"],
                             smooth=0.0).fit(train, at, phase_mode="downbeat")
        emis_t = FrozenPhaseEmission(emis, meters=tuple(range(MET0, MET0 + NM))).to(DEV)
    merge = LayerMerge().to(DEV); merge.load_state_dict(ck["merge"])
    model = E3VAE(h_dim=768, emission=emis_t, hidden=cfg["hidden"], num_meters=NM,
                  meter_offset=MET0,
                  drop_tempo_from_decoder=bool(cfg["drop_tempo"])).to(DEV)
    model.load_state_dict(ck["model"])
    merge.eval(); model.eval()

    out = {"ckpt": a.ckpt, "config": cfg}
    rho, s_lv, s_dv, a_dv = [], [], [], []
    with torch.no_grad():
        for s in ev:
            T = min(s["feats"].shape[1], 4000)
            f = torch.from_numpy(s["feats"][:, :T, :].astype(np.float32)).unsqueeze(0).to(DEV)
            ctx = model.encode_prior(merge(f))[0]
            rho.append(model.prior_phase_conc(ctx).cpu().numpy())
            s_lv.append(model.prior_level_scale(ctx).cpu().numpy())
            s_dv.append(model.prior_dev_scale(ctx).cpu().numpy())
            a_dv.append(model.prior_dev_coef(ctx).cpu().numpy())
            del f
    rho = np.concatenate(rho); s_lv = np.concatenate(s_lv)
    s_dv = np.concatenate(s_dv); a_dv = np.concatenate(a_dv)
    out["transition"] = dict(
        rho_mean=float(rho.mean()), rho_med=float(np.median(rho)),
        rho_p10=float(np.percentile(rho, 10)), rho_p90=float(np.percentile(rho, 90)),
        gamma_med_rad=float(-math.log(max(np.median(rho), 1e-9))),
        level_sigma_med=float(np.median(s_lv)), dev_sigma_med=float(np.median(s_dv)),
        dev_ar_med=float(np.median(a_dv)),
        level_ar=float(model.level_ar()), tempo_dof=float(model.tempo_dof()))
    print("TRANSITION", json.dumps(out["transition"], indent=1), flush=True)

    # ---- side-channel probe on the posterior (teacher-forced) ----
    obs_tr = build_obs(train, f"{ARMS}/act_train.npz")
    phi_tr = {}
    for s in train:
        p = ideal_barphase(s["downs"], s["T"], FPS, mode="extrap")
        if p is not None:
            phi_tr[s["stem"]] = p
    rng = np.random.default_rng(7)
    LT, COS, SIN, BT, PHT, PHE = [], [], [], [], [], []
    for _ in range(a.batches):
        f, b, d, o, pht = sample_batch(train, obs_tr, phi_tr, rng, 8, 256, DEV)
        with torch.no_grad():
            _, _, Z = elbo_e3(model, merge(f), b, d, o, temperature=0.3, beta=1.0,
                              want_phase=True)
        Z = Z.cpu().numpy()
        COS.append(Z[..., 0].ravel()); SIN.append(Z[..., 1].ravel())
        LT.append(Z[..., 2].ravel()); BT.append(b.cpu().numpy().ravel())
        PHT.append(pht.cpu().numpy().ravel())
        PHE.append(np.arctan2(Z[..., 1], Z[..., 0]).ravel())
    cat = lambda x: np.concatenate(x)
    lt, cs, sn, bt = cat(LT), cat(COS), cat(SIN), cat(BT)
    out["side_channel"] = dict(
        corr_logtempo_beats=float(np.corrcoef(lt, bt)[0, 1]),
        corr_cosphi_beats=float(np.corrcoef(cs, bt)[0, 1]),
        corr_sinphi_beats=float(np.corrcoef(sn, bt)[0, 1]),
        logtempo_std=float(lt.std()), logtempo_mean=float(lt.mean()),
        R_phase_vs_true=circ_R(cat(PHE), cat(PHT)))
    print("SIDE CHANNEL", json.dumps(out["side_channel"], indent=1), flush=True)
    json.dump(out, open(a.ckpt.replace(".pt", "_diag.json"), "w"), indent=1, default=float)
    print("WROTE", a.ckpt.replace(".pt", "_diag.json"), flush=True)


if __name__ == "__main__":
    main()
