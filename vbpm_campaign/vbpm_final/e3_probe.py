"""E3 mechanism probes -- run on a trained checkpoint.

P1  INSTRUMENT CHECK: is the frozen emission differentiable and does its gradient point
    back to the TRUE bar phase?  Offset the true phase by delta and read
    d(-log p(o|phi))/d(delta).  A working instrument gives a restoring force
    (gradient sign opposite to delta for small |delta|).

P2  OBJECTIVE vs OPTIMISER: evaluate the SAME trained model's ELBO terms with
      (a) its own posterior phase sample, and
      (b) the TRUE bar phase substituted for phi (everything else unchanged).
    If (b) has the LOWER loss, the training objective prefers the truth and the model is
    stuck in a local optimum (an OPTIMISATION failure).  If (b) is WORSE, the objective
    itself prefers the non-pointer phase code (a MODEL/OBJECTIVE failure).
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")
import e3_common as C                                             # noqa: E402
from e3_common import FPS, TWO_PI, FINAL, _estimate_meter         # noqa: E402
from e3_emission import PhaseEmission, load_act                   # noqa: E402
from e3_model import FrozenPhaseEmission, E3VAE, elbo_e3          # noqa: E402
from e3_vae import LayerMerge, build_obs, sample_batch, circ_R    # noqa: E402
from audit_common import load_split, ideal_barphase               # noqa: E402

DEV = "cuda:0"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--n_eval", type=int, default=10)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    MET0 = 2 if cfg.get("num_meters", 3) == 3 else 1

    train = load_split("train", with_feats=True)
    ev = load_split("eval", with_feats=True, cap=a.n_eval)
    at = load_act("train")
    emis = PhaseEmission(bins_per_beat=cfg["bpb"], likelihood=cfg["lik"],
                         smooth=0.0).fit(train, at, phase_mode="downbeat")
    emis_t = FrozenPhaseEmission(emis, meters=tuple(
        range(MET0, MET0 + cfg.get("num_meters", 3)))).to(DEV)
    merge = LayerMerge().to(DEV); merge.load_state_dict(ck["merge"])
    model = E3VAE(h_dim=768, emission=(emis_t if cfg["emission"] == "frozen" else None),
                  hidden=cfg["hidden"], num_meters=cfg.get("num_meters", 3),
                  meter_offset=MET0,
                  drop_tempo_from_decoder=bool(cfg["drop_tempo"])).to(DEV)
    model.load_state_dict(ck["model"]); merge.eval(); model.eval()
    out = {"ckpt": a.ckpt, "config": cfg}

    # ---------------- P1: restoring force of the frozen emission ----------------
    obs_ev = build_obs(ev, f"{ARMS}/act_eval.npz")
    deltas = np.array([-0.6, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, 0.6])
    grid, lls = [], []
    for s in ev:
        T = min(s["T"], 4000)
        ph = ideal_barphase(s["downs"], T, FPS, mode="extrap")
        if ph is None:
            continue
        m = _estimate_meter(s["beats"], s["downs"])
        j = max(0, min(int(m) - MET0, model.K - 1))
        o = torch.from_numpy(obs_ev[s["stem"]][:T]).to(DEV)
        mt = F.one_hot(torch.tensor([j] * T, device=DEV), model.K).float()
        ltv = torch.zeros(T, device=DEV)
        row_g, row_l = [], []
        for d in deltas:
            dt = torch.tensor(float(d), device=DEV, requires_grad=True)
            p = (torch.from_numpy(ph).float().to(DEV) + dt) % TWO_PI
            ll = model.obs_logp(model.z_features(mt, p, ltv), o).mean()
            g, = torch.autograd.grad(ll, dt)
            row_g.append(float(g)); row_l.append(float(ll))
        grid.append(row_g); lls.append(row_l)
    G = np.array(grid); L = np.array(lls)
    restoring = float(np.mean(np.sign(G) == -np.sign(deltas)[None, :]))
    out["P1_instrument"] = dict(
        deltas=deltas.tolist(), mean_dLL_ddelta=G.mean(0).tolist(),
        mean_LL=L.mean(0).tolist(), frac_restoring=restoring, n_songs=int(G.shape[0]))
    print("P1 deltas          ", np.round(deltas, 3))
    print("P1 d logp / d delta", np.round(G.mean(0), 4))
    print("P1 mean logp/frame ", np.round(L.mean(0), 4))
    print(f"P1 fraction of (song,delta) with a RESTORING gradient = {restoring:.3f}", flush=True)

    # ---------------- P2: objective at the model's phase vs the TRUE phase ------
    obs_tr = build_obs(train, f"{ARMS}/act_train.npz")
    phi_tr = {}
    for s in train:
        p = ideal_barphase(s["downs"], s["T"], FPS, mode="extrap")
        if p is not None:
            phi_tr[s["stem"]] = p
    rng = np.random.default_rng(11)
    acc = {"own": [], "true": [], "own_obs": [], "true_obs": [],
           "own_b": [], "true_b": [], "R": [],
           "truem": [], "truem_obs": [], "truem_b": []}
    mgt_cache = {s["stem"]: _estimate_meter(s["beats"], s["downs"]) for s in train}
    for _ in range(a.batches):
        f, b, d, o, pht = sample_batch(train, obs_tr, phi_tr, rng, 8, 256, DEV)
        with torch.no_grad():
            h = merge(f)
            _, info, Z = elbo_e3(model, h, b, d, o, temperature=0.3, beta=1.0,
                                 want_phase=True)
            B, T, _ = Z.shape
            Zt = Z.clone()
            Zt[..., 0] = torch.cos(pht); Zt[..., 1] = torch.sin(pht)
            def terms(ZZ):
                lg = model.decoder(model.dec_feat(ZZ))
                rb = F.binary_cross_entropy_with_logits(lg[..., 0], b, reduction="none").sum(1)
                rd = F.binary_cross_entropy_with_logits(lg[..., 1], d, reduction="none").sum(1)
                ol = model.obs_logp(ZZ.reshape(B * T, -1), o.reshape(B * T, -1)).reshape(B, T).sum(1)
                return float((rb + rd).mean()), float((-ol).mean())
            rb0, ro0 = terms(Z)
            rb1, ro1 = terms(Zt)
            # true phase AND the majority GT meter (removes the meter-collapse confound)
            Ztm = Zt.clone()
            jgt = max(0, min(4 - MET0, model.K - 1))
            Ztm[..., 3:] = 0.0
            Ztm[..., 3 + jgt] = 1.0
            rb2, ro2 = terms(Ztm)
        acc["own"].append(rb0 + ro0); acc["true"].append(rb1 + ro1)
        acc["own_obs"].append(ro0); acc["true_obs"].append(ro1)
        acc["own_b"].append(rb0); acc["true_b"].append(rb1)
        acc["truem"].append(rb2 + ro2); acc["truem_obs"].append(ro2)
        acc["truem_b"].append(rb2)
        acc["R"].append(circ_R(np.arctan2(Z[..., 1].cpu().numpy(), Z[..., 0].cpu().numpy()).ravel(),
                               pht.cpu().numpy().ravel()))
    M = {k: float(np.mean(v)) for k, v in acc.items()}
    M["delta_recon_true_minus_own"] = M["true"] - M["own"]
    M["delta_recon_truephase_meter4_minus_own"] = M["truem"] - M["own"]
    out["P2_objective"] = M
    print("\nP2  (per 256-frame crop, lower = better)")
    print(f"    model's own phase : recon_beat+db {M['own_b']:9.1f}   -log p(o|phi) {M['own_obs']:9.1f}"
          f"   total {M['own']:9.1f}")
    print(f"    TRUE bar phase    : recon_beat+db {M['true_b']:9.1f}   -log p(o|phi) {M['true_obs']:9.1f}"
          f"   total {M['true']:9.1f}")
    print(f"    delta(true - own) = {M['delta_recon_true_minus_own']:+.1f}  "
          f"(<0 => the OBJECTIVE prefers the truth and this is an OPTIMISATION failure; "
          f">0 => the OBJECTIVE prefers the learned code)")
    print(f"    TRUE phase + meter=4: recon_beat+db {M['truem_b']:9.1f}   "
          f"-log p(o|phi) {M['truem_obs']:9.1f}   total {M['truem']:9.1f}"
          f"   delta {M['delta_recon_truephase_meter4_minus_own']:+.1f}")
    print(f"    R(phi_model, phi_true) = {M['R']:.3f}", flush=True)
    json.dump(out, open(a.ckpt.replace(".pt", "_probe.json"), "w"), indent=1, default=float)
    print("WROTE", a.ckpt.replace(".pt", "_probe.json"))


if __name__ == "__main__":
    main()
