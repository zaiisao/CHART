"""EXPERIMENT 3 driver -- train the VAE with BOTH fixes, then deploy with the PF.

  --drop_tempo 1 --emission frozen   = the E3 model  (both fixes)          <- (ii)
  --drop_tempo 0 --emission frozen   = fix 2 only    (side-channel intact)
  --drop_tempo 1 --emission learned  = fix 1 only    (emission still learned end-to-end)
  --drop_tempo 0 --emission learned  = the broken baseline (== arm i, head_gauss)

Deploy = bootstrap particle filter over the model's OWN learned prior transition,
weights p(o_t|z_t).  Read-outs: weighted circular mean, per-frame MAP, ancestral path,
smoothed mean.  All scored through e3_common with the mandatory blind controls.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")
import e3_common as C                                                    # noqa: E402
from e3_common import (FPS, TWO_PI, METERS, FINAL, _estimate_meter,       # noqa: E402
                       score_traj, summarize, pr)
from e3_emission import PhaseEmission, load_act, obs_contrast            # noqa: E402
from e3_model import FrozenPhaseEmission, E3VAE, elbo_e3                 # noqa: E402
from e3_pf_learned import particle_filter_learned                        # noqa: E402
from audit_common import load_split, ideal_barphase                      # noqa: E402
from common import targets, smooth_phase                                 # noqa: E402

DEV = "cuda:0"      # physical GPU selected with CUDA_VISIBLE_DEVICES in the shell
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"


class LayerMerge(nn.Module):
    """learnable softmax over the 13 MERT layers -> [B,T,768] (same as vbpm_arms/arm_i)."""

    def __init__(self, n_layers=13):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, feats):
        return torch.einsum("l,bltf->btf", torch.softmax(self.layer_logits, 0), feats)

    def weights(self):
        return torch.softmax(self.layer_logits.detach(), 0).cpu().numpy()


def build_obs(songs, act_npz):
    """observation o_t = LOGIT of the frozen 2-channel activation (matches the
    supervised gauss emission, which is fitted on logit(a))."""
    d = np.load(act_npz, allow_pickle=True)
    out = {}
    for s in songs:
        a = np.asarray(d[s["stem"] + "|act"], np.float32)
        a = np.clip(a, 1e-4, 1.0 - 1e-4)
        out[s["stem"]] = np.log(a / (1.0 - a))
    return out


def sample_batch(train, obs_cache, phi_cache, rng, bs, frames, dev):
    fe, bb, dd, oo, pp = [], [], [], [], []
    while len(fe) < bs:
        s = train[rng.integers(len(train))]
        T = s["feats"].shape[1]
        if T <= frames or s["stem"] not in phi_cache:
            continue
        st = int(rng.integers(0, T - frames))
        fe.append(torch.from_numpy(s["feats"][:, st:st + frames, :].astype(np.float32)))
        b, d = targets(s["beats"], s["downs"], st, frames)
        bb.append(torch.from_numpy(b)); dd.append(torch.from_numpy(d))
        oo.append(torch.from_numpy(obs_cache[s["stem"]][st:st + frames]))
        pp.append(torch.from_numpy(phi_cache[s["stem"]][st:st + frames].astype(np.float32)))
    return (torch.stack(fe).to(dev), torch.stack(bb).to(dev), torch.stack(dd).to(dev),
            torch.stack(oo).to(dev), torch.stack(pp).to(dev))


@torch.no_grad()
def model_obs_contrast(model, obs_t, downs, ref, T, n_off=12):
    """obs_contrast measured THROUGH the model's own obs_logp (the deploy instrument)."""
    if len(downs) < 3:
        return float("nan")
    phi = ideal_barphase(downs, T, FPS, mode="extrap")
    if phi is None:
        return float("nan")
    m = _estimate_meter(ref, downs)
    bar_frames = float(np.median(np.diff(downs))) * FPS
    lt = math.log(TWO_PI / max(bar_frames, 1e-6))
    dv = obs_t.device
    j = int(m) - getattr(model, "meter_offset", 1)
    j = max(0, min(j, model.K - 1))
    mt = F.one_hot(torch.tensor([j] * T, device=dv), model.K).float()
    ltv = torch.full((T,), lt, device=dv)
    ph = torch.from_numpy(phi).float().to(dv)
    ll_true = float(model.obs_logp(model.z_features(mt, ph, ltv), obs_t).mean())
    offs = [float(model.obs_logp(model.z_features(
        mt, (ph + TWO_PI * k / n_off) % TWO_PI, ltv), obs_t).mean()) for k in range(1, n_off)]
    return float(math.exp(min(ll_true - float(np.mean(offs)), 60.0)))


def circ_R(a, b):
    """phase-locking resultant |E exp(i(a-b))| in [0,1]; 1 = perfectly aligned."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(abs(np.mean(np.exp(1j * d))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop_tempo", type=int, default=1)
    ap.add_argument("--emission", default="frozen", choices=["frozen", "learned"])
    ap.add_argument("--lik", default="gauss", choices=["gauss", "bern"])
    ap.add_argument("--bpb", type=int, default=24)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--obs_w", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=600)
    ap.add_argument("--num_meters", type=int, default=3)   # classes = 2,3,4 (no flat class)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.25])
    ap.add_argument("--n_eval", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    MET0 = 2 if a.num_meters == 3 else 1     # meter VALUE of latent class 0
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    train = load_split("train", with_feats=True)
    ev = load_split("eval", with_feats=True, cap=(a.n_eval or None))
    assert all(s["fold"] != 0 for s in train) and all(s["fold"] == 0 for s in ev)
    print(f"train {len(train)}  eval {len(ev)}  loaded {time.time()-t0:.0f}s", flush=True)

    obs_tr = build_obs(train, f"{ARMS}/act_train.npz")
    obs_ev = build_obs(ev, f"{ARMS}/act_eval.npz")
    phi_tr = {}
    for s in train:
        p = ideal_barphase(s["downs"], s["T"], FPS, mode="extrap")
        if p is not None:
            phi_tr[s["stem"]] = p

    # ---------- frozen supervised emission (fitted on the TRAIN fold only) ----------
    emis_t = None
    if a.emission == "frozen":
        at = load_act("train")
        emis = PhaseEmission(bins_per_beat=a.bpb, likelihood=a.lik,
                             smooth=0.0).fit(train, at, phase_mode="downbeat")
        c_tr, _ = obs_contrast(emis, train, at, phase_mode="downbeat")
        ae = load_act("eval")
        c_ev, _ = obs_contrast(emis, ev, ae, phase_mode="downbeat")
        print(f"SUPERVISED EMISSION lik={a.lik} bpb={a.bpb} songs/meter={emis.n_used} "
              f"table obs_contrast train={c_tr:.4f} eval={c_ev:.4f}", flush=True)
        emis_t = FrozenPhaseEmission(emis, meters=tuple(
            range(MET0, MET0 + a.num_meters))).to(DEV)

    merge = LayerMerge().to(DEV)
    model = E3VAE(h_dim=768, emission=emis_t, hidden=a.hidden,
                  num_meters=a.num_meters, meter_offset=MET0,
                  drop_tempo_from_decoder=bool(a.drop_tempo)).to(DEV)
    print(f"MODEL drop_tempo={a.drop_tempo} emission={a.emission} "
          f"meters={[MET0 + i for i in range(a.num_meters)]} "
          f"decoder_in={model.decoder[0].in_features}", flush=True)

    # instrument check: contrast of the FROZEN emission through model.obs_logp
    with torch.no_grad():
        s0 = ev[0]
        o0 = torch.from_numpy(obs_ev[s0["stem"]][:s0["T"]]).to(DEV)
        print(f"  instrument check obs_contrast(song0, untrained model) = "
              f"{model_obs_contrast(model, o0, s0['downs'], s0['beats'], s0['T']):.4f}",
              flush=True)

    params = [p for p in list(merge.parameters()) + list(model.parameters())
              if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)

    print("=" * 78 + "\n(1) TRAIN (train fold only)\n" + "=" * 78, flush=True)
    hist = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        beta = min(1.0, step / a.warmup)
        temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        f, b, d, o, phT = sample_batch(train, obs_tr, phi_tr, rng, a.bs, a.frames, DEV)
        opt.zero_grad()
        loss, info, Z = elbo_e3(model, merge(f), b, d, o, temperature=temp, beta=beta,
                                obs_w=a.obs_w, want_phase=True)
        if not torch.isfinite(loss):
            print("NaN @", step, flush=True); break
        loss.backward()
        log_now = (step % 100 == 0 or step == 1)
        gn = {}
        if log_now:      # ---- gradient audit: does the phase actually get gradient? ----
            for nm, mod in (("post_head", model.post_head), ("decoder", model.decoder),
                            ("prior_phase_rho", model.prior_phase_rho)):
                g = [p.grad for p in mod.parameters() if p.grad is not None]
                gn[nm] = float(sum(float(x.norm()) for x in g))
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if log_now:
            with torch.no_grad():
                phi_est = torch.atan2(Z[..., 1], Z[..., 0]).cpu().numpy()
            R = circ_R(phi_est.ravel(), phT.cpu().numpy().ravel())
            with torch.no_grad():
                mmass = Z[..., 3:].mean(dim=(0, 1)).cpu().numpy()
            info["obs_per_frame"] = info["recon_obs"] / a.frames
            row = dict(step=step, R_phase=R, meter_mass=mmass.tolist(),
                       **info, **{f"g_{k}": v for k, v in gn.items()})
            hist.append(row)
            print(f"  s{step:5d} loss={info['loss']:.1f} rec_b={info['recon_beat']:.1f} "
                  f"rec_db={info['recon_db']:.1f} rec_obs={info['recon_obs']:.1f} "
                  f"kl={info['kl']:.1f} kl_phi={info['kl_phase']:.1f} "
                  f"obs/fr={info['obs_per_frame']:.3f} R(phi,phi_true)={R:.3f} "
                  f"m={np.array2string(mmass, precision=2)} |g|dec={gn['decoder']:.2f} "
                  f"|g|post={gn['post_head']:.2f} {step/(time.time()-t0):.2f} it/s", flush=True)
    merge.eval(); model.eval()
    lw = merge.weights()
    torch.save({"merge": merge.state_dict(), "model": model.state_dict(),
                "config": vars(a)}, f"{FINAL}/{a.tag}.pt")

    print("=" * 78 + "\n(2) DEPLOY = BOOTSTRAP PARTICLE FILTER (eval fold 0, full length)\n"
          + "=" * 78, flush=True)
    res = {"config": vars(a), "layer_w": lw.tolist(), "train_hist": hist}
    for alpha in a.alphas:
        rows = {k: [] for k in ("mean", "map", "path", "smooth_mean")}
        t1 = time.time()
        for i, s in enumerate(ev):
            T = s["feats"].shape[1]
            ref = s["beats"][s["beats"] < T / FPS]
            dref = s["downs"][s["downs"] < T / FPS]
            if len(ref) < 3:
                continue
            m_gt = _estimate_meter(ref, dref)
            fT = torch.from_numpy(s["feats"][:, :T, :].astype(np.float32)).unsqueeze(0).to(DEV)
            with torch.no_grad():
                h = merge(fT)
                obs = torch.from_numpy(obs_ev[s["stem"]][:T]).unsqueeze(0).to(DEV)
                torch.manual_seed(1234 + i)
                out = particle_filter_learned(model, h, obs, K=a.K, alpha=alpha)
                oc = model_obs_contrast(model, obs[0], dref, ref, T)
            m_pf = int(np.bincount(out["meter_path"]).argmax())
            base = dict(stem=s["stem"], dataset=s["dataset"], T=T, n_true=len(ref),
                        n_true_db=len(dref), ess=out["ess"], obs_contrast=oc,
                        meter_ok=float(m_pf == m_gt))
            for k, ph in (("mean", out["phase_mean"]), ("map", out["phase_map"]),
                          ("path", out["phase_path"]),
                          ("smooth_mean", smooth_phase(out["phase_mean"], 5))):
                rows[k].append({**base, **score_traj(ph, m_gt, ref, dref, T)})
            del fT, h, obs
            if i % 20 == 0:
                print(f"    {i}/{len(ev)} {time.time()-t1:.0f}s", flush=True)
        print(f"  PF alpha={alpha} K={a.K} ({time.time()-t1:.0f}s)", flush=True)
        for k, rr in rows.items():
            d = summarize(rr, f"E3 learned-trans {k} a={alpha}")
            pr(d)
            res.setdefault("pf", {})[f"a{alpha}_{k}"] = d
        res.setdefault("rows", {})[f"a{alpha}"] = rows["path"]
    json.dump(res, open(f"{FINAL}/{a.tag}.json", "w"), indent=1, default=float)
    print("WROTE", a.tag + ".json", flush=True)


if __name__ == "__main__":
    main()
