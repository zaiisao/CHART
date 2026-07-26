"""EXPERIMENT 1 -- CUT THE TEMPO SIDE-CHANNEL.

Established diagnosis (not re-litigated): the encoder smuggles the beat pattern through a
time-varying log_tempo; both decoders read it off there; the bar-pointer PHASE is inert
(phase ablation costs EXACTLY 0.00 nats, obs_contrast 0.9998, corr(cos phi, beats) = -0.024).

This file changes exactly ONE thing vs arm_ii (activation-only arm): WHAT THE DECODERS SEE.

  z_feat = [cos phi, sin phi, log_tempo, meter one-hot x4]      (7 dims, unchanged latent)
  --view full        decoders see all 7                          (baseline replication)
  --view cut_tempo   decoders see [cos phi, sin phi, meter]      (6 dims)  <-- THE FIX
  --view cut_phase   decoders see [log_tempo, meter]             (5 dims)  <-- MIRROR CONTROL

log_tempo stays in the LATENT and in the transition dynamics (phase advance) in every view --
only the decoder / emission INPUT changes.  Everything else (h = frozen 2-ch activation,
obs = that same activation, Bernoulli likelihood, 1200 steps, warm-up 600, ELBO, bootstrap
particle-filter deploy) is arm_ii's protocol, using arm_ii's own control/scoring code.

Nothing under vbpm/, vbpm_fix/ or vbpm_arms/ is modified.
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

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms")

import variant_b as VB                                                    # noqa: E402
from vbpm.distributions import (                                          # noqa: E402
    TWO_PI, gumbel_softmax, sample_wrapped_cauchy, sample_student_t,
)
from vbpm.evaluate import (                                               # noqa: E402
    beats_from_barphase, downbeats_from_barphase, f_measure, _estimate_meter, metronome,
)
from audit_common import load_split, banner, FPS                          # noqa: E402
from common import targets, smooth_phase                                  # noqa: E402
from arm_ii import blind_grid_controls, phase_diag, summarize, pr, obs_contrast_song  # noqa: E402

DEV = "cuda:0"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
OUT = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final"

# z_feat layout: 0=cos phi, 1=sin phi, 2=log_tempo, 3..6=meter one-hot
VIEWS = {
    "full":      [0, 1, 2, 3, 4, 5, 6],
    "cut_tempo": [0, 1, 3, 4, 5, 6],
    "cut_phase": [2, 3, 4, 5, 6],
}


# --------------------------------------------------------------------------- model
class SlicedMLP(nn.Module):
    """The SAME 1-hidden-layer tanh MLP as vbpm/model.py's decoder / variant_b's h_dec,
    but reading only a SUBSET of the z-feature dims. Slicing lives inside the module so
    VB.elbo_b / VB.particle_filter call it unchanged."""

    def __init__(self, idx, hidden, out_dim):
        super().__init__()
        self.register_buffer("idx", torch.tensor(list(idx), dtype=torch.long))
        self.net = nn.Sequential(nn.Linear(len(idx), hidden), nn.Tanh(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x.index_select(-1, self.idx))


class CutModel(VB.BarPointerVAE_B):
    def __init__(self, view, h_dim, hidden=128, num_meters=4, obs_dim=2, obs_type="bern"):
        super().__init__(h_dim=h_dim, hidden=hidden, num_meters=num_meters,
                         obs_dim=obs_dim, obs_type=obs_type)
        idx = VIEWS[view]
        self.view = view
        self.view_idx = list(idx)
        self.decoder = SlicedMLP(idx, hidden, 2)          # p(b,db | z)
        self.h_dec = SlicedMLP(idx, hidden, obs_dim)      # p(o   | z)


# --------------------------------------------------------------------------- data
def build_obs_cache(songs, act_npz):
    d = np.load(act_npz, allow_pickle=True)
    out = {}
    for s in songs:
        a = np.asarray(d[s["stem"] + "|act"], np.float32)
        assert a.shape[0] == s["T"], (s["stem"], a.shape, s["T"])
        out[s["stem"]] = np.clip(a, 1e-4, 1.0 - 1e-4)
    return out


def sample_batch(songs, obs_cache, rng, bs, frames, dev):
    bb, dd, oo = [], [], []
    while len(bb) < bs:
        s = songs[rng.integers(len(songs))]
        T = s["T"]
        if T <= frames:
            continue
        st = int(rng.integers(0, T - frames))
        b, d = targets(s["beats"], s["downs"], st, frames)
        bb.append(torch.from_numpy(b))
        dd.append(torch.from_numpy(d))
        oo.append(torch.from_numpy(obs_cache[s["stem"]][st:st + frames]))
    return (torch.stack(bb).to(dev), torch.stack(dd).to(dev), torch.stack(oo).to(dev))


# --------------------------------------------------------------------------- probes
@torch.no_grad()
def rollout(model, h, b, temp=0.3):
    """Exactly elbo_b's recursion (teacher-forced posterior), returning the Z trace."""
    B, T, _ = h.shape
    post_ctx = model.encode_posterior(h, b)
    prior_ctx = model.encode_prior(h)
    dof = model.tempo_dof()
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
        model.post_head(torch.cat([post_ctx[:, 0], z0], -1)))
    meter = gumbel_softmax(q_m, temp)
    phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
    level = sample_student_t(dof, q_lv_mu, q_lv_s)
    dev = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
    lt = level + dev
    Zs, QRHO, PRHO = [model.z_features(meter, phi, lt)], [q_ph_rho], []
    mp, pp, ltp = meter, phi, lt
    for t in range(1, T):
        zpf = model.z_features(mp, pp, ltp)
        q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
            model.post_head(torch.cat([post_ctx[:, t], zpf], -1)))
        adv = pp + torch.exp(ltp.clamp(-12, 6))
        cross = (adv >= TWO_PI).float()
        PRHO.append(model.prior_phase_conc(prior_ctx[:, t]))
        phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
        level = sample_student_t(dof, q_lv_mu, q_lv_s)
        dev = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
        lt = level + dev
        meter = torch.where(cross.unsqueeze(-1) > 0.5, gumbel_softmax(q_m, temp), mp)
        Zs.append(model.z_features(meter, phi, lt))
        QRHO.append(q_ph_rho)
        mp, pp, ltp = meter, phi, lt
    return torch.stack(Zs, 1), torch.stack(QRHO, 1), torch.stack(PRHO, 1)


def _corr(x, y):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _bce_sum(logit, tgt):
    return float(F.binary_cross_entropy_with_logits(logit, tgt, reduction="none").sum(1).mean())


@torch.no_grad()
def side_channel_probe(model, songs, obs_cache, seed, n_batches, bs, frames, label):
    """(a) phase revival: correlations + the phase/tempo ABLATION deltas on rec_beat."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    acc = {}
    n = 0
    corrs = {k: [] for k in ("cos_beat", "sin_beat", "lt_beat", "cosm_beat", "lt_db")}
    dphi_stats = []
    for _ in range(n_batches):
        b, db, o = sample_batch(songs, obs_cache, rng, bs, frames, DEV)
        h = o
        Z, qrho, prho = rollout(model, h, b)
        bn, dn = b.cpu().numpy(), db.cpu().numpy()
        cos, sin, lt = Z[..., 0].cpu().numpy(), Z[..., 1].cpu().numpy(), Z[..., 2].cpu().numpy()
        m_hat = int(Z[..., 3:].mean((0, 1)).argmax()) + 1
        ph = np.arctan2(sin, cos) % TWO_PI
        corrs["cos_beat"].append(_corr(cos, bn))
        corrs["sin_beat"].append(_corr(sin, bn))
        corrs["lt_beat"].append(_corr(lt, bn))
        corrs["lt_db"].append(_corr(lt, dn))
        corrs["cosm_beat"].append(_corr(np.cos(m_hat * ph), bn))
        d = (np.diff(ph, axis=1) + math.pi) % TWO_PI - math.pi
        dphi_stats.append((float(np.mean(d < 0)), float(d.mean()), float(d.std())))

        def score(Zx, name):
            lg = model.decoder(Zx)
            rb = _bce_sum(lg[..., 0], b)
            rd = _bce_sum(lg[..., 1], db)
            ro = float(-model.obs_logp(Zx.reshape(-1, model.z_feat_dim),
                                       o.reshape(-1, model.obs_dim)
                                       ).reshape(Zx.shape[0], -1).sum(1).mean())
            a = acc.setdefault(name, [0.0, 0.0, 0.0])
            a[0] += rb; a[1] += rd; a[2] += ro

        score(Z, "FULL z")
        Za = Z.clone()
        pr_ = torch.rand_like(Z[..., 0]) * TWO_PI
        Za[..., 0], Za[..., 1] = torch.cos(pr_), torch.sin(pr_)
        score(Za, "phase -> UNIFORM RANDOM")
        Zb = Z.clone(); Zb[..., 0], Zb[..., 1] = 1.0, 0.0
        score(Zb, "phase -> CONSTANT 0")
        Zc = Z.clone(); Zc[..., 2] = Z[..., 2].mean(1, keepdim=True)
        score(Zc, "log_tempo -> per-crop mean")
        Zd = Z.clone()
        flat = Z[..., 2].reshape(-1)
        Zd[..., 2] = flat[torch.randperm(flat.numel(), device=Z.device)].reshape(Z[..., 2].shape)
        score(Zd, "log_tempo -> shuffled")
        Ze = Z.clone(); Ze[..., 3:] = 1.0 / model.K
        score(Ze, "meter -> flat")
        n += 1
        acc.setdefault("_baserate", [0.0, 0.0, 0.0])
        acc["_baserate"][0] += float(F.binary_cross_entropy(
            b.mean().expand_as(b), b, reduction="none").sum(1).mean())
        acc["_baserate"][1] += float(F.binary_cross_entropy(
            db.mean().expand_as(db), db, reduction="none").sum(1).mean())
        acc["_baserate"][2] += float(F.binary_cross_entropy(
            o.mean(dim=(0, 1)).expand_as(o), o, reduction="none").sum((1, 2)).mean())
        acc.setdefault("_rho", [0.0, 0.0, 0.0])
        acc["_rho"][0] += float(qrho.mean()); acc["_rho"][1] += float(prho.mean())

    abl = {k: dict(rec_b=v[0] / n, rec_db=v[1] / n, rec_obs=v[2] / n) for k, v in acc.items()}
    cm = {k: float(np.mean(v)) for k, v in corrs.items()}
    dn_ = np.array(dphi_stats)
    return dict(
        label=label,
        corr_cosphi_beat=cm["cos_beat"], corr_sinphi_beat=cm["sin_beat"],
        corr_cos_m_phi_beat=cm["cosm_beat"],
        corr_logtempo_beat=cm["lt_beat"], corr_logtempo_downbeat=cm["lt_db"],
        post_rho=acc["_rho"][0] / n, prior_rho=acc["_rho"][1] / n,
        tf_frac_neg=float(dn_[:, 0].mean()), tf_mean_adv=float(dn_[:, 1].mean()),
        tf_jitter=float(dn_[:, 2].mean()),
        ablation=abl,
        d_rec_b_phase_random=abl["phase -> UNIFORM RANDOM"]["rec_b"] - abl["FULL z"]["rec_b"],
        d_rec_b_phase_const=abl["phase -> CONSTANT 0"]["rec_b"] - abl["FULL z"]["rec_b"],
        d_rec_b_tempo_flat=abl["log_tempo -> per-crop mean"]["rec_b"] - abl["FULL z"]["rec_b"],
        d_rec_b_tempo_shuf=abl["log_tempo -> shuffled"]["rec_b"] - abl["FULL z"]["rec_b"],
        d_obs_phase_random=(abl["phase -> UNIFORM RANDOM"]["rec_obs"]
                            - abl["FULL z"]["rec_obs"]) / frames,
        d_obs_tempo_flat=(abl["log_tempo -> per-crop mean"]["rec_obs"]
                          - abl["FULL z"]["rec_obs"]) / frames,
    )


def decoder_weight_report(model):
    """|w| of the FIRST layer, per z-feature group (the 'never learned to use z' check)."""
    rep = {}
    names = ["cos_phi", "sin_phi", "log_tempo", "m1", "m2", "m3", "m4"]
    for tag, mod in (("p_b", model.decoder), ("p_o", model.h_dec)):
        W = mod.net[0].weight.detach()
        idx = mod.idx.tolist()
        rep[tag] = {names[j]: float(W[:, k].abs().mean()) for k, j in enumerate(idx)}
        rep[tag + "|all"] = float(W.abs().mean())
    return rep


# --------------------------------------------------------------------------- deploy
@torch.no_grad()
def eval_pf(model, songs, obs_cache, K, alpha, smooth=5, seed=1234, max_frames=None):
    rows = []
    for i, s in enumerate(songs):
        T = s["T"] if max_frames is None else min(s["T"], max_frames)
        ref = s["beats"][s["beats"] < T / FPS]
        dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 3:
            continue
        obs = torch.from_numpy(obs_cache[s["stem"]][:T]).unsqueeze(0).to(DEV)
        h = obs                                # ARM (ii): pointer sees ONLY the activation
        torch.manual_seed(seed + i)
        out = VB.particle_filter(model, h, obs, K=K, alpha=alpha)
        m = _estimate_meter(ref, dref)
        row = dict(stem=s["stem"], dataset=s["dataset"], T=T, meter=int(m),
                   n_true=int(len(ref)), n_true_db=int(len(dref)), ess=float(out["ess"]),
                   metronome_F=f_measure(ref, metronome(T, FPS)),
                   obs_contrast=obs_contrast_song(model, obs[0], dref, ref, T))
        for tag, ph in (("mean", out["phase_mean"].numpy()),
                        ("map", out["phase_map"].numpy()),
                        ("smooth", smooth_phase(out["phase_mean"].numpy(), smooth))):
            est = beats_from_barphase(ph, m, FPS)
            dest = downbeats_from_barphase(ph, FPS)
            b0, bb = blind_grid_controls(ref, T, len(est))
            d0, db = blind_grid_controls(dref, T, len(dest)) if len(dref) >= 2 else (np.nan, np.nan)
            pd = phase_diag(ph)
            row.update({
                f"{tag}|beat_F": f_measure(ref, est),
                f"{tag}|db_F": f_measure(dref, dest) if len(dref) >= 2 else float("nan"),
                f"{tag}|n_est": int(len(est)), f"{tag}|n_est_db": int(len(dest)),
                f"{tag}|blind0": b0, f"{tag}|blind_best": bb,
                f"{tag}|blind_db0": d0, f"{tag}|blind_db_best": db,
                f"{tag}|frac_neg": pd["frac_neg"], f"{tag}|mean_adv": pd["mean_adv"],
                f"{tag}|jitter": pd["jitter"], f"{tag}|jit_adv": pd["jitter_over_adv"]})
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True, choices=list(VIEWS))
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--obs_w", type=float, default=1.0)
    ap.add_argument("--Ks", type=int, nargs="+", default=[300, 600])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--n_eval", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tag = f"{a.view}_s{a.seed}"

    banner(f"EXP1 CUT-THE-SIDE-CHANNEL view={a.view} decoder dims={VIEWS[a.view]} seed={a.seed}")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    t0 = time.time()
    train = load_split("train")
    ev = load_split("eval", cap=(a.n_eval or None))
    assert all(s["fold"] != 0 for s in train) and all(s["fold"] == 0 for s in ev)
    obs_tr = build_obs_cache(train, f"{ARMS}/act_train.npz")
    obs_ev = build_obs_cache(ev, f"{ARMS}/act_eval.npz")
    print(f"  train {len(train)}  eval {len(ev)}  ({time.time()-t0:.0f}s); "
          f"h = obs = frozen 2-ch activation, Bernoulli", flush=True)

    torch.manual_seed(a.seed)
    model = CutModel(a.view, h_dim=2, hidden=a.hidden, num_meters=4,
                     obs_dim=2, obs_type="bern").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    banner("(1) TRAIN  (train fold only)")
    t0 = time.time()
    hist = []
    for step in range(1, a.steps + 1):
        beta = min(1.0, step / a.warmup)
        temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        b, d, o = sample_batch(train, obs_tr, rng, a.bs, a.frames, DEV)
        opt.zero_grad()
        loss, info = VB.elbo_b(model, o, b, d, o, temperature=temp, beta=beta, obs_w=a.obs_w)
        if not torch.isfinite(loss):
            print("NaN @", step, flush=True)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            info["step"] = step
            hist.append(dict(info))
            print(f"  s{step:5d} loss={info['loss']:.1f} rec_b={info['recon_beat']:.1f} "
                  f"rec_db={info['recon_db']:.1f} rec_obs={info['recon_obs']:.1f} "
                  f"kl={info['kl']:.1f} kl_phi={info['kl_phase']:.1f} "
                  f"{step/(time.time()-t0):.2f} it/s", flush=True)
    model.eval()
    torch.save({"model": model.state_dict(), "view": a.view, "config": vars(a)},
               f"{OUT}/exp1_{tag}.pt")

    res = {"config": vars(a), "view_idx": VIEWS[a.view], "train_hist": hist}

    banner("(2) PROBE (a): DOES THE PHASE REVIVE?  teacher-forced posterior rollout")
    for label, songs, cache in (("train", train, obs_tr), ("eval", ev, obs_ev)):
        p = side_channel_probe(model, songs, cache, 7, 6, a.bs, a.frames, label)
        res[f"probe_{label}"] = p
        print(f"  [{label}] corr(cos phi,beat)={p['corr_cosphi_beat']:+.4f}  "
              f"corr(sin phi,beat)={p['corr_sinphi_beat']:+.4f}  "
              f"corr(cos(m*phi),beat)={p['corr_cos_m_phi_beat']:+.4f}  "
              f"corr(log_tempo,beat)={p['corr_logtempo_beat']:+.4f}")
        print(f"  [{label}] q_rho={p['post_rho']:.4f} p_rho={p['prior_rho']:.4f}  "
              f"TF phase: frac_neg={p['tf_frac_neg']:.3f} adv={p['tf_mean_adv']:+.4f} "
              f"jit={p['tf_jitter']:.4f}")
        print(f"  [{label}] ABLATION (nats / 256-frame crop):")
        for k in ("_baserate", "FULL z", "phase -> UNIFORM RANDOM", "phase -> CONSTANT 0",
                  "log_tempo -> per-crop mean", "log_tempo -> shuffled", "meter -> flat"):
            v = p["ablation"][k]
            print(f"      {k:28s} rec_b={v['rec_b']:8.2f}  rec_db={v['rec_db']:7.2f}  "
                  f"rec_obs={v['rec_obs']:9.2f}")
        print(f"  [{label}] DELTA rec_b: phase-random {p['d_rec_b_phase_random']:+.3f}  "
              f"phase-const {p['d_rec_b_phase_const']:+.3f}  "
              f"tempo-flat {p['d_rec_b_tempo_flat']:+.3f}  "
              f"tempo-shuffled {p['d_rec_b_tempo_shuf']:+.3f}", flush=True)

    res["decoder_weights"] = decoder_weight_report(model)
    print("  first-layer |w| by input group:")
    for k, v in res["decoder_weights"].items():
        print(f"      {k}: {v}")

    banner("(3) DEPLOY = BOOTSTRAP PARTICLE FILTER  (eval fold 0, full length)")
    res["pf"] = {}
    for K in a.Ks:
        for alpha in a.alphas:
            t1 = time.time()
            rows = eval_pf(model, ev, obs_ev, K, alpha)
            print(f"  PF K={K} alpha={alpha}  ({time.time()-t1:.0f}s, {len(rows)} songs)",
                  flush=True)
            for rd in ("mean", "map", "smooth"):
                s = summarize(rows, rd)
                pr(s)
                res["pf"][f"K{K}_a{alpha}_{rd}"] = s
            res.setdefault("rows", {})[f"K{K}_a{alpha}"] = rows
    json.dump(res, open(f"{OUT}/exp1_{tag}.json", "w"), indent=1, default=float)
    print(f"WROTE {OUT}/exp1_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
