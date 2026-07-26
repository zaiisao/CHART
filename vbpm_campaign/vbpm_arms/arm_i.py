"""ARM (i) -- RICH CONDITIONING (full merged MERT h) + BEAT-SALIENT OBSERVATION.

  prior p_psi(z|h) / posterior q_phi(z|h,b)   <-  FULL merged MERT [T,768]
  observation p_theta(o_t|z_t)                <-  frozen shared act head's [T,2] activation
  emission net                                 z_feat(7) -> 128 -> obs_dim
  deploy                                       VB.particle_filter, weights p(o_t|z_t)

Three --obs modes (run in parallel on separate GPUs):
  head_bern  : o = the [0,1] activation, Bernoulli likelihood
  head_gauss : o = logit(activation),    Gaussian likelihood (learned per-dim sigma)
  pca_gauss  : CONTROL = the OLD fixed PCA-32 MERT projection (beat-blind), Gaussian.
               Re-run under THIS arm's protocol (79 eval songs, FULL length) so the
               "old observation" comparison is apples-to-apples instead of the
               25-song / 1200-frame protocol the 0.449 number came from.

Nothing in vbpm/ or vbpm_fix/variant_b.py is modified.
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

import variant_b as VB                                              # noqa: E402
from vbpm.evaluate import (                                          # noqa: E402
    beats_from_barphase, downbeats_from_barphase, f_measure, _estimate_meter, metronome,
)
from audit_common import load_split, ideal_barphase, banner, FPS     # noqa: E402
from common import targets, smooth_phase                             # noqa: E402

DEV = "cuda:0"                       # physical GPU chosen by CUDA_VISIBLE_DEVICES in the shell
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- h merge
class LayerMerge(nn.Module):
    """learnable softmax over the 13 MERT layers -> [B,T,768]  (the RICH conditioning h)."""

    def __init__(self, n_layers=13):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, feats):                                   # [B,13,T,768] -> [B,T,768]
        return torch.einsum("l,bltf->btf", torch.softmax(self.layer_logits, 0), feats)

    def weights(self):
        return torch.softmax(self.layer_logits.detach(), 0).cpu().numpy()


class FixedProj(nn.Module):
    """CONTROL observation target: fixed PCA-32 of the uniform-layer-mean MERT, z-scored."""

    def __init__(self, mean, comps):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("comps", comps)

    @torch.no_grad()
    def forward(self, feats):                                   # [B,13,T,768] -> [B,T,32]
        x = feats.mean(1)
        o = (x - self.mean) @ self.comps
        return (o - o.mean(1, keepdim=True)) / o.std(1, keepdim=True).clamp(min=1e-4)


def fit_pca(train, rng, n_songs=48, n_frames=250, obs_dim=32):
    X = []
    for s in train[:n_songs]:
        f = s["feats"]
        T = f.shape[1]
        idx = rng.choice(T, size=min(n_frames, T), replace=False)
        X.append(np.asarray(f[:, idx, :], np.float32).mean(0))
    X = torch.from_numpy(np.concatenate(X, 0))
    mu = X.mean(0)
    _, _, V = torch.pca_lowrank(X - mu, q=obs_dim, niter=4)
    return mu, V[:, :obs_dim].contiguous()


# --------------------------------------------------------------------------- controls
def blind_grid_controls(ref, T, n_est, n_off=12):
    """MANDATORY density-matched blind control: uniform grid with the SAME beat count.
    Returns (F at offset 0, best-of-n_off-offsets F)."""
    dur = T / FPS
    if n_est < 2 or len(ref) < 2:
        return float("nan"), float("nan")
    per = dur / n_est
    base = np.arange(n_est) * per
    f0 = f_measure(ref, base)
    best = max(f_measure(ref, base + k * per / n_off) for k in range(n_off))
    return float(f0), float(max(best, f0))


def phase_diag(ph):
    """PF phase pathology diagnostics on a bar-phase trajectory."""
    d = (np.diff(np.asarray(ph, float)) + math.pi) % TWO_PI - math.pi
    if len(d) == 0:
        return dict(frac_neg=float("nan"), mean_adv=float("nan"),
                    jitter=float("nan"), jitter_over_adv=float("nan"))
    adv = float(d.mean())
    return dict(frac_neg=float(np.mean(d < 0)), mean_adv=adv, jitter=float(d.std()),
                jitter_over_adv=float(d.std() / max(abs(adv), 1e-9)))


def M(rows, k):
    v = [r[k] for r in rows if k in r and isinstance(r[k], float) and not math.isnan(r[k])]
    return float(np.mean(v)) if v else float("nan")


# --------------------------------------------------------------------------- data
def build_obs_cache(songs, act_npz, mode):
    """Per-song observation array [T,obs_dim] (head modes only; pca is computed on GPU)."""
    d = np.load(act_npz, allow_pickle=True)
    out = {}
    for s in songs:
        a = np.asarray(d[s["stem"] + "|act"], np.float32)
        assert a.shape[0] == s["T"], (s["stem"], a.shape, s["T"])
        a = np.clip(a, 1e-4, 1.0 - 1e-4)
        out[s["stem"]] = a if mode == "head_bern" else np.log(a / (1.0 - a))
    return out


def sample_batch(train, obs_cache, rng, bs, frames, dev):
    fe, bb, dd, oo = [], [], [], []
    while len(fe) < bs:
        s = train[rng.integers(len(train))]
        T = s["feats"].shape[1]
        if T <= frames:
            continue
        st = int(rng.integers(0, T - frames))
        fe.append(torch.from_numpy(s["feats"][:, st:st + frames, :].astype(np.float32)))
        b, d = targets(s["beats"], s["downs"], st, frames)
        bb.append(torch.from_numpy(b))
        dd.append(torch.from_numpy(d))
        if obs_cache is not None:
            oo.append(torch.from_numpy(obs_cache[s["stem"]][st:st + frames]))
    o = torch.stack(oo).to(dev) if obs_cache is not None else None
    return (torch.stack(fe).to(dev), torch.stack(bb).to(dev), torch.stack(dd).to(dev), o)


# --------------------------------------------------------------------------- eval
@torch.no_grad()
def obs_contrast_song(model, obs_t, downs, ref, T, n_off=12):
    """Likelihood contrast of the TRAINED emission: geometric-mean per-frame ratio
    p(o|z at TRUE bar phase) / p(o|z at a wrong bar phase), averaged over 11 wrong offsets."""
    if len(downs) < 3:
        return float("nan")
    phi = ideal_barphase(downs, T, FPS, mode="extrap")
    if phi is None:
        return float("nan")
    m = _estimate_meter(ref, downs)
    bar_frames = float(np.median(np.diff(downs))) * FPS
    lt = math.log(TWO_PI / max(bar_frames, 1e-6))
    dv = obs_t.device
    mt = F.one_hot(torch.tensor([m - 1] * T, device=dv), model.K).float()
    ltv = torch.full((T,), lt, device=dv)
    ph = torch.from_numpy(phi).float().to(dv)
    ll_true = float(model.obs_logp(model.z_features(mt, ph, ltv), obs_t).mean())
    offs = []
    for k in range(1, n_off):
        pk = (ph + TWO_PI * k / n_off) % TWO_PI
        offs.append(float(model.obs_logp(model.z_features(mt, pk, ltv), obs_t).mean()))
    return float(math.exp(min(ll_true - float(np.mean(offs)), 60.0)))


@torch.no_grad()
def eval_pf(merge, model, songs, obs_cache, proj, K, alpha, dev, smooth=5, seed=1234,
            max_frames=None, want_contrast=True):
    rows = []
    for i, s in enumerate(songs):
        T = s["feats"].shape[1] if max_frames is None else min(s["feats"].shape[1], max_frames)
        ref = s["beats"][s["beats"] < T / FPS]
        dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 3:
            continue
        f = torch.from_numpy(s["feats"][:, :T, :].astype(np.float32)).unsqueeze(0).to(dev)
        h = merge(f)
        obs = (proj(f) if proj is not None
               else torch.from_numpy(obs_cache[s["stem"]][:T]).unsqueeze(0).to(dev))
        torch.manual_seed(seed + i)
        out = VB.particle_filter(model, h, obs, K=K, alpha=alpha)
        m = _estimate_meter(ref, dref)
        row = dict(stem=s["stem"], dataset=s["dataset"], T=T, meter=int(m),
                   n_true=int(len(ref)), n_true_db=int(len(dref)), ess=float(out["ess"]),
                   metronome_F=f_measure(ref, metronome(T, FPS)))
        if want_contrast:
            row["obs_contrast"] = obs_contrast_song(model, obs[0], dref, ref, T)
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
        del f, h, obs
    return rows


def summarize(rows, tag):
    ne = sum(r[f"{tag}|n_est"] for r in rows)
    nt = sum(r["n_true"] for r in rows)
    ned = sum(r[f"{tag}|n_est_db"] for r in rows)
    ntd = sum(r["n_true_db"] for r in rows)
    bf, bb = M(rows, f"{tag}|beat_F"), M(rows, f"{tag}|blind_best")
    dfm, dbb = M(rows, f"{tag}|db_F"), M(rows, f"{tag}|blind_db_best")
    return dict(
        readout=tag, beat_F=bf, downbeat_F=dfm,
        n_ratio=ne / max(nt, 1), n_ratio_db=ned / max(ntd, 1),
        blind_same_density=M(rows, f"{tag}|blind0"), blind_best_offset=bb,
        margin_over_blind=bf - bb,
        blind_db_best=dbb, margin_db_over_blind=dfm - dbb,
        frac_neg=M(rows, f"{tag}|frac_neg"), mean_adv=M(rows, f"{tag}|mean_adv"),
        jitter=M(rows, f"{tag}|jitter"), jitter_over_adv=M(rows, f"{tag}|jit_adv"),
        ess=M(rows, "ess"), obs_contrast=M(rows, "obs_contrast"),
        metronome=M(rows, "metronome_F"), n_songs=len(rows))


def pr(d):
    print(f"    [{d['readout']:6s}] beat_F={d['beat_F']:.4f} db_F={d['downbeat_F']:.4f} "
          f"n_ratio={d['n_ratio']:.3f} blind0={d['blind_same_density']:.4f} "
          f"blindbest={d['blind_best_offset']:.4f} MARGIN={d['margin_over_blind']:+.4f} | "
          f"db_blind={d['blind_db_best']:.4f} MARGIN_db={d['margin_db_over_blind']:+.4f} | "
          f"frac_neg={d['frac_neg']:.3f} jit/adv={d['jitter_over_adv']:.2f} "
          f"ESS={d['ess']:.1f} obs_contrast={d['obs_contrast']:.3g}", flush=True)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, choices=["head_bern", "head_gauss", "pca_gauss"])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--obs_w", type=float, default=1.0)
    ap.add_argument("--Ks", type=int, nargs="+", default=[300, 600])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--n_eval", type=int, default=0)          # 0 = all 79
    ap.add_argument("--max_frames", type=int, default=0)      # 0 = full length
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or a.obs
    mf = a.max_frames or None

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    banner(f"ARM (i)  obs={a.obs}  tag={tag}")
    t0 = time.time()
    train = load_split("train", with_feats=True)
    ev = load_split("eval", with_feats=True, cap=(a.n_eval or None))
    print(f"  train {len(train)}  eval {len(ev)}   loaded in {time.time()-t0:.0f}s", flush=True)
    assert all(s["fold"] != 0 for s in train) and all(s["fold"] == 0 for s in ev)

    if a.obs == "pca_gauss":
        obs_dim, obs_type = 32, "gauss"
        torch.manual_seed(0)
        mu, comps = fit_pca(train, np.random.default_rng(0), obs_dim=obs_dim)
        proj = FixedProj(mu, comps).to(DEV)
        obs_tr = obs_ev = None
    else:
        obs_dim = 2
        obs_type = "bern" if a.obs == "head_bern" else "gauss"
        proj = None
        obs_tr = build_obs_cache(train, f"{ARMS}/act_train.npz", a.obs)
        obs_ev = build_obs_cache(ev, f"{ARMS}/act_eval.npz", a.obs)
        print(f"  observation = frozen head activation, obs_dim={obs_dim}, "
              f"likelihood={obs_type}", flush=True)

    torch.manual_seed(0)
    merge = LayerMerge().to(DEV)
    model = VB.BarPointerVAE_B(h_dim=768, hidden=a.hidden, num_meters=4,
                               obs_dim=obs_dim, obs_type=obs_type).to(DEV)
    params = list(merge.parameters()) + list(model.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr)

    banner("(1) TRAIN  (train fold only)")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        beta = min(1.0, step / a.warmup)
        temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        f, b, d, o = sample_batch(train, obs_tr, rng, a.bs, a.frames, DEV)
        if proj is not None:
            o = proj(f)
        opt.zero_grad()
        loss, info = VB.elbo_b(model, merge(f), b, d, o, temperature=temp, beta=beta,
                               obs_w=a.obs_w)
        if not torch.isfinite(loss):
            print("NaN @", step, flush=True)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"  s{step:5d} loss={info['loss']:.1f} rec_b={info['recon_beat']:.1f} "
                  f"rec_db={info['recon_db']:.1f} rec_obs={info['recon_obs']:.1f} "
                  f"kl={info['kl']:.1f} kl_phi={info['kl_phase']:.1f} "
                  f"{step/(time.time()-t0):.2f} it/s", flush=True)
    merge.eval(); model.eval()
    lw = merge.weights()
    print("  layer softmax: " + " ".join(f"{v:.3f}" for v in lw) + f"  argmax={int(lw.argmax())}")
    torch.save({"merge": merge.state_dict(), "model": model.state_dict(),
                "obs": a.obs, "config": vars(a)}, f"{ARMS}/arm_i_{tag}.pt")

    banner("(2) DEPLOY = BOOTSTRAP PARTICLE FILTER  (eval fold 0, full length)")
    res = {"config": vars(a), "layer_w": lw.tolist(), "pf": {}}
    for K in a.Ks:
        for alpha in a.alphas:
            t1 = time.time()
            rows = eval_pf(merge, model, ev, obs_ev, proj, K, alpha, DEV, max_frames=mf)
            print(f"  PF K={K} alpha={alpha}  ({time.time()-t1:.0f}s, {len(rows)} songs)",
                  flush=True)
            for rd in ("mean", "map", "smooth"):
                s = summarize(rows, rd)
                pr(s)
                res["pf"][f"K{K}_a{alpha}_{rd}"] = s
            res.setdefault("rows", {})[f"K{K}_a{alpha}"] = rows
    json.dump(res, open(f"{ARMS}/arm_i_{tag}.json", "w"), indent=1, default=float)
    print(f"WROTE {ARMS}/arm_i_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
