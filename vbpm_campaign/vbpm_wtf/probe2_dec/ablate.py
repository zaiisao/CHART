"""PROBE 2 -- what does p(b|z) actually use? (ablation of z_feat at decode time)"""
from __future__ import annotations
import argparse, json, math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms")

import variant_b as VB                                                # noqa: E402
from vbpm.distributions import (TWO_PI, gumbel_softmax, sample_wrapped_cauchy,
                                sample_student_t)                     # noqa: E402
from audit_common import load_split, FPS                              # noqa: E402
from common import targets                                            # noqa: E402

DEV = "cuda:0"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
OUT = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf/probe2_dec"


class LayerMerge(nn.Module):
    def __init__(self, n_layers=13):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, feats):
        return torch.einsum("l,bltf->btf", torch.softmax(self.layer_logits, 0), feats)


@torch.no_grad()
def posterior_Z(model, h, b, temperature=0.3):
    """Exactly elbo_b's q-sampling recursion; returns Z, phi, log_tempo, meter, rho."""
    B, T, _ = h.shape
    post_ctx = model.encode_posterior(h, b)
    dof = model.tempo_dof()
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
        model.post_head(torch.cat([post_ctx[:, 0], z0], dim=-1)))
    meter = gumbel_softmax(q_m, temperature)
    phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
    level = sample_student_t(dof, q_lv_mu, q_lv_s)
    dev = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
    log_tempo = level + dev
    Zs, PH, LT, MT = [model.z_features(meter, phi, log_tempo)], [phi], [log_tempo], [meter]
    RHO = [q_ph_rho]
    meter_prev, phi_prev, lt_prev = meter, phi, log_tempo
    for t in range(1, T):
        zpf = model.z_features(meter_prev, phi_prev, lt_prev)
        q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
            model.post_head(torch.cat([post_ctx[:, t], zpf], dim=-1)))
        advance = phi_prev + torch.exp(lt_prev.clamp(-12.0, 6.0))
        cross = (advance >= TWO_PI).to(h.dtype)
        phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
        level = sample_student_t(dof, q_lv_mu, q_lv_s)
        dev = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
        log_tempo = level + dev
        q_draw = gumbel_softmax(q_m, temperature)
        meter = torch.where(cross.unsqueeze(-1) > 0.5, q_draw, meter_prev)
        Zs.append(model.z_features(meter, phi, log_tempo))
        PH.append(phi); LT.append(log_tempo); MT.append(meter); RHO.append(q_ph_rho)
        meter_prev, phi_prev, lt_prev = meter, phi, log_tempo
    return (torch.stack(Zs, 1), torch.stack(PH, 1), torch.stack(LT, 1),
            torch.stack(MT, 1), torch.stack(RHO, 1))


def ablate(Z, kind, g):
    """Z [B,T,7]: cols 0,1 = cos/sin phi ; 2 = log_tempo ; 3.. = meter one-hot."""
    B, T, D = Z.shape
    Z = Z.clone(); dv = Z.device; K = D - 3

    def rand_phase():
        u = torch.rand(B, T, device=dv, generator=g) * TWO_PI
        return torch.cos(u), torch.sin(u)

    def rand_meter():
        idx = torch.randint(0, K, (B,), device=dv, generator=g)
        return F.one_hot(idx, K).float().unsqueeze(1).expand(-1, T, -1)

    if kind == "none":
        return Z
    if kind == "phase_rand":
        c, s = rand_phase(); Z[..., 0], Z[..., 1] = c, s
    elif kind == "phase_shuf_t":
        idx = torch.argsort(torch.rand(B, T, device=dv, generator=g), dim=1)
        Z[..., 0] = torch.gather(Z[..., 0], 1, idx)
        Z[..., 1] = torch.gather(Z[..., 1], 1, idx)
    elif kind == "phase_roll":
        for i in range(B):
            k = int(torch.randint(1, T, (1,), device=dv, generator=g).item())
            Z[i, :, 0] = torch.roll(Z[i, :, 0], k)
            Z[i, :, 1] = torch.roll(Z[i, :, 1], k)
    elif kind == "phase_const":
        c = Z[..., 0].mean(1, keepdim=True); s = Z[..., 1].mean(1, keepdim=True)
        n = torch.sqrt(c ** 2 + s ** 2).clamp(min=1e-6)
        Z[..., 0] = (c / n).expand(-1, T); Z[..., 1] = (s / n).expand(-1, T)
    elif kind == "tempo_shuf_t":
        idx = torch.argsort(torch.rand(B, T, device=dv, generator=g), dim=1)
        Z[..., 2] = torch.gather(Z[..., 2], 1, idx)
    elif kind == "tempo_shuf_b":
        Z[..., 2] = Z[torch.randperm(B, device=dv, generator=g), :, 2]
    elif kind == "tempo_const":
        Z[..., 2] = Z[..., 2].mean(1, keepdim=True).expand(-1, T)
    elif kind == "tempo_global":
        Z[..., 2] = -2.66
    elif kind == "meter_rand":
        Z[..., 3:] = rand_meter()
    elif kind == "meter_unif":
        Z[..., 3:] = 1.0 / K
    elif kind == "allmean":
        Z = Z.mean(1, keepdim=True).expand(-1, T, -1).contiguous()
    elif kind == "keep_tempo_only":
        c, s = rand_phase(); Z[..., 0], Z[..., 1] = c, s
        Z[..., 3:] = rand_meter()
    elif kind == "keep_phase_only":
        Z[..., 2] = Z[..., 2].mean(1, keepdim=True).expand(-1, T)
        Z[..., 3:] = rand_meter()
    elif kind == "keep_nothing":
        c, s = rand_phase(); Z[..., 0], Z[..., 1] = c, s
        Z[..., 2] = Z[..., 2].mean(1, keepdim=True).expand(-1, T)
        Z[..., 3:] = rand_meter()
    else:
        raise ValueError(kind)
    return Z


KINDS = ["none", "phase_rand", "phase_shuf_t", "phase_roll", "phase_const",
         "tempo_shuf_t", "tempo_shuf_b", "tempo_const", "tempo_global",
         "meter_rand", "meter_unif", "allmean",
         "keep_tempo_only", "keep_phase_only", "keep_nothing"]


def bce_sum(logits, tgt):
    return F.binary_cross_entropy_with_logits(logits, tgt, reduction="none").sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["i", "ii"])
    ap.add_argument("--crops_per_song", type=int, default=6)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--split", default="eval")
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    tag = f"{a.arm}_bern"
    ck = torch.load(f"{ARMS}/arm_i_{tag}.pt", map_location="cpu")
    h_dim = 768 if a.arm == "i" else 2
    model = VB.BarPointerVAE_B(h_dim=h_dim, hidden=128, num_meters=4,
                               obs_dim=2, obs_type="bern").to(DEV)
    model.load_state_dict(ck["model"]); model.eval()
    merge = LayerMerge().to(DEV); merge.load_state_dict(ck["merge"]); merge.eval()

    ev = load_split(a.split, with_feats=False)
    act = np.load(f"{ARMS}/act_{'eval' if a.split == 'eval' else 'train'}.npz",
                  allow_pickle=True)
    rng = np.random.default_rng(7)
    g = torch.Generator(device=DEV); g.manual_seed(7)

    T = a.frames
    acc = {k: {"b": [], "db": [], "obs": []} for k in KINDS}
    nbeats, ndb = [], []
    lt_all, b_all, ph_all = [], [], []
    n_crops = 0
    for s in ev:
        d = np.load(s["path"], allow_pickle=True)
        feats = d["feats"] if a.arm == "i" else None
        A = np.clip(np.asarray(act[s["stem"] + "|act"], np.float32), 1e-4, 1 - 1e-4)
        Tt = s["T"]
        if Tt <= T + 1:
            continue
        starts = rng.integers(0, Tt - T, size=a.crops_per_song)
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
        n_crops += b.shape[0]
        nbeats.append(b.sum(1).cpu().numpy()); ndb.append(db.sum(1).cpu().numpy())
        for _ in range(a.reps):
            Z, PH, LT, MT, RHO = posterior_Z(model, h, b, a.temp)
            lt_all.append(LT.cpu().numpy()); ph_all.append(PH.cpu().numpy())
            b_all.append(b.cpu().numpy())
            for k in KINDS:
                Za = ablate(Z, k, g)
                lg = model.decoder(Za)
                acc[k]["b"].append(bce_sum(lg[..., 0], b).cpu().numpy())
                acc[k]["db"].append(bce_sum(lg[..., 1], db).cpu().numpy())
                ol = model.obs_logp(Za.reshape(-1, Za.shape[-1]),
                                    obs.reshape(-1, obs.shape[-1])).reshape(b.shape[0], T)
                acc[k]["obs"].append((-ol.sum(1)).cpu().numpy())
        del feats, d

    nb = np.concatenate(nbeats); nd = np.concatenate(ndb)
    p_glob_b = nb.sum() / (n_crops * T); p_glob_d = nd.sum() / (n_crops * T)

    def H(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    def const_bce(n, p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(n * np.log(p) + (T - n) * np.log(1 - p))

    out = {"arm": a.arm, "n_crops": n_crops, "reps": a.reps, "frames": T,
           "beat_density": float(p_glob_b), "db_density": float(p_glob_d),
           "base_rate_BCE_beat_global": float(const_bce(nb, p_glob_b).mean()),
           "base_rate_BCE_db_global": float(const_bce(nd, p_glob_d).mean()),
           "percrop_optconst_BCE_beat": float((T * H(nb / T)).mean()),
           "percrop_optconst_BCE_db": float((T * H(nd / T)).mean()),
           "mean_beats_per_crop": float(nb.mean()), "mean_db_per_crop": float(nd.mean())}
    for k in KINDS:
        out[f"rec_beat|{k}"] = float(np.concatenate(acc[k]["b"]).mean())
        out[f"rec_db|{k}"] = float(np.concatenate(acc[k]["db"]).mean())
        out[f"rec_obs|{k}"] = float(np.concatenate(acc[k]["obs"]).mean())

    LTf = np.concatenate(lt_all).ravel(); Bf = np.concatenate(b_all).ravel()
    PHf = np.concatenate(ph_all).ravel()
    r = np.argsort(np.argsort(LTf)) + 1.0
    n1 = Bf.sum(); n0 = len(Bf) - n1
    auc_lt = float((r[Bf > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    out.update(dict(auc_logtempo_beat=auc_lt,
                    logtempo_mean=float(LTf.mean()), logtempo_std=float(LTf.std()),
                    logtempo_mean_at_beats=float(LTf[Bf > 0.5].mean()),
                    logtempo_mean_off_beats=float(LTf[Bf < 0.5].mean()),
                    phase_lock_R_at_beats=float(np.abs(np.mean(np.exp(1j * PHf[Bf > 0.5])))),
                    phase_lock_R_all=float(np.abs(np.mean(np.exp(1j * PHf))))))
    out["decoder_W0_colnorm"] = [round(float(x), 4)
                                 for x in model.decoder[0].weight.detach().norm(dim=0).cpu()]
    out["h_dec_W0_colnorm"] = [round(float(x), 4)
                               for x in model.h_dec[0].weight.detach().norm(dim=0).cpu()]
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{OUT}/ablate_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
