"""PROBE 5 -- train/deploy z_feat distribution mismatch.  Read-only w.r.t. vbpm*/ ."""
from __future__ import annotations
import glob, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms")

import variant_b as VB                                                  # noqa: E402
from vbpm.distributions import (TWO_PI, gumbel_softmax, sample_wrapped_cauchy,   # noqa: E402
                                sample_student_t)

CACHE = "/disk1/jaehoon/vbpm_mert_cache"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
FPS = 50.0


def load_some(split, n, with_feats=True):
    files = sorted(glob.glob(f"{CACHE}/{split}__*.npz"))
    if n and n < len(files):
        idx = np.linspace(0, len(files) - 1, n).round().astype(int)
        files = [files[i] for i in idx]
    out = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        r = dict(stem=Path(f).stem, T=int(d["feats"].shape[1]),
                 beats=np.asarray(d["beats"], float), downs=np.asarray(d["downs"], float),
                 dataset=str(d["dataset"]), fold=int(d["fold"]))
        if with_feats:
            r["feats"] = np.asarray(d["feats"])
        out.append(r)
    return out


def obs_cache(songs, npz, mode="head_bern"):
    d = np.load(npz, allow_pickle=True)
    out = {}
    for s in songs:
        a = np.asarray(d[s["stem"] + "|act"], np.float32)
        a = np.clip(a, 1e-4, 1.0 - 1e-4)
        out[s["stem"]] = a if mode == "head_bern" else np.log(a / (1.0 - a))
    return out


def load_arm(tag, dev):
    """tag in {i_bern, ii_bern, i_gauss}."""
    ck = torch.load(f"{ARMS}/arm_i_{tag}.pt", map_location=dev)
    cfg = ck["config"]
    h_dim = 2 if tag.startswith("ii") else 768
    obs_type = "bern" if ck["obs"] == "head_bern" else "gauss"
    model = VB.BarPointerVAE_B(h_dim=h_dim, hidden=cfg["hidden"], num_meters=4,
                               obs_dim=2, obs_type=obs_type).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    lw = torch.softmax(ck["merge"]["layer_logits"], 0).to(dev)
    return model, lw, cfg, ck


def merged_h(feats_np, lw, dev):
    f = torch.from_numpy(np.asarray(feats_np, np.float32)).to(dev)
    return torch.einsum("l,ltf->tf", lw, f).unsqueeze(0)


# --------------------------------------------------------------------------- TRAIN regime
@torch.no_grad()
def train_trace(model, h, b, temperature=0.3):
    """EXACT copy of variant_b.elbo_b's posterior recursion, recording z per frame."""
    B, T, _ = h.shape
    post_ctx = model.encode_posterior(h, b)
    prior_ctx = model.encode_prior(h)
    dof = model.tempo_dof()

    z0 = model.z0.unsqueeze(0).expand(B, -1)
    q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
        model.post_head(torch.cat([post_ctx[:, 0], z0], dim=-1)))

    meter = gumbel_softmax(q_m, temperature)
    phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
    level = sample_student_t(dof, q_lv_mu, q_lv_s)
    dev_ = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
    log_tempo = level + dev_

    zeros = torch.zeros_like(q_ph_rho)
    rec = dict(phi=[phi], level=[level], dev=[dev_], lt=[log_tempo], meter=[meter],
               q_rho=[q_ph_rho], p_rho=[zeros], q_mu=[q_ph_mu], p_mu=[zeros],
               q_lv_s=[q_lv_s], q_dv_s=[q_dv_s], p_lv_s=[zeros], p_dv_s=[zeros],
               cross=[torch.ones_like(q_ph_rho)])
    zf = [model.z_features(meter, phi, log_tempo)]

    level_anchor = level
    a_lv = model.level_ar()
    meter_prev, phi_prev = meter, phi
    level_prev, dev_prev, lt_prev = level, dev_, log_tempo

    for t in range(1, T):
        z_prev_feat = model.z_features(meter_prev, phi_prev, lt_prev)
        q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
            model.post_head(torch.cat([post_ctx[:, t], z_prev_feat], dim=-1)))
        tempo_prev = torch.exp(lt_prev.clamp(-12.0, 6.0))
        advance = phi_prev + tempo_prev
        cross = (advance >= TWO_PI).to(h.dtype)
        p_ph_mu = advance % TWO_PI
        p_ph_rho = model.prior_phase_conc(prior_ctx[:, t])
        a = model.prior_dev_coef(prior_ctx[:, t])
        p_lv_s = model.prior_level_scale(prior_ctx[:, t])
        p_dv_s = model.prior_dev_scale(prior_ctx[:, t])

        phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
        level = sample_student_t(dof, q_lv_mu, q_lv_s)
        dev_ = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
        log_tempo = level + dev_
        q_meter_draw = gumbel_softmax(q_m, temperature)
        meter = torch.where(cross.unsqueeze(-1) > 0.5, q_meter_draw, meter_prev)

        rec["phi"].append(phi); rec["level"].append(level); rec["dev"].append(dev_)
        rec["lt"].append(log_tempo); rec["meter"].append(meter)
        rec["q_rho"].append(q_ph_rho); rec["p_rho"].append(p_ph_rho)
        rec["q_mu"].append(q_ph_mu); rec["p_mu"].append(p_ph_mu)
        rec["q_lv_s"].append(q_lv_s); rec["q_dv_s"].append(q_dv_s)
        rec["p_lv_s"].append(p_lv_s); rec["p_dv_s"].append(p_dv_s)
        rec["cross"].append(cross)
        zf.append(model.z_features(meter, phi, log_tempo))

        meter_prev, phi_prev = meter, phi
        level_prev, dev_prev, lt_prev = level, dev_, log_tempo

    out = {k: torch.stack(v, 1) for k, v in rec.items()}
    out["z"] = torch.stack(zf, 1)
    return out


# --------------------------------------------------------------------------- DEPLOY regime
@torch.no_grad()
def pf_trace(model, h, obs, K=300, alpha=1.0, seed=0, lt_lo=-3.55, lt_hi=-2.18,
             ess_frac=0.5, record_every=1):
    """variant_b.particle_filter, instrumented with the full particle cloud."""
    torch.manual_seed(seed)
    T = h.shape[1]
    dv = h.device
    prior_ctx = model.encode_prior(h)
    ctx = prior_ctx[0]
    dof = model.tempo_dof()
    a_lv = model.level_ar()
    Km = model.K
    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _a, _b = model.unpack(
        model.prior_init_head(prior_ctx.mean(1)))
    phi = torch.rand(K, device=dv) * TWO_PI
    level = lt_lo + (lt_hi - lt_lo) * torch.rand(K, device=dv)
    dev_ = torch.zeros(K, device=dv)
    log_tempo = level + dev_
    m_idx = torch.multinomial(F.softmax(p_m, -1).expand(K, Km), 1).squeeze(1)
    meter = F.one_hot(m_idx, Km).to(h.dtype)
    anchor = level.clone()

    zf = model.z_features(meter, phi, log_tempo)
    logw = alpha * model.obs_logp(zf, obs[0, 0].unsqueeze(0).expand(K, -1))
    w = F.softmax(logw, 0)

    P_phi, P_lt, P_m, P_w, P_rho, P_dphi = [], [], [], [], [], []

    def snap():
        P_phi.append(phi.cpu().numpy().copy()); P_lt.append(log_tempo.cpu().numpy().copy())
        P_m.append(meter.argmax(-1).cpu().numpy().copy()); P_w.append(w.cpu().numpy().copy())
    snap()

    for t in range(1, T):
        ctx_t = ctx[t].unsqueeze(0).expand(K, -1)
        advance = phi + torch.exp(log_tempo.clamp(-12.0, 6.0))
        cross = advance >= TWO_PI
        p_ph_mu_t = advance % TWO_PI
        rho = model.prior_phase_conc(ctx_t)
        a = model.prior_dev_coef(ctx_t)
        s_lv = model.prior_level_scale(ctx_t)
        s_dv = model.prior_dev_scale(ctx_t)
        P_rho.append(float(rho.mean()))
        phi_new = sample_wrapped_cauchy(p_ph_mu_t, rho)
        P_dphi.append(float(((phi_new - p_ph_mu_t + math.pi) % TWO_PI - math.pi).abs().mean()))
        level_new = sample_student_t(dof, anchor + a_lv * (level - anchor), s_lv)
        dev_new = a * dev_ + s_dv * torch.randn(K, device=dv)
        lt_new = level_new + dev_new
        log_pi = model.meter_prior_logp(meter, phi_new, phi, ctx_t)
        draw = torch.multinomial(log_pi.exp().clamp(min=1e-12), 1).squeeze(1)
        meter_new = torch.where(cross.unsqueeze(-1), F.one_hot(draw, Km).to(h.dtype), meter)
        phi, level, dev_, log_tempo, meter = phi_new, level_new, dev_new, lt_new, meter_new

        zf = model.z_features(meter, phi, log_tempo)
        logw = logw + alpha * model.obs_logp(zf, obs[0, t].unsqueeze(0).expand(K, -1))
        w = F.softmax(logw, 0)
        ess = float(1.0 / (w ** 2).sum())
        if t % record_every == 0:
            snap()
        if ess < ess_frac * K:
            idx = _sysres(w)
            phi, level, dev_, log_tempo = phi[idx], level[idx], dev_[idx], log_tempo[idx]
            meter, anchor = meter[idx], anchor[idx]
            logw = torch.zeros(K, device=dv)
            w = torch.full((K,), 1.0 / K, device=dv)
    return dict(phi=np.array(P_phi), lt=np.array(P_lt), m=np.array(P_m), w=np.array(P_w),
                rho=np.array(P_rho), dphi_noise=np.array(P_dphi))


def _sysres(w):
    K = w.shape[0]
    pos = (torch.arange(K, device=w.device, dtype=w.dtype) + torch.rand(1, device=w.device)) / K
    cdf = torch.cumsum(w, 0)
    cdf = cdf / cdf[-1].clamp(min=1e-30)
    return torch.searchsorted(cdf.contiguous(), pos.contiguous()).clamp(max=K - 1)


def qs(x, name=""):
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return name + " EMPTY"
    return (f"{name} n={x.size} mean={x.mean():+.4f} sd={x.std():.4f} "
            f"min={x.min():+.4f} p1={np.percentile(x,1):+.4f} p25={np.percentile(x,25):+.4f} "
            f"med={np.median(x):+.4f} p75={np.percentile(x,75):+.4f} "
            f"p99={np.percentile(x,99):+.4f} max={x.max():+.4f}")
