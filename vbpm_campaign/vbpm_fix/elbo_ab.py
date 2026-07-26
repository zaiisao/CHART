"""ELBO + deploy paths for VARIANT A+B.  Copy of vbpm/elbo.py logic, modified.

The single most important invariant: the TRAINING roll-out and every DEPLOY path call the
same function `transition_params()` for the prior dynamics.  There is exactly one place
where mu^p_phi / mu^p_level are defined, so training and deploy semantics cannot drift.

Deploy paths provided
---------------------
  free_run_ab(...)        (A only, or baseline-equivalent with use_corr=False):
                          the noise-free prior MEAN chain, same read-out as vbpm/elbo.py.
  particle_filter(...)    (B):  bootstrap PF, K particles, proposal = the prior transition,
                          weights = p(o_t | z_t), systematic resampling at ESS < K/2,
                          circular weighted-mean phase read-out (+ MAP ancestral path).
                          `use_corr` toggles (A) inside the PF -> gives the 2x2 ablation.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from vbpm.distributions import (
    TWO_PI, gumbel_softmax, sample_wrapped_cauchy, sample_student_t,
    kl_categorical, kl_wrapped_cauchy, kl_log_normal, kl_student_t_mc,
)
from vbpm.elbo import _stationary_dev_sigma


# =====================================================================================
# THE ONE definition of the prior transition (shared by training and every deploy path)
# =====================================================================================
def transition_params(model, ctx_t, phi_prev, log_tempo_prev, level_prev, dev_prev,
                      level_anchor, a_lv, use_corr: bool = True):
    """Returns (p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, p_dv_mu, p_dv_s, a_dev, cross)."""
    tempo_prev = torch.exp(log_tempo_prev.clamp(-12.0, 6.0))
    advance = phi_prev + tempo_prev                                  # baseline bar-pointer
    if use_corr:                                                     # ---- (A) ----
        advance = advance + model.phase_correction(ctx_t, advance, log_tempo_prev)
    cross = (advance >= TWO_PI).to(phi_prev.dtype)
    p_ph_mu = advance % TWO_PI
    p_ph_rho = model.prior_phase_conc(ctx_t)
    a_dev = model.prior_dev_coef(ctx_t)
    p_lv_mu = level_anchor + a_lv * (level_prev - level_anchor)
    if use_corr:                                                     # ---- (A) ----
        p_lv_mu = p_lv_mu + model.level_correction(ctx_t, advance, log_tempo_prev)
    p_lv_s = model.prior_level_scale(ctx_t)
    p_dv_mu = a_dev * dev_prev
    p_dv_s = model.prior_dev_scale(ctx_t)
    return p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, p_dv_mu, p_dv_s, a_dev, cross


# =====================================================================================
# Training: Algorithm 1 + the observation term
# =====================================================================================
def elbo_ab(model, h, b, db, temperature: float = 0.5, beta: float = 1.0,
            obs_weight: float = 1.0, use_corr: bool = True):
    B, T, _ = h.shape
    post_ctx = model.encode_posterior(h, b)
    prior_ctx = model.encode_prior(h)
    obs = model.observe(h)                       # [B,T,obs_dim]   (B) observation channel
    dof = model.tempo_dof()

    kl_m = h.new_zeros(B); kl_p = h.new_zeros(B)
    kl_lv = h.new_zeros(B); kl_dv = h.new_zeros(B)
    z_feats = []
    n_cross = h.new_zeros(B)

    # ---------- t = 1 ----------
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
        model.post_head(torch.cat([post_ctx[:, 0], z0], dim=-1)))
    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _a, _b = model.unpack(
        model.prior_init_head(prior_ctx.mean(1)))
    a0 = model.prior_dev_coef(prior_ctx[:, 0])
    sd0 = model.prior_dev_scale(prior_ctx[:, 0])
    p_dv_mu = torch.zeros_like(q_dv_mu)
    p_dv_s = _stationary_dev_sigma(sd0, a0)

    meter = gumbel_softmax(q_m, temperature)
    phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
    level = sample_student_t(dof, q_lv_mu, q_lv_s)
    dev = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
    log_tempo = level + dev

    kl_m = kl_m + kl_categorical(torch.log_softmax(q_m, -1), torch.log_softmax(p_m, -1))
    kl_p = kl_p + kl_wrapped_cauchy(q_ph_mu, q_ph_rho, p_ph_mu, p_ph_rho)
    kl_lv = kl_lv + kl_student_t_mc(dof, q_lv_mu, q_lv_s, p_lv_mu, p_lv_s, level)
    kl_dv = kl_dv + kl_log_normal(q_dv_mu, q_dv_s, p_dv_mu, p_dv_s)
    n_cross = n_cross + 1.0

    z_feats.append(model.z_features(meter, phi, log_tempo))
    level_anchor = level
    a_lv = model.level_ar()
    meter_prev, phi_prev = meter, phi
    level_prev, dev_prev, log_tempo_prev = level, dev, log_tempo

    # ---------- t = 2..T ----------
    for t in range(1, T):
        z_prev_feat = model.z_features(meter_prev, phi_prev, log_tempo_prev)
        q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
            model.post_head(torch.cat([post_ctx[:, t], z_prev_feat], dim=-1)))

        (p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, p_dv_mu, p_dv_s, a_dev,
         cross) = transition_params(model, prior_ctx[:, t], phi_prev, log_tempo_prev,
                                    level_prev, dev_prev, level_anchor, a_lv, use_corr)

        phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
        level = sample_student_t(dof, q_lv_mu, q_lv_s)
        dev = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
        log_tempo = level + dev

        q_meter_draw = gumbel_softmax(q_m, temperature)
        meter = torch.where(cross.unsqueeze(-1) > 0.5, q_meter_draw, meter_prev)
        log_pi_p = model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[:, t])

        kl_p = kl_p + kl_wrapped_cauchy(q_ph_mu, q_ph_rho, p_ph_mu, p_ph_rho)
        kl_lv = kl_lv + kl_student_t_mc(dof, q_lv_mu, q_lv_s, p_lv_mu, p_lv_s, level)
        kl_dv = kl_dv + kl_log_normal(q_dv_mu, q_dv_s, p_dv_mu, p_dv_s)
        kl_m = kl_m + cross * kl_categorical(torch.log_softmax(q_m, -1), log_pi_p)
        n_cross = n_cross + cross

        z_feats.append(model.z_features(meter, phi, log_tempo))
        meter_prev, phi_prev = meter, phi
        level_prev, dev_prev, log_tempo_prev = level, dev, log_tempo

    Z = torch.stack(z_feats, dim=1)                                   # [B,T,zf]
    logits = model.decode(Z.reshape(B * T, -1), None).reshape(B, T, 2)
    beat_logits, db_logits = logits[..., 0], logits[..., 1]
    recon_b = F.binary_cross_entropy_with_logits(beat_logits, b, reduction="none").sum(1)
    recon_db = F.binary_cross_entropy_with_logits(db_logits, db, reduction="none").sum(1)

    # ---------- (B) observation reconstruction: -log p(o_t | z_t) ----------
    obs_ll = model.obs_logprob(Z.reshape(B * T, -1), obs.reshape(B * T, -1)).reshape(B, T)
    recon_o = (-obs_ll).sum(1)                                        # [B]

    L_kl = kl_m + kl_p + kl_lv + kl_dv
    loss = (recon_b + recon_db + obs_weight * recon_o + beta * L_kl).mean()

    info = {"loss": float(loss), "recon_beat": float(recon_b.mean()),
            "recon_db": float(recon_db.mean()), "recon_obs": float(recon_o.mean()),
            "kl": float(L_kl.mean()), "kl_meter": float(kl_m.mean()),
            "kl_phase": float(kl_p.mean()), "kl_level": float(kl_lv.mean()),
            "kl_dev": float(kl_dv.mean()), "n_cross": float(n_cross.mean()),
            "tempo_dof": float(dof.detach()), "beta": float(beta)}
    return loss, info


# =====================================================================================
# Deploy path 1 : prior MEAN chain  ("free run").  use_corr=True  ->  variant (A).
# =====================================================================================
@torch.no_grad()
def free_run_ab(model, h, temperature: float = 0.3, use_corr: bool = True):
    B, T, _ = h.shape
    prior_ctx = model.encode_prior(h)
    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _x, _y = model.unpack(
        model.prior_init_head(prior_ctx.mean(1)))

    phi_mu = p_ph_mu % TWO_PI
    level_mu = p_lv_mu
    dev_mu = torch.zeros_like(p_lv_mu)
    level_anchor = level_mu
    a_lv = model.level_ar()
    log_tempo_mu = level_mu + dev_mu
    phase_traj, tempo_traj = [phi_mu], [log_tempo_mu]

    for t in range(1, T):
        ctx = prior_ctx[:, t]
        (p_ph_mu, _r, p_lv_mu, _s, _dm, _ds, a_dev,
         _c) = transition_params(model, ctx, phi_mu, log_tempo_mu, level_mu, dev_mu,
                                 level_anchor, a_lv, use_corr)
        phi_mu = p_ph_mu
        level_mu = p_lv_mu
        dev_mu = a_dev * dev_mu
        log_tempo_mu = level_mu + dev_mu
        phase_traj.append(phi_mu); tempo_traj.append(log_tempo_mu)

    return {"phase_mu": torch.stack(phase_traj, dim=1),
            "log_tempo": torch.stack(tempo_traj, dim=1)}


# =====================================================================================
# Deploy path 2 : bootstrap PARTICLE FILTER.  Weights = p(o_t | z_t).   variant (B).
# =====================================================================================
def _systematic_resample(w):
    K = w.shape[0]
    pos = (torch.rand(1, device=w.device) + torch.arange(K, device=w.device, dtype=w.dtype)) / K
    cdf = torch.cumsum(w, 0)
    cdf = cdf / cdf[-1]
    return torch.searchsorted(cdf, pos).clamp(max=K - 1)


def _circ_wmean(phi, w):
    return torch.atan2((w * torch.sin(phi)).sum(), (w * torch.cos(phi)).sum()) % TWO_PI


@torch.no_grad()
def particle_filter(model, h, K: int = 500, use_corr: bool = True, temper: float = 1.0,
                    ess_frac: float = 0.5, keep_path: bool = True):
    """h [1,T,h_dim].  Returns phase read-outs [1,T]."""
    assert h.shape[0] == 1
    dev = h.device
    T = h.shape[1]
    prior_ctx = model.encode_prior(h)[0]                 # [T,hidden]
    o = model.observe(h)[0]                              # [T,obs_dim]
    dof = model.tempo_dof()
    a_lv = model.level_ar()

    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _x, _y = model.unpack(
        model.prior_init_head(prior_ctx.mean(0, keepdim=True)))
    exp = lambda x: x.expand(K).contiguous()
    a0 = model.prior_dev_coef(prior_ctx[0:1]); sd0 = model.prior_dev_scale(prior_ctx[0:1])
    p_dv_s0 = _stationary_dev_sigma(sd0, a0)

    phi = sample_wrapped_cauchy(exp(p_ph_mu), exp(p_ph_rho))
    level = sample_student_t(dof, exp(p_lv_mu), exp(p_lv_s))
    dev_s = exp(p_dv_s0) * torch.randn(K, device=dev)
    log_tempo = level + dev_s
    m_idx = torch.multinomial(torch.softmax(p_m, -1).repeat(K, 1), 1).squeeze(-1)
    meter = F.one_hot(m_idx, model.K).to(h.dtype)
    level_anchor = level.clone()

    zf = model.z_features(meter, phi, log_tempo)
    logw = temper * model.obs_logprob(zf, o[0:1])
    logw = logw - torch.logsumexp(logw, 0)

    phi_hist = torch.zeros(T, K, device=dev, dtype=h.dtype)
    phi_hist[0] = phi
    w = torch.softmax(logw, 0)
    out_mean = [_circ_wmean(phi, w)]
    ess_log = [float(1.0 / (w ** 2).sum())]

    meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo
    level_prev, dev_prev = level, dev_s

    for t in range(1, T):
        ctx = prior_ctx[t:t + 1].expand(K, -1)
        (p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, p_dv_mu, p_dv_s, a_dev,
         cross) = transition_params(model, ctx, phi_prev, log_tempo_prev, level_prev,
                                    dev_prev, level_anchor, a_lv, use_corr)
        # bootstrap proposal == the prior transition (identical semantics to training)
        phi = sample_wrapped_cauchy(p_ph_mu, p_ph_rho)
        level = sample_student_t(dof, p_lv_mu, p_lv_s)
        dev_s = p_dv_mu + p_dv_s * torch.randn(K, device=dev)
        log_tempo = level + dev_s
        logpi = model.meter_prior_logp(meter_prev, phi, phi_prev, ctx)
        draw = F.one_hot(torch.multinomial(torch.softmax(logpi, -1), 1).squeeze(-1),
                         model.K).to(h.dtype)
        meter = torch.where(cross.unsqueeze(-1) > 0.5, draw, meter_prev)

        zf = model.z_features(meter, phi, log_tempo)
        logw = logw + temper * model.obs_logprob(zf, o[t:t + 1])
        logw = logw - torch.logsumexp(logw, 0)
        w = torch.exp(logw)

        phi_hist[t] = phi
        out_mean.append(_circ_wmean(phi, w))
        ess = 1.0 / (w ** 2).sum()
        ess_log.append(float(ess))
        if ess < ess_frac * K:
            idx = _systematic_resample(w)
            phi = phi[idx]; level = level[idx]; dev_s = dev_s[idx]
            log_tempo = log_tempo[idx]; meter = meter[idx]
            level_anchor = level_anchor[idx]
            if keep_path:
                phi_hist[:t + 1] = phi_hist[:t + 1][:, idx]
            logw = torch.full((K,), -math.log(K), device=dev, dtype=h.dtype)

        meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo
        level_prev, dev_prev = level, dev_s

    best = int(torch.argmax(logw))
    return {"phase_mean": torch.stack(out_mean).unsqueeze(0),      # [1,T] circular w-mean
            "phase_path": phi_hist[:, best].unsqueeze(0),          # [1,T] MAP ancestral path
            "ess": ess_log}
