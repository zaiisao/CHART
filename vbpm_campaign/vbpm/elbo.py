"""Algorithm 1 (SGVB training rollout) and the deploy-path free-run for the
reality-adjusted VBPM. A single MC sample over the sampled z_{t-1}; beta = 1.

Objective (still the strict ELBO, no free-bits / no anneal / no supervision):

    L = sum_t [ BCE(beat_t) + BCE(downbeat_t) ]
        + sum_t [ KL_meter(gated) + KL_phase + KL_level + KL_dev ]

Per-latent breakdown:
  * phase   : wrapped Cauchy, closed-form KL
  * level   : slow log-tempo random walk, Student-t increments, 1-sample MC KL (reuses draw)
  * dev     : fast mean-reverting AR(1) deviation, Gaussian, closed-form KL
  * meter   : bar-gated Categorical; KL charged ONLY at phase-crossing frames, class copied between
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .distributions import (
    TWO_PI, gumbel_softmax, sample_wrapped_cauchy, sample_student_t,
    kl_categorical, kl_wrapped_cauchy, kl_log_normal, kl_student_t_mc,
)


def _stationary_dev_sigma(sigma, a):
    """Std of the AR(1) stationary law N(0, sigma^2/(1-a^2)) used for the t=1 deviation prior."""
    return sigma / torch.sqrt((1.0 - a ** 2).clamp(min=1e-3))


def strict_elbo(model, h: torch.Tensor, b: torch.Tensor, db: torch.Tensor,
                temperature: float = 0.5, beta: float = 1.0):
    """Algorithm 1 over a batch. ``h`` [B,T,n_mels], ``b`` [B,T] beats, ``db`` [B,T] downbeats.

    Returns (loss, info); ``loss`` is the per-sequence negative ELBO averaged over the batch.
    """
    B, T, _ = h.shape
    post_ctx = model.encode_posterior(h, b)    # reads (b, h)  [B,T,hidden]
    prior_ctx = model.encode_prior(h)          # reads h       [B,T,hidden]
    dof = model.tempo_dof()

    kl_m = h.new_zeros(B); kl_p = h.new_zeros(B)
    kl_lv = h.new_zeros(B); kl_dv = h.new_zeros(B)
    z_feats = []
    post_phase_mu = []
    n_cross = h.new_zeros(B)   # diagnostic: how many bar crossings (meter-KL events)

    # ---------- t = 1 : initial state ----------
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
        model.post_head(torch.cat([post_ctx[:, 0], z0], dim=-1)))
    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _p_dv_mu, _p_dv_s = model.unpack(
        model.prior_init_head(prior_ctx.mean(1)))

    # deviation init prior = AR(1) stationary N(0, sigma_d^2/(1-a^2)) from t=1 audio context
    a0 = model.prior_dev_coef(prior_ctx[:, 0])
    sd0 = model.prior_dev_scale(prior_ctx[:, 0])
    p_dv_mu = torch.zeros_like(q_dv_mu)
    p_dv_s = _stationary_dev_sigma(sd0, a0)

    meter = gumbel_softmax(q_m, temperature)                       # t=1 is a "crossing"
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
    post_phase_mu.append(q_ph_mu)
    level_anchor = level                                           # per-song tempo level (OU target)
    a_lv = model.level_ar()
    meter_prev, phi_prev = meter, phi
    level_prev, dev_prev, log_tempo_prev = level, dev, log_tempo

    # ---------- t = 2..T : transitions ----------
    for t in range(1, T):
        z_prev_feat = model.z_features(meter_prev, phi_prev, log_tempo_prev)
        q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
            model.post_head(torch.cat([post_ctx[:, t], z_prev_feat], dim=-1)))

        # prior MEANS = deterministic bar-pointer dynamics on sampled z_{t-1}
        tempo_prev = torch.exp(log_tempo_prev.clamp(-12.0, 6.0))
        advance = phi_prev + tempo_prev
        cross = (advance >= TWO_PI).to(h.dtype)                     # [B] bar-crossing indicator
        p_ph_mu = advance % TWO_PI
        p_ph_rho = model.prior_phase_conc(prior_ctx[:, t])
        a = model.prior_dev_coef(prior_ctx[:, t])
        p_lv_mu = level_anchor + a_lv * (level_prev - level_anchor)  # slow OU: revert to song level
        p_lv_s = model.prior_level_scale(prior_ctx[:, t])
        p_dv_mu = a * dev_prev                                      # AR(1) revert to 0
        p_dv_s = model.prior_dev_scale(prior_ctx[:, t])

        # sample current latents from the posterior
        phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
        level = sample_student_t(dof, q_lv_mu, q_lv_s)
        dev = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
        log_tempo = level + dev

        # meter: bar-gated -- resample from q only at a crossing, else copy m_{t-1}
        q_meter_draw = gumbel_softmax(q_m, temperature)
        meter = torch.where(cross.unsqueeze(-1) > 0.5, q_meter_draw, meter_prev)
        log_pi_p = model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[:, t])

        kl_p = kl_p + kl_wrapped_cauchy(q_ph_mu, q_ph_rho, p_ph_mu, p_ph_rho)
        kl_lv = kl_lv + kl_student_t_mc(dof, q_lv_mu, q_lv_s, p_lv_mu, p_lv_s, level)
        kl_dv = kl_dv + kl_log_normal(q_dv_mu, q_dv_s, p_dv_mu, p_dv_s)
        kl_m = kl_m + cross * kl_categorical(torch.log_softmax(q_m, -1), log_pi_p)  # gated
        n_cross = n_cross + cross

        z_feats.append(model.z_features(meter, phi, log_tempo))
        post_phase_mu.append(q_ph_mu)
        meter_prev, phi_prev = meter, phi
        level_prev, dev_prev, log_tempo_prev = level, dev, log_tempo

    # ---------- decode: two-Bernoulli emission (beat + downbeat) ----------
    logits = torch.stack([model.decode(z_feats[t], prior_ctx[:, t]) for t in range(T)], dim=1)  # [B,T,2]
    beat_logits, db_logits = logits[..., 0], logits[..., 1]
    recon_b = F.binary_cross_entropy_with_logits(beat_logits, b, reduction="none").sum(1)   # [B]
    recon_db = F.binary_cross_entropy_with_logits(db_logits, db, reduction="none").sum(1)    # [B]
    recon = recon_b + recon_db
    L_kl = kl_m + kl_p + kl_lv + kl_dv                             # [B]
    loss = (recon + beta * L_kl).mean()                            # beta=1 -> strict ELBO; <1 = KL warm-up

    info = {
        "loss": float(loss), "recon": float(recon.mean()),
        "recon_beat": float(recon_b.mean()), "recon_db": float(recon_db.mean()),
        "kl": float(L_kl.mean()), "kl_meter": float(kl_m.mean()), "kl_phase": float(kl_p.mean()),
        "kl_level": float(kl_lv.mean()), "kl_dev": float(kl_dv.mean()),
        "n_cross": float(n_cross.mean()), "tempo_dof": float(dof.detach()), "beta": float(beta),
        "beat_prob": torch.sigmoid(beat_logits).detach(),
        "db_prob": torch.sigmoid(db_logits).detach(),
        "post_phase_mu": torch.stack(post_phase_mu, dim=1).detach(),
    }
    return loss, info


@torch.no_grad()
def free_run(model, h: torch.Tensor, temperature: float = 0.3):
    """Deploy path: roll the PRIOR forward with NO beats (the test-time generative model)."""
    B, T, _ = h.shape
    prior_ctx = model.encode_prior(h)
    dof = model.tempo_dof()

    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _p_dv_mu, _p_dv_s = model.unpack(
        model.prior_init_head(prior_ctx.mean(1)))
    a0 = model.prior_dev_coef(prior_ctx[:, 0])
    sd0 = model.prior_dev_scale(prior_ctx[:, 0])
    p_dv_s = _stationary_dev_sigma(sd0, a0)

    meter = gumbel_softmax(p_m, temperature)
    phi = sample_wrapped_cauchy(p_ph_mu, p_ph_rho)
    level = sample_student_t(dof, p_lv_mu, p_lv_s)
    dev = p_dv_s * torch.randn_like(p_dv_s)               # mean 0
    log_tempo = level + dev

    # noise-free MEAN chain for the phase-wrap read-out
    phi_mu = p_ph_mu % TWO_PI
    level_mu = p_lv_mu
    dev_mu = torch.zeros_like(p_lv_mu)

    level_anchor = level                                 # per-song OU target (stochastic path)
    level_mu_anchor = level_mu                           # per-song OU target (mean chain)
    a_lv = model.level_ar()
    z_feats = [model.z_features(meter, phi, log_tempo)]
    phase_traj, phase_mu_traj = [phi], [phi_mu]
    log_tempo_traj, meter_traj = [log_tempo], [meter.argmax(-1)]
    meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo
    level_prev, dev_prev = level, dev

    for t in range(1, T):
        tempo_prev = torch.exp(log_tempo_prev.clamp(-12.0, 6.0))
        advance = phi_prev + tempo_prev
        cross = (advance >= TWO_PI)
        p_ph_mu = advance % TWO_PI
        p_ph_rho = model.prior_phase_conc(prior_ctx[:, t])
        a = model.prior_dev_coef(prior_ctx[:, t])
        p_lv_s = model.prior_level_scale(prior_ctx[:, t])
        p_dv_s = model.prior_dev_scale(prior_ctx[:, t])

        phi = sample_wrapped_cauchy(p_ph_mu, p_ph_rho)
        level = sample_student_t(dof, level_anchor + a_lv * (level_prev - level_anchor), p_lv_s)  # OU
        dev = a * dev_prev + p_dv_s * torch.randn_like(p_dv_s)     # AR(1) deviation
        log_tempo = level + dev
        q_meter_draw = gumbel_softmax(model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[:, t]), temperature)
        meter = torch.where(cross.unsqueeze(-1), q_meter_draw, meter_prev)

        # deterministic mean chain: level reverts to per-song anchor, dev decays by a
        level_mu = level_mu_anchor + a_lv * (level_mu - level_mu_anchor)
        dev_mu = a * dev_mu
        log_tempo_mu = level_mu + dev_mu
        phi_mu = (phi_mu + torch.exp(log_tempo_mu)) % TWO_PI

        z_feats.append(model.z_features(meter, phi, log_tempo))
        phase_traj.append(phi); phase_mu_traj.append(phi_mu)
        log_tempo_traj.append(log_tempo); meter_traj.append(meter.argmax(-1))
        meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo
        level_prev, dev_prev = level, dev

    logits = torch.stack([model.decode(z_feats[t], prior_ctx[:, t]) for t in range(T)], dim=1)  # [B,T,2]
    return {
        "phase": torch.stack(phase_traj, dim=1),
        "phase_mu": torch.stack(phase_mu_traj, dim=1),
        "log_tempo": torch.stack(log_tempo_traj, dim=1),
        "meter": torch.stack(meter_traj, dim=1),
        "decoder_prob": torch.sigmoid(logits[..., 0]),      # beat channel
        "downbeat_prob": torch.sigmoid(logits[..., 1]),     # downbeat channel
    }
