"""VARIANT B -- latent generates the OBSERVATION, and deploy = BOOTSTRAP PARTICLE FILTER.

Root cause being fixed (established, not re-litigated here):
  vbpm/elbo.py :: free_run is an OPEN-LOOP METRONOME.  The prior recursion mean is
  (phi_prev + tempo_prev) with no h, and the only *observed* variable in the likelihood is b
  (the prediction target), which is absent at deploy.  So free_run draws (tempo, phase) once
  from prior_init_head and rolls forward, audio-blind.

Fix (this file):
  (1) add an observation decoder  p_theta(h_t | z_t)   -- the latent GENERATES the audio
      observation.  Dirac h  -> two-Bernoulli on the impulse channels.
      MERT  h  -> Gaussian (learned per-dim scale) on a FIXED random 32-d projection of h,
      z-scored over time inside the crop and DETACHED (so there is no incentive to shrink /
      collapse h to make it "predictable" -- the target is scale-free and carries no gradient).
  (2) ELBO gains   + log p_theta(h_t | z_t)   next to the existing p(b_t|z_t) and the KLs.
  (3) DEPLOY = bootstrap particle filter.  Propagate K particles through the PRIOR transition,
      weight by p_theta(h_t | z_t^k) using the trained observation decoder, systematic-resample
      when ESS < K/2.  This is structurally madmom's DBN forward pass: transition + observation.
  (4) tempo-init fix: `unpack` adds a constant offset to the log-tempo LEVEL so that a
      zero-output head means the *correct* bar-advance rate (log phidot ~ -2.66 rad/frame,
      = 133 BPM at m=4) instead of log_tempo ~ 0 (14x too fast -> wrap aliasing).

Everything here SUBCLASSES / COPIES vbpm/ -- nothing in vbpm/ is mutated.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vbpm.model import BarPointerVAE
from vbpm.distributions import (
    TWO_PI, gumbel_softmax, sample_wrapped_cauchy, sample_student_t,
    kl_categorical, kl_wrapped_cauchy, kl_log_normal, kl_student_t_mc,
)

# log of the true mean bar-advance (rad/frame). Median beat IBI over the train cache is
# 0.502 s -> phidot = 2*pi/(4*0.502*50) = 0.0626 -> log = -2.77.  Task note quotes -2.66.
LOG_TEMPO_INIT = -2.66


class BarPointerVAE_B(BarPointerVAE):
    """BarPointerVAE + p_theta(h_t|z_t) observation decoder + fixed tempo init."""

    def __init__(self, h_dim: int, hidden: int = 128, num_meters: int = 4,
                 obs_dim: int = 2, obs_type: str = "bern",
                 log_tempo_init: float = LOG_TEMPO_INIT, **kw):
        super().__init__(h_dim=h_dim, hidden=hidden, num_meters=num_meters, **kw)
        self.obs_dim = obs_dim
        self.obs_type = obs_type
        # observation decoder: LATENT-ONLY, exactly as specified.
        self.h_dec = nn.Sequential(
            nn.Linear(self.z_feat_dim, hidden), nn.Tanh(), nn.Linear(hidden, obs_dim))
        if obs_type == "gauss":
            self.obs_log_sigma = nn.Parameter(torch.zeros(obs_dim))
        self.register_buffer("level_offset", torch.tensor(float(log_tempo_init)))

    # ---- tempo-init fix: shift the log-tempo LEVEL so head-output 0 == real tempo ----
    def unpack(self, vec: torch.Tensor):
        (meter_logits, phase_mu, phase_rho, level_mu, level_sigma,
         dev_mu, dev_sigma) = super().unpack(vec)
        return (meter_logits, phase_mu, phase_rho, level_mu + self.level_offset,
                level_sigma, dev_mu, dev_sigma)

    # ---- p_theta(h_t | z_t) ----
    def obs_logp(self, z_feat: torch.Tensor, o_t: torch.Tensor) -> torch.Tensor:
        """log p(o_t | z_t), summed over obs dims. z_feat [N,z_feat_dim], o_t [N,obs_dim]."""
        pred = self.h_dec(z_feat)
        if self.obs_type == "bern":
            return -F.binary_cross_entropy_with_logits(pred, o_t, reduction="none").sum(-1)
        sig = F.softplus(self.obs_log_sigma) + 1e-3
        return (-0.5 * ((o_t - pred) / sig) ** 2 - torch.log(sig)
                - 0.5 * math.log(TWO_PI)).sum(-1)


# ---------------------------------------------------------------------------
# observation targets
# ---------------------------------------------------------------------------
def dirac_obs(h: torch.Tensor) -> torch.Tensor:
    """Dirac h -> binary impulse channels (beat, downbeat). [B,T,h_dim] -> [B,T,2]."""
    return (h[..., :2] > 0.5).to(h.dtype)


class MertObsProjector(nn.Module):
    """FIXED random projection of the merged MERT h to obs_dim, z-scored over time per crop.

    Fixed + detached on purpose: the observation TARGET must not be trainable, otherwise the
    model can trivially maximise log p(h|z) by making h constant (representation collapse).
    z-scoring over time makes the target scale-free, so shrinking h buys nothing either.
    """

    def __init__(self, h_dim: int, obs_dim: int = 32, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        P = torch.randn(h_dim, obs_dim, generator=g) / math.sqrt(h_dim)
        self.register_buffer("P", P)

    @torch.no_grad()
    def forward(self, h: torch.Tensor) -> torch.Tensor:      # [B,T,h_dim] -> [B,T,obs_dim]
        o = h.detach() @ self.P
        mu = o.mean(1, keepdim=True)
        sd = o.std(1, keepdim=True).clamp(min=1e-4)
        return (o - mu) / sd


# ---------------------------------------------------------------------------
# ELBO with the observation likelihood  (copy of vbpm/elbo.py::strict_elbo + obs term)
# ---------------------------------------------------------------------------
def _stationary_dev_sigma(sigma, a):
    return sigma / torch.sqrt((1.0 - a ** 2).clamp(min=1e-3))


def elbo_b(model, h, b, db, obs, temperature: float = 0.5, beta: float = 1.0,
           obs_w: float = 1.0):
    """L = -[ log p(b|z) + log p(db|z) + obs_w * log p(h|z) ] + beta * sum KL."""
    B, T, _ = h.shape
    post_ctx = model.encode_posterior(h, b)
    prior_ctx = model.encode_prior(h)
    dof = model.tempo_dof()

    kl_m = h.new_zeros(B); kl_p = h.new_zeros(B)
    kl_lv = h.new_zeros(B); kl_dv = h.new_zeros(B)
    z_feats = []
    n_cross = h.new_zeros(B)

    z0 = model.z0.unsqueeze(0).expand(B, -1)
    q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
        model.post_head(torch.cat([post_ctx[:, 0], z0], dim=-1)))
    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _a, _c = model.unpack(
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

    for t in range(1, T):
        z_prev_feat = model.z_features(meter_prev, phi_prev, log_tempo_prev)
        q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
            model.post_head(torch.cat([post_ctx[:, t], z_prev_feat], dim=-1)))

        tempo_prev = torch.exp(log_tempo_prev.clamp(-12.0, 6.0))
        advance = phi_prev + tempo_prev
        cross = (advance >= TWO_PI).to(h.dtype)
        p_ph_mu = advance % TWO_PI
        p_ph_rho = model.prior_phase_conc(prior_ctx[:, t])
        a = model.prior_dev_coef(prior_ctx[:, t])
        p_lv_mu = level_anchor + a_lv * (level_prev - level_anchor)
        p_lv_s = model.prior_level_scale(prior_ctx[:, t])
        p_dv_mu = a * dev_prev
        p_dv_s = model.prior_dev_scale(prior_ctx[:, t])

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

    Z = torch.stack(z_feats, dim=1)                                     # [B,T,z_feat]
    logits = model.decoder(Z)                                           # latent-only decode
    beat_logits, db_logits = logits[..., 0], logits[..., 1]
    recon_b = F.binary_cross_entropy_with_logits(beat_logits, b, reduction="none").sum(1)
    recon_db = F.binary_cross_entropy_with_logits(db_logits, db, reduction="none").sum(1)

    # --- the new term: -log p_theta(h_t | z_t) ---
    obs_ll = model.obs_logp(Z.reshape(B * T, -1), obs.reshape(B * T, -1)).reshape(B, T).sum(1)
    recon_obs = -obs_ll

    L_kl = kl_m + kl_p + kl_lv + kl_dv
    loss = (recon_b + recon_db + obs_w * recon_obs + beta * L_kl).mean()

    info = {"loss": float(loss), "recon_beat": float(recon_b.mean()),
            "recon_db": float(recon_db.mean()), "recon_obs": float(recon_obs.mean()),
            "kl": float(L_kl.mean()), "kl_phase": float(kl_p.mean()),
            "kl_level": float(kl_lv.mean()), "kl_dev": float(kl_dv.mean()),
            "kl_meter": float(kl_m.mean()), "n_cross": float(n_cross.mean()),
            "tempo_dof": float(dof.detach())}
    return loss, info


# ---------------------------------------------------------------------------
# DEPLOY: bootstrap particle filter
# ---------------------------------------------------------------------------
def _systematic_resample(w: torch.Tensor) -> torch.Tensor:
    K = w.shape[0]
    pos = (torch.arange(K, device=w.device, dtype=w.dtype)
           + torch.rand(1, device=w.device)) / K
    cdf = torch.cumsum(w, 0)
    cdf = cdf / cdf[-1].clamp(min=1e-30)
    return torch.searchsorted(cdf.contiguous(), pos.contiguous()).clamp(max=K - 1)


def _circ_wmean(phi: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.atan2((w * torch.sin(phi)).sum(), (w * torch.cos(phi)).sum()) % TWO_PI


@torch.no_grad()
def particle_filter(model, h: torch.Tensor, obs: torch.Tensor, K: int = 400,
                    alpha: float = 1.0, diffuse_init: bool = True,
                    lt_lo: float = -3.55, lt_hi: float = -2.18,
                    ess_frac: float = 0.5):
    """Bootstrap PF over the model's own prior transition, weighted by p_theta(h_t|z_t).

    h/obs are [1,T,*].  `diffuse_init` = uniform initial phase and log-tempo over a 55-215 BPM
    band (at m=4), which is exactly madmom's uniform initial DBN state distribution;
    `diffuse_init=False` instead samples z_1 from the model's own prior_init_head.
    """
    assert h.shape[0] == 1
    T = h.shape[1]
    dv = h.device
    prior_ctx = model.encode_prior(h)                       # [1,T,hidden]
    ctx = prior_ctx[0]
    dof = model.tempo_dof()
    a_lv = model.level_ar()
    Km = model.K

    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _a, _b = model.unpack(
        model.prior_init_head(prior_ctx.mean(1)))
    if diffuse_init:
        phi = torch.rand(K, device=dv) * TWO_PI
        level = lt_lo + (lt_hi - lt_lo) * torch.rand(K, device=dv)
    else:
        phi = sample_wrapped_cauchy(p_ph_mu.expand(K), p_ph_rho.expand(K))
        level = sample_student_t(dof, p_lv_mu.expand(K), p_lv_s.expand(K))
    dev = torch.zeros(K, device=dv)
    log_tempo = level + dev
    m_idx = torch.multinomial(F.softmax(p_m, -1).expand(K, Km), 1).squeeze(1)
    meter = F.one_hot(m_idx, Km).to(h.dtype)
    anchor = level.clone()

    zf = model.z_features(meter, phi, log_tempo)
    logw = alpha * model.obs_logp(zf, obs[0, 0].unsqueeze(0).expand(K, -1))
    w = F.softmax(logw, 0)

    phase_mean = [_circ_wmean(phi, w)]
    phase_map = [phi[w.argmax()]]
    lt_map = [log_tempo[w.argmax()]]
    meter_map = [int(meter[w.argmax()].argmax())]
    ess_hist = [float(1.0 / (w ** 2).sum())]

    for t in range(1, T):
        ctx_t = ctx[t].unsqueeze(0).expand(K, -1)
        advance = phi + torch.exp(log_tempo.clamp(-12.0, 6.0))
        cross = advance >= TWO_PI
        p_ph_mu_t = advance % TWO_PI
        rho = model.prior_phase_conc(ctx_t)
        a = model.prior_dev_coef(ctx_t)
        s_lv = model.prior_level_scale(ctx_t)
        s_dv = model.prior_dev_scale(ctx_t)

        phi_new = sample_wrapped_cauchy(p_ph_mu_t, rho)
        level_new = sample_student_t(dof, anchor + a_lv * (level - anchor), s_lv)
        dev_new = a * dev + s_dv * torch.randn(K, device=dv)
        lt_new = level_new + dev_new

        log_pi = model.meter_prior_logp(meter, phi_new, phi, ctx_t)     # [K,Km]
        draw = torch.multinomial(log_pi.exp().clamp(min=1e-12), 1).squeeze(1)
        meter_new = torch.where(cross.unsqueeze(-1),
                                F.one_hot(draw, Km).to(h.dtype), meter)

        phi, level, dev, log_tempo, meter = phi_new, level_new, dev_new, lt_new, meter_new

        zf = model.z_features(meter, phi, log_tempo)
        logw = logw + alpha * model.obs_logp(zf, obs[0, t].unsqueeze(0).expand(K, -1))
        w = F.softmax(logw, 0)

        ess = float(1.0 / (w ** 2).sum())
        phase_mean.append(_circ_wmean(phi, w))
        phase_map.append(phi[w.argmax()])
        lt_map.append(log_tempo[w.argmax()])
        meter_map.append(int(meter[w.argmax()].argmax()))
        ess_hist.append(ess)

        if ess < ess_frac * K:
            idx = _systematic_resample(w)
            phi, level, dev, log_tempo = phi[idx], level[idx], dev[idx], log_tempo[idx]
            meter, anchor = meter[idx], anchor[idx]
            logw = torch.zeros(K, device=dv)
            w = torch.full((K,), 1.0 / K, device=dv)

    return {
        "phase_mean": torch.stack(phase_mean).cpu(),
        "phase_map": torch.stack(phase_map).cpu(),
        "log_tempo": torch.stack(lt_map).cpu(),
        "meter_map": meter_map,
        "ess": float(sum(ess_hist) / len(ess_hist)),
    }
