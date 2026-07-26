"""VARIANT A -- AUDIO-CONDITIONED PRIOR MEAN for VBPM.

Root cause being repaired (established, not re-litigated): the deploy path
``vbpm/elbo.py:free_run`` is an open-loop metronome.  The prior recursion MEAN is
``(phi_prev + tempo_prev)`` with no ``h`` anywhere, so once ``prior_init_head`` picks a
start phase and a tempo the chain runs forever without ever looking at the audio.  Audio
only touches concentrations / scales / the meter transition, none of which move the mean.

Variant A makes ``p_psi(z_t | z_{t-1}, h)`` genuinely h-dependent IN ITS MEAN:

    corr_t   = tanh(W_phi  prior_ctx_t) * CORR_SCALE
    p_ph_mu  = (phi_prev + tempo_prev + corr_t) % 2pi          # phase mean now reads audio
    tcorr_t  = tanh(W_tmp  prior_ctx_t) * TEMPO_CORR_SCALE     # optional
    p_lv_mu  = level_anchor + a_lv*(level_prev - level_anchor) + tcorr_t

The SAME correction is applied in the training rollout and in free-run (including the
deterministic mean chain that the phase-wrap read-out uses) -- otherwise train and deploy
would be different models.

Also implemented (separately ablatable): TEMPO INIT FIX -- bias the level-mean output of
``prior_init_head``/``post_head`` to log(2pi/(median_IBI*m*fps)) ~ -2.66 instead of 0, which
is ~14x too fast and causes wrap ALIASING (advance > pi/frame).

NOTHING in vbpm/ is modified: ``AudioCondPriorVAE`` subclasses ``BarPointerVAE`` and the
ELBO / free-run are private copies of ``vbpm/elbo.py`` with the correction spliced in.
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

# log(2*pi / (median_IBI * m * fps)) with median_IBI ~ 0.5 s, m = 4, fps = 50
TEMPO_INIT_LOG = -2.66


def _stationary_dev_sigma(sigma, a):
    return sigma / torch.sqrt((1.0 - a ** 2).clamp(min=1e-3))


class AudioCondPriorVAE(BarPointerVAE):
    """BarPointerVAE + audio-conditioned prior recursion MEANS (Variant A)."""

    def __init__(self, *args, corr_scale: float = 0.5, tempo_corr_scale: float = 0.0,
                 tempo_init: bool = True, **kw):
        super().__init__(*args, **kw)
        self.corr_scale = float(corr_scale)
        self.tempo_corr_scale = float(tempo_corr_scale)
        # f^corr_psi : the audio read that MOVES the phase mean
        self.prior_phase_corr = nn.Linear(self.hidden, 1)
        nn.init.zeros_(self.prior_phase_corr.weight); nn.init.zeros_(self.prior_phase_corr.bias)
        # optional audio read that nudges the log-tempo LEVEL mean
        self.prior_tempo_corr = nn.Linear(self.hidden, 1)
        nn.init.zeros_(self.prior_tempo_corr.weight); nn.init.zeros_(self.prior_tempo_corr.bias)
        if tempo_init:
            K = self.K
            # index K+3 of the raw parameter vector is the log-tempo LEVEL mean
            with torch.no_grad():
                self.prior_init_head[-1].bias[K + 3] = TEMPO_INIT_LOG
                self.post_head[-1].bias[K + 3] = TEMPO_INIT_LOG

    def phase_corr(self, prior_ctx_t):
        if self.corr_scale == 0.0:
            return torch.zeros(prior_ctx_t.shape[0], device=prior_ctx_t.device,
                               dtype=prior_ctx_t.dtype)
        return torch.tanh(self.prior_phase_corr(prior_ctx_t).squeeze(-1)) * self.corr_scale

    def tempo_corr(self, prior_ctx_t):
        if self.tempo_corr_scale == 0.0:
            return torch.zeros(prior_ctx_t.shape[0], device=prior_ctx_t.device,
                               dtype=prior_ctx_t.dtype)
        return torch.tanh(self.prior_tempo_corr(prior_ctx_t).squeeze(-1)) * self.tempo_corr_scale


# ---------------------------------------------------------------- training rollout
def elbo_A(model, h, b, db, temperature: float = 0.5, beta: float = 1.0):
    """Copy of vbpm.elbo.strict_elbo with the audio-conditioned prior means spliced in."""
    B, T, _ = h.shape
    post_ctx = model.encode_posterior(h, b)
    prior_ctx = model.encode_prior(h)
    dof = model.tempo_dof()

    kl_m = h.new_zeros(B); kl_p = h.new_zeros(B)
    kl_lv = h.new_zeros(B); kl_dv = h.new_zeros(B)
    z_feats = []
    n_cross = h.new_zeros(B)
    corr_abs = h.new_zeros(B)

    # ---- t = 1 ----
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
        model.post_head(torch.cat([post_ctx[:, 0], z0], dim=-1)))
    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _pa, _pb = model.unpack(
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

    # ---- t = 2..T ----
    for t in range(1, T):
        z_prev_feat = model.z_features(meter_prev, phi_prev, log_tempo_prev)
        q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
            model.post_head(torch.cat([post_ctx[:, t], z_prev_feat], dim=-1)))

        tempo_prev = torch.exp(log_tempo_prev.clamp(-12.0, 6.0))
        corr = model.phase_corr(prior_ctx[:, t])                 # <-- AUDIO MOVES THE MEAN
        advance = phi_prev + tempo_prev + corr
        cross = (advance >= TWO_PI).to(h.dtype)
        p_ph_mu = advance % TWO_PI
        p_ph_rho = model.prior_phase_conc(prior_ctx[:, t])
        a = model.prior_dev_coef(prior_ctx[:, t])
        p_lv_mu = (level_anchor + a_lv * (level_prev - level_anchor)
                   + model.tempo_corr(prior_ctx[:, t]))          # <-- optional tempo read
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
        corr_abs = corr_abs + corr.abs()

        z_feats.append(model.z_features(meter, phi, log_tempo))
        meter_prev, phi_prev = meter, phi
        level_prev, dev_prev, log_tempo_prev = level, dev, log_tempo

    logits = torch.stack([model.decode(z_feats[t], prior_ctx[:, t]) for t in range(T)], dim=1)
    beat_logits, db_logits = logits[..., 0], logits[..., 1]
    recon_b = F.binary_cross_entropy_with_logits(beat_logits, b, reduction="none").sum(1)
    recon_db = F.binary_cross_entropy_with_logits(db_logits, db, reduction="none").sum(1)
    recon = recon_b + recon_db
    L_kl = kl_m + kl_p + kl_lv + kl_dv
    loss = (recon + beta * L_kl).mean()

    info = {
        "loss": float(loss), "recon": float(recon.mean()),
        "recon_beat": float(recon_b.mean()), "recon_db": float(recon_db.mean()),
        "kl": float(L_kl.mean()), "kl_meter": float(kl_m.mean()), "kl_phase": float(kl_p.mean()),
        "kl_level": float(kl_lv.mean()), "kl_dev": float(kl_dv.mean()),
        "n_cross": float(n_cross.mean()), "tempo_dof": float(dof.detach()), "beta": float(beta),
        "corr_mean_abs": float(corr_abs.mean() / max(T - 1, 1)),
        "log_tempo_mean": float(log_tempo_prev.mean().detach()),
    }
    return loss, info


# ---------------------------------------------------------------- deploy path
@torch.no_grad()
def free_run_A(model, h, temperature: float = 0.3):
    """Copy of vbpm.elbo.free_run with the SAME audio-conditioned means as elbo_A.

    The correction is applied to BOTH the stochastic chain and the deterministic mean
    chain (``phase_mu``) -- the mean chain is what the phase-wrap read-out consumes, so
    that is where the audio-blindness had to be broken.
    """
    B, T, _ = h.shape
    prior_ctx = model.encode_prior(h)
    dof = model.tempo_dof()

    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _pa, _pb = model.unpack(
        model.prior_init_head(prior_ctx.mean(1)))
    a0 = model.prior_dev_coef(prior_ctx[:, 0])
    sd0 = model.prior_dev_scale(prior_ctx[:, 0])
    p_dv_s = _stationary_dev_sigma(sd0, a0)

    meter = gumbel_softmax(p_m, temperature)
    phi = sample_wrapped_cauchy(p_ph_mu, p_ph_rho)
    level = sample_student_t(dof, p_lv_mu, p_lv_s)
    dev = p_dv_s * torch.randn_like(p_dv_s)
    log_tempo = level + dev

    phi_mu = p_ph_mu % TWO_PI
    level_mu = p_lv_mu
    dev_mu = torch.zeros_like(p_lv_mu)

    level_anchor = level
    level_mu_anchor = level_mu
    a_lv = model.level_ar()
    z_feats = [model.z_features(meter, phi, log_tempo)]
    phase_traj, phase_mu_traj = [phi], [phi_mu]
    log_tempo_traj, meter_traj = [log_tempo], [meter.argmax(-1)]
    corr_traj = [torch.zeros_like(phi_mu)]
    meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo
    level_prev, dev_prev = level, dev

    for t in range(1, T):
        corr = model.phase_corr(prior_ctx[:, t])                 # <-- AUDIO MOVES THE MEAN
        tcorr = model.tempo_corr(prior_ctx[:, t])

        tempo_prev = torch.exp(log_tempo_prev.clamp(-12.0, 6.0))
        advance = phi_prev + tempo_prev + corr
        cross = (advance >= TWO_PI)
        p_ph_mu = advance % TWO_PI
        p_ph_rho = model.prior_phase_conc(prior_ctx[:, t])
        a = model.prior_dev_coef(prior_ctx[:, t])
        p_lv_s = model.prior_level_scale(prior_ctx[:, t])
        p_dv_s = model.prior_dev_scale(prior_ctx[:, t])

        phi = sample_wrapped_cauchy(p_ph_mu, p_ph_rho)
        level = sample_student_t(
            dof, level_anchor + a_lv * (level_prev - level_anchor) + tcorr, p_lv_s)
        dev = a * dev_prev + p_dv_s * torch.randn_like(p_dv_s)
        log_tempo = level + dev
        q_meter_draw = gumbel_softmax(
            model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[:, t]), temperature)
        meter = torch.where(cross.unsqueeze(-1), q_meter_draw, meter_prev)

        # deterministic mean chain -- SAME correction (train/deploy must match exactly)
        level_mu = level_mu_anchor + a_lv * (level_mu - level_mu_anchor) + tcorr
        dev_mu = a * dev_mu
        log_tempo_mu = (level_mu + dev_mu).clamp(-12.0, 6.0)
        phi_mu = (phi_mu + torch.exp(log_tempo_mu) + corr) % TWO_PI

        z_feats.append(model.z_features(meter, phi, log_tempo))
        phase_traj.append(phi); phase_mu_traj.append(phi_mu)
        log_tempo_traj.append(log_tempo); meter_traj.append(meter.argmax(-1))
        corr_traj.append(corr)
        meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo
        level_prev, dev_prev = level, dev

    logits = torch.stack([model.decode(z_feats[t], prior_ctx[:, t]) for t in range(T)], dim=1)
    return {
        "phase": torch.stack(phase_traj, dim=1),
        "phase_mu": torch.stack(phase_mu_traj, dim=1),
        "log_tempo": torch.stack(log_tempo_traj, dim=1),
        "meter": torch.stack(meter_traj, dim=1),
        "corr": torch.stack(corr_traj, dim=1),
        "decoder_prob": torch.sigmoid(logits[..., 0]),
        "downbeat_prob": torch.sigmoid(logits[..., 1]),
    }
