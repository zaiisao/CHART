"""EXPERIMENT 3 -- BOTH FIXES TOGETHER.

FIX 1 (cut the tempo side-channel):  the two Bernoulli decoders p(b|z), p(db|z) see
       [cos phi, sin phi, meter one-hots] ONLY.  log_tempo is removed from their input,
       so the encoder can no longer smuggle the beat pattern through a time-varying
       log-tempo -- the ONLY way to reconstruct beats is to put them in the BAR PHASE.
       (log_tempo is still a latent: it drives the phase advance and the posterior
        recursion context, exactly as the generative model says it should.)

FIX 2 (frozen supervised observation model):  p(o_t | phi_t, m_t) is the E2 table,
       fitted on the TRAIN fold from labels and FROZEN.  It is inserted into the ELBO
       as the observation likelihood, so the VAE trains its dynamics and posterior
       AROUND an emission that is already phase-tuned instead of having to discover
       phase-tuning by itself (which it demonstrably never does).

Nothing in vbpm/, vbpm_fix/, vbpm_arms/ is modified: the model SUBCLASSES
vbpm_fix/variant_b.py::BarPointerVAE_B and the ELBO is a copy of ``elbo_b`` with the
decoder input masked.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

for _p in ("/home/sogang/jaehoon/VBPM_reintegration",
           "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
           "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms",
           "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import variant_b as VB                                                  # noqa: E402
from vbpm.distributions import (                                        # noqa: E402
    TWO_PI, gumbel_softmax, sample_wrapped_cauchy, sample_student_t,
    kl_categorical, kl_wrapped_cauchy, kl_log_normal, kl_student_t_mc,
)

METERS = (2, 3, 4)


# =============================================================== frozen emission
class FrozenPhaseEmission(nn.Module):
    """log p(o_t | bar phase, meter) from the SUPERVISED table, as a frozen torch module.

    The numpy table is per-(meter, phase-bin).  Here it is made DIFFERENTIABLE in phi by
    circular LINEAR INTERPOLATION between adjacent bins, so the ELBO's reparameterised
    phase sample receives a real gradient from the observation term (a hard bin lookup
    would give zero gradient and reproduce the dead-phase pathology we are fixing).

    Meter class j corresponds to meter `meters[j]` (default 2,3,4).  There is deliberately
    NO "meter 1" / phase-marginal class: a phase-flat class is a likelihood ESCAPE HATCH --
    measured on the train fold it beats a MISALIGNED sharp table by +0.96 nats/frame while
    losing to the aligned table by only 0.62, so a model that starts with random phase can
    make the whole observation term phase-blind by parking meter mass on it.  Every class
    here is phase-tuned.
    """

    def __init__(self, emis, meters=(2, 3, 4)):
        super().__init__()
        self.meters = tuple(meters)
        self.K = len(self.meters)
        self.likelihood = emis.likelihood
        Bmax = max(emis.nb[m] for m in self.meters)
        self.Bmax = Bmax
        mu = np.zeros((self.K, Bmax, 2), np.float32)
        sd = np.ones((self.K, Bmax, 2), np.float32)
        nb = np.ones(self.K, np.int64)
        for j, m in enumerate(self.meters):
            n = emis.nb[m]
            nb[j] = n
            mu[j, :n] = emis.mu[m]
            sd[j, :n] = np.maximum(emis.sd[m], 1e-2)
        self.register_buffer("mu", torch.from_numpy(mu))
        self.register_buffer("sd", torch.from_numpy(sd))
        self.register_buffer("nb", torch.from_numpy(nb))
        for q in self.parameters():
            q.requires_grad_(False)

    def logp_phi(self, phi: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """[N] phase, [N,2] observation -> [N,K] log p(o|phi, meter=j+1) for every meter."""
        N = phi.shape[0]
        u = (phi.unsqueeze(-1) % TWO_PI) / TWO_PI * self.nb.to(phi.dtype)   # [N,K]
        i0 = torch.floor(u)
        f = (u - i0).unsqueeze(-1)                                          # [N,K,1]
        nbf = self.nb.to(torch.long).unsqueeze(0)                           # [1,K]
        i0 = i0.to(torch.long) % nbf
        i1 = (i0 + 1) % nbf
        j = torch.arange(self.K, device=phi.device).unsqueeze(0).expand(N, -1)
        mu = (1 - f) * self.mu[j, i0] + f * self.mu[j, i1]                  # [N,K,2]
        sd = ((1 - f) * self.sd[j, i0] + f * self.sd[j, i1]).clamp(min=1e-2)
        if self.likelihood == "bern":
            p = mu.clamp(1e-3, 1 - 1e-3)
            oo = o.unsqueeze(1)
            return (oo * torch.log(p) + (1 - oo) * torch.log(1 - p)).sum(-1)
        oo = o.unsqueeze(1)                                                 # [N,1,2]
        return (-0.5 * ((oo - mu) / sd) ** 2 - torch.log(sd)
                - 0.5 * math.log(TWO_PI)).sum(-1)                           # [N,K]

    def logp_zfeat(self, z_feat: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """z_feat = [cos phi, sin phi, log_tempo, meter one-hots] -> [N] log-likelihood.
        NOTE: log_tempo (column 2) is NOT read -- the observation model is phase+meter only."""
        phi = torch.atan2(z_feat[..., 1], z_feat[..., 0]) % TWO_PI
        lp = self.logp_phi(phi, o)                                          # [N,K]
        w = z_feat[..., 3:3 + self.K]
        return (w * lp).sum(-1)


# =============================================================== the E3 model
class E3VAE(VB.BarPointerVAE_B):
    """BarPointerVAE_B with (1) log_tempo cut from the beat/downbeat decoders and
    (2) obs_logp replaced by the FROZEN supervised phase emission."""

    def __init__(self, h_dim: int, emission=None, hidden: int = 128,
                 num_meters: int = 3, drop_tempo_from_decoder: bool = True,
                 meter_offset: int = 2, **kw):
        super().__init__(h_dim=h_dim, hidden=hidden, num_meters=num_meters,
                         obs_dim=2, obs_type="gauss", **kw)
        self.drop_tempo = bool(drop_tempo_from_decoder)
        self.meter_offset = int(meter_offset)   # meter VALUE of latent class 0
        dec_in = self.z_feat_dim - (1 if self.drop_tempo else 0)
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        if emission is not None:
            # the LEARNED observation decoder of variant B is replaced wholesale
            del self.h_dec
            if hasattr(self, "obs_log_sigma"):
                del self.obs_log_sigma
            self.emission = emission      # frozen buffers only, no parameters
        else:
            self.emission = None          # CONTROL: keep variant-B's learned emission

    def dec_feat(self, Z: torch.Tensor) -> torch.Tensor:
        """[..., z_feat_dim] -> decoder input.  Drops column 2 = log_tempo."""
        if not self.drop_tempo:
            return Z
        return torch.cat([Z[..., :2], Z[..., 3:]], dim=-1)

    def obs_logp(self, z_feat, o_t):
        if self.emission is None:
            return super().obs_logp(z_feat, o_t)
        return self.emission.logp_zfeat(z_feat, o_t)


# =============================================================== ELBO
def _stationary_dev_sigma(sigma, a):
    return sigma / torch.sqrt((1.0 - a ** 2).clamp(min=1e-3))


def elbo_e3(model, h, b, db, obs, temperature: float = 0.5, beta: float = 1.0,
            obs_w: float = 1.0, want_phase: bool = False):
    """L = -[ log p(b|z) + log p(db|z) + obs_w * log p_frozen(o|phi,m) ] + beta * sum KL.

    Copy of vbpm_fix/variant_b.py::elbo_b with ONE change: the decoder is fed
    model.dec_feat(Z) (no log_tempo).  The observation term now uses the frozen
    supervised emission through model.obs_logp.
    """
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
    logits = model.decoder(model.dec_feat(Z))                           # <-- FIX 1
    beat_logits, db_logits = logits[..., 0], logits[..., 1]
    recon_b = F.binary_cross_entropy_with_logits(beat_logits, b, reduction="none").sum(1)
    recon_db = F.binary_cross_entropy_with_logits(db_logits, db, reduction="none").sum(1)

    obs_ll = model.obs_logp(Z.reshape(B * T, -1), obs.reshape(B * T, -1)).reshape(B, T).sum(1)
    recon_obs = -obs_ll                                                 # <-- FIX 2

    L_kl = kl_m + kl_p + kl_lv + kl_dv
    loss = (recon_b + recon_db + obs_w * recon_obs + beta * L_kl).mean()

    info = {"loss": float(loss), "recon_beat": float(recon_b.mean()),
            "recon_db": float(recon_db.mean()), "recon_obs": float(recon_obs.mean()),
            "kl": float(L_kl.mean()), "kl_phase": float(kl_p.mean()),
            "kl_level": float(kl_lv.mean()), "kl_dev": float(kl_dv.mean()),
            "kl_meter": float(kl_m.mean()), "n_cross": float(n_cross.mean()),
            "tempo_dof": float(dof.detach())}
    if want_phase:
        return loss, info, Z
    return loss, info
