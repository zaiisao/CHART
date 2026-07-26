"""The reality-adjusted bar-pointer VAE (VBPM).

Same variational bar-pointer model as ``faithful/`` -- amortized encoder q_phi(z|h),
reparameterized latents, ELBO training -- with the five certified reality-adjusted
fixes baked in (alternatives workflow, 2026-07-25). It is a VAE, NOT the HMM rungs.

Fixes vs. faithful/ (each a real-data-justified in-model swap; see docstrings):
  1. TEMPO PRIOR    log-Gaussian RW  -> Student-t(nu) increments on the tempo LEVEL
  2. TEMPO DYNAMICS RW-1            -> two-timescale: slow RW level + fast AR(1) deviation
  3. PHASE KERNEL   von Mises        -> wrapped Cauchy (heavy circular tail)
  4. METER          per-frame K x K  -> bar-gated K x K (switch only at phase crossings)
  5. EMISSION       beat-only Bern.  -> beat + downbeat two-Bernoulli (meter-conditioned)

Faithful prior structure is UNCHANGED in spirit: the audio h drives only
concentrations / scales / the meter transition, never the recursion MEANS.
  * phase prior mean   mu^p_phi   = phi_{t-1} + phidot_{t-1}       (bar-pointer advance)
  * level prior mean   mu^p_level = level_{t-1}                    (slow random walk)
  * dev   prior mean   mu^p_dev   = a * dev_{t-1}                  (AR(1) revert to 0)
  * phi = level + dev  ->  log-tempo scalar fed to phase advance / z-features / decoder
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import TWO_PI


class BarPointerVAE(nn.Module):
    def __init__(self, h_dim: int, hidden: int = 64, num_meters: int = 4,
                 latent_only: bool = True, dev_ar_clamp: float = 0.98):
        super().__init__()
        self.K = num_meters
        self.hidden = hidden
        self.latent_only = latent_only
        self.dev_ar_clamp = dev_ar_clamp
        z_feat_dim = 3 + num_meters            # cos phi, sin phi, log_tempo(=level+dev), onehot(meter)
        self.z_feat_dim = z_feat_dim
        # meter logits (K) | phase(u,v)=2 | phase-concentration=1 | level(mu,logsig)=2 | dev(mu,logsig)=2
        param_dim = num_meters + 2 + 1 + 2 + 2
        self.param_dim = param_dim

        # f_phi : inference read of (b, h)
        self.post_gru = nn.GRU(h_dim + 1, hidden, batch_first=True, bidirectional=True)
        self.post_ctx = nn.Linear(2 * hidden, hidden)
        # f_psi : generative read of h
        self.prior_gru = nn.GRU(h_dim, hidden, batch_first=True, bidirectional=True)
        self.prior_ctx = nn.Linear(2 * hidden, hidden)

        # posterior head: [context_t, z_{t-1} features] -> distribution params
        self.post_head = nn.Sequential(
            nn.Linear(hidden + z_feat_dim, hidden), nn.Tanh(), nn.Linear(hidden, param_dim))
        self.z0 = nn.Parameter(torch.zeros(z_feat_dim))   # learned initial token

        # prior heads (audio drives concentrations/scales/transition ONLY)
        self.prior_init_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, param_dim))
        self.prior_phase_rho = nn.Linear(hidden, 1)       # f^phi_psi(h) -> circular concentration
        self.prior_level_sigma = nn.Linear(hidden, 1)     # slow-level RW innovation scale
        self.prior_dev_sigma = nn.Linear(hidden, 1)       # fast-deviation AR(1) innovation scale
        self.prior_dev_ar = nn.Linear(hidden, 1)          # AR(1) coefficient a in (-clamp, clamp)
        self.meter_prior = nn.Sequential(                 # f^m_psi(m_{t-1}, phi_t, phi_{t-1}, h)
            nn.Linear(num_meters + 4 + hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, num_meters * num_meters))

        # global learned Student-t degrees-of-freedom for the tempo LEVEL (floored nu>2 -> finite var)
        self.tempo_log_dof = nn.Parameter(torch.tensor(math.log(2.0)))
        # level mean-reversion (OU) coefficient a_lv = sigmoid(.)*0.99 < 1 : bounds the deploy
        # log-tempo walk (fixes the unbounded-RW blowup) and encodes the reality-check finding
        # that tempo MEAN-REVERTS to a per-song level (VR2=0.53), not just the fast deviation.
        self.level_ar_logit = nn.Parameter(torch.tensor(2.0))

        # decoder NN_theta(z_t, h) -> TWO Bernoulli logits: [beat, downbeat] (identifiability fix)
        dec_in = z_feat_dim if latent_only else z_feat_dim + hidden
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    # nu > 2 keeps the level's variance finite and its free-run mean = loc well-defined
    def tempo_dof(self) -> torch.Tensor:
        return F.softplus(self.tempo_log_dof) + 2.0 + 1e-3   # strictly >2 (finite variance) even under float underflow

    def level_ar(self) -> torch.Tensor:
        """OU coefficient for the slow log-tempo level; <1 -> stationary (bounded) tempo."""
        return torch.sigmoid(self.level_ar_logit) * 0.99

    # ---- shared sequence encoders ----
    def encode_posterior(self, h: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, b.unsqueeze(-1)], dim=-1)        # [B, T, h_dim+1]
        out, _ = self.post_gru(x)
        return torch.tanh(self.post_ctx(out))              # [B, T, hidden]

    def encode_prior(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.prior_gru(h)
        return torch.tanh(self.prior_ctx(out))             # [B, T, hidden]

    # ---- unpack a batched raw parameter vector into named distribution params ----
    def unpack(self, vec: torch.Tensor):
        K = self.K
        meter_logits = vec[:, :K]                          # [B, K]
        u, v = vec[:, K], vec[:, K + 1]                    # [B]
        phase_mu = torch.atan2(v, u) % TWO_PI              # [B]
        phase_rho = torch.sigmoid(vec[:, K + 2]) * (1.0 - 1e-4)   # [B] wrapped-Cauchy concentration
        level_mu = vec[:, K + 3]                           # [B]
        level_sigma = F.softplus(vec[:, K + 4]) + 1e-3     # [B]
        dev_mu = vec[:, K + 5]                             # [B]
        dev_sigma = F.softplus(vec[:, K + 6]) + 1e-3       # [B]
        return meter_logits, phase_mu, phase_rho, level_mu, level_sigma, dev_mu, dev_sigma

    def z_features(self, meter_soft, phi, log_tempo):
        # meter_soft [B,K], phi [B], log_tempo [B] -> [B, z_feat_dim]
        log_tempo = log_tempo.clamp(-12.0, 6.0)   # NaN/inf guard on the deterministic feature use
        return torch.cat([torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1),
                          log_tempo.unsqueeze(-1), meter_soft], dim=-1)

    # ---- prior heads that read audio context (concentrations / scales / AR coef) ----
    def prior_phase_conc(self, ctx_t):
        return torch.sigmoid(self.prior_phase_rho(ctx_t).squeeze(-1)) * (1.0 - 1e-4)

    def prior_level_scale(self, ctx_t):
        return F.softplus(self.prior_level_sigma(ctx_t).squeeze(-1)) + 1e-3

    def prior_dev_scale(self, ctx_t):
        return F.softplus(self.prior_dev_sigma(ctx_t).squeeze(-1)) + 1e-3

    def prior_dev_coef(self, ctx_t):
        return torch.tanh(self.prior_dev_ar(ctx_t).squeeze(-1)) * self.dev_ar_clamp

    # ---- prior meter transition (returns log prior over m_t), batched ----
    def meter_prior_logp(self, meter_prev, phi_t, phi_prev, prior_ctx_t):
        feats = torch.cat([meter_prev,
                           torch.cos(phi_t).unsqueeze(-1), torch.sin(phi_t).unsqueeze(-1),
                           torch.cos(phi_prev).unsqueeze(-1), torch.sin(phi_prev).unsqueeze(-1),
                           prior_ctx_t], dim=-1)                       # [B, K+4+hidden]
        Pi = F.softmax(self.meter_prior(feats).reshape(-1, self.K, self.K), dim=2)  # rows: from-meter
        pi_p = torch.bmm(meter_prev.unsqueeze(1), Pi).squeeze(1)       # [B, K]
        return torch.log(pi_p + 1e-9)

    def decode(self, z_feat, prior_ctx_t):
        """Likelihood p_theta(b_t | z_t): [B,2] logits (beat, downbeat), LATENT-ONLY by design.
        Audio h enters the model ONLY through the prior (concentrations/scales/meter transition),
        never the emission -- an h-reading likelihood is a shortcut that collapses the latent.
        `latent_only=False` (decoder reads prior_ctx) is kept ONLY as a collapse ablation."""
        x = z_feat if self.latent_only else torch.cat([z_feat, prior_ctx_t], dim=-1)
        return self.decoder(x)                                        # [B, 2]
