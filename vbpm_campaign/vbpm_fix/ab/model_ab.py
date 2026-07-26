"""VARIANT A+B : audio-conditioned prior mean (A) + observation decoder p(o_t|z_t) (B).

Subclass of vbpm.model.BarPointerVAE.  NOTHING in vbpm/ is modified.

Root cause being repaired (established, not re-litigated):
    vbpm/elbo.py:free_run rolls  mu^p_phi = phi_{t-1} + phidot_{t-1}  with NO h.
    Audio touches only concentrations/scales/meter-transition, so the deploy path is an
    open-loop metronome (shift test 0.0099 rad).

  (A) AUDIO-CONDITIONED PRIOR MEAN.  A tanh-bounded correction read from the generative
      context prior_ctx_t (plus the current pointer state) is ADDED to the phase prior mean
      and to the log-tempo level prior mean:

          mu^p_phi   = phi_{t-1} + phidot_{t-1} + c_phi * tanh(g_phi(ctx_t, adv, logtempo))
          mu^p_level = anchor + a_lv (level_{t-1} - anchor) + c_lv * tanh(g_lv(...))

      Identical in the ELBO and in the deploy roll-out (same method, same call site).
      Trained purely by the ELBO: KL(q||p) is minimised when the prior mean equals the
      posterior mean, so g_phi learns "given the audio and the previous pointer, where is
      the pointer now" -- a learned observation update inside the generative model.
      Both heads are ZERO-INITIALISED so the model starts *exactly* at the baseline.

  (B) OBSERVATION DECODER p(o_t | z_t).  The model gains an observable at deploy:
          o_t = BatchNorm( W h_t )   in R^{obs_dim}
      a low-dim, batch-normalised projection of the audio, and a Gaussian likelihood
      p(o_t|z_t) = N(mu_theta(z_feat_t), diag sigma_theta(z_feat_t)^2) trained as an extra
      reconstruction term of the SAME ELBO.
      The BatchNorm (affine=False) is the anti-collapse constraint: without it the ELBO
      could trivially make o_t constant (perfectly predictable, zero information); BN forces
      every channel to unit variance across (batch, time) so o_t cannot be time-constant.
      At deploy this is the particle-filter weight -- the "observation likelihood switched
      back on" that madmom's DBN has and VBPM lacked.

  (TEMPO INIT FIX) log-tempo is re-centred at LOG_TEMPO_CENTER = -2.7 (corpus mean of
      log phidot, bar-radians per frame at fps=50).  Baseline unpack() emits level_mu ~ 0
      => phidot ~ 1 rad/frame, ~14x too fast, which aliases the 2pi wrap.  The offset is
      added to level_mu in unpack(), i.e. to BOTH q and p identically -- a pure
      reparameterisation of the head's output bias, no train/deploy asymmetry.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vbpm.model import BarPointerVAE

# mean of log(phidot): bar = m beats, IBI ~ 0.6 s, fps = 50
#   phidot = 2*pi / (m * IBI * fps) ~ 2*pi/120 = 0.052  ->  log = -2.95 ; corpus mean -2.66
LOG_TEMPO_CENTER = -2.7


class BarPointerVAE_AB(BarPointerVAE):
    def __init__(self, h_dim: int, hidden: int = 128, num_meters: int = 4,
                 obs_dim: int = 16, max_phase_corr: float = 0.30,
                 max_level_corr: float = 0.02, tempo_center: float = LOG_TEMPO_CENTER,
                 **kw):
        super().__init__(h_dim=h_dim, hidden=hidden, num_meters=num_meters, **kw)
        self.obs_dim = obs_dim
        self.max_phase_corr = float(max_phase_corr)
        self.max_level_corr = float(max_level_corr)
        self.tempo_center = float(tempo_center)

        # ---- (A) audio-conditioned prior-mean corrections ----------------------------
        # input: [prior_ctx_t (hidden), cos(adv), sin(adv), log_tempo_prev]
        self.phase_corr = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.level_corr = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        for head in (self.phase_corr, self.level_corr):      # start == baseline exactly
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

        # ---- (B) observation channel -------------------------------------------------
        self.obs_proj = nn.Linear(h_dim, obs_dim)
        self.obs_bn = nn.BatchNorm1d(obs_dim, affine=False)   # anti-collapse: unit var/channel
        self.obs_dec = nn.Sequential(
            nn.Linear(self.z_feat_dim, hidden), nn.Tanh(), nn.Linear(hidden, 2 * obs_dim))

    # -------------------------------------------------------------------------------
    # tempo-init fix: re-centre the log-tempo latent (applied to q AND p -> no asymmetry)
    # -------------------------------------------------------------------------------
    def unpack(self, vec: torch.Tensor):
        (meter_logits, phase_mu, phase_rho, level_mu, level_sigma,
         dev_mu, dev_sigma) = super().unpack(vec)
        return (meter_logits, phase_mu, phase_rho, level_mu + self.tempo_center,
                level_sigma, dev_mu, dev_sigma)

    # -------------------------------------------------------------------------------
    # (A) the corrections.  ONE implementation, called from both the ELBO and deploy.
    # -------------------------------------------------------------------------------
    def _corr_feats(self, ctx_t, advance, log_tempo_prev):
        return torch.cat([ctx_t,
                          torch.cos(advance).unsqueeze(-1),
                          torch.sin(advance).unsqueeze(-1),
                          log_tempo_prev.clamp(-12.0, 6.0).unsqueeze(-1)], dim=-1)

    def phase_correction(self, ctx_t, advance, log_tempo_prev):
        f = self._corr_feats(ctx_t, advance, log_tempo_prev)
        return self.max_phase_corr * torch.tanh(self.phase_corr(f).squeeze(-1))

    def level_correction(self, ctx_t, advance, log_tempo_prev):
        f = self._corr_feats(ctx_t, advance, log_tempo_prev)
        return self.max_level_corr * torch.tanh(self.level_corr(f).squeeze(-1))

    # -------------------------------------------------------------------------------
    # (B) observation channel
    # -------------------------------------------------------------------------------
    def observe(self, h: torch.Tensor) -> torch.Tensor:
        """h [B,T,h_dim] -> o [B,T,obs_dim], batch-normalised (anti-collapse)."""
        B, T, _ = h.shape
        o = self.obs_proj(h).reshape(B * T, self.obs_dim)
        o = self.obs_bn(o)
        return o.reshape(B, T, self.obs_dim)

    def obs_logprob(self, z_feat: torch.Tensor, o_t: torch.Tensor) -> torch.Tensor:
        """log N(o_t; mu_theta(z_feat), sigma_theta(z_feat)). z_feat [N,zf], o_t [N,d] or [1,d]."""
        out = self.obs_dec(z_feat)
        mu, log_sig = out[..., :self.obs_dim], out[..., self.obs_dim:]
        log_sig = log_sig.clamp(-4.0, 3.0)
        sig = torch.exp(log_sig)
        z = (o_t - mu) / sig
        return (-0.5 * z ** 2 - log_sig - 0.5 * math.log(2 * math.pi)).sum(-1)
