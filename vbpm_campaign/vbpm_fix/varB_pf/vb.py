"""VARIANT B -- latent generates the OBSERVATION p_theta(h_t|z_t), deploy = PARTICLE FILTER.

Nothing in vbpm/ is touched.  BarPointerVAE is SUBCLASSED; the ELBO is a copy of
vbpm/elbo.py::strict_elbo with one extra term.

ROOT CAUSE (given, not re-litigated): vbpm/elbo.py::free_run is an open-loop metronome.
The prior recursion mean is (phi_{t-1} + tempo_{t-1}) with no h anywhere, and the ONLY
observed variable in the likelihood is b -- the prediction target, absent at deploy.  So
free_run draws (phase, tempo) once from prior_init_head and rolls forward audio-blind.

WHAT THIS FILE CHANGES
  1. p_theta(h_t | z_t): an observation decoder.  Now there IS an observed variable at
     deploy, so real Bayesian filtering is possible (this is exactly what makes madmom's
     DBN forward/Viterbi work: p(activation | phase) reweights the state every frame).
       * DIRAC h : 2 Bernoulli channels (beat impulse, downbeat impulse).
       * MERT  h : full 768-d Gaussian with a LEARNED PER-DIM scale.  Dims predictable
         from metrical position get a small sigma and dominate the particle weights;
         unpredictable dims get a large sigma and cancel in the weight normalisation.
         h is LayerNorm'd (fixed, non-affine) so the target is bounded and cannot be
         collapsed to a constant (the layer merge is a simplex-constrained convex
         combination of FROZEN MERT layers, so h is data, not a free parameter).
  2. FOURIER PHASE FEATURES for both emissions.  A Tanh MLP on the raw (cos phi, sin phi)
     pair cannot express a ~1-frame-wide impulse comb (a bar is ~90 frames at 130 BPM);
     that is exactly what vbpm/probe_stages.py::S3 flags.  Both p(b|z) and p(h|z) get
     [cos k*phi, sin k*phi]_{k=1..n_harm} plus the meter one-hot.
     log_tempo is DELIBERATELY REMOVED from both emissions: it is the known tempo
     side-channel, and at deploy it would let physically-absurd-tempo particles win the
     weight race by Morse-coding event times instead of tracking phase.
     `decode()` is overridden, so vbpm.elbo.free_run runs unchanged on this subclass and
     gives a like-for-like OPEN-LOOP baseline for the SAME trained weights.
  3. INIT FIXES (initialisations only -- all remain free learned parameters):
       level_mu bias  -> -2.66 = log(0.070 rad/frame) ~ 130 BPM at m=4, fps=50.
         (was 0.0 => exp(0)=1.0 rad/frame, ~14x too fast, which ALIASES the wrap:
          the advance exceeds pi per frame so the phase read-out is meaningless.)
       prior level/dev sigma bias -> softplus(-4)=0.018  (was softplus(0)=0.69 nats of
         log-tempo innovation PER FRAME: an OU walk with stationary std 1.4).
       prior phase rho bias -> sigmoid(3.5)=0.97 => wrapped-Cauchy gamma=0.030 rad
         (was sigmoid(0)=0.5 => gamma=0.69 rad of phase jitter per frame, 11% of a bar).
  4. DEPLOY = bootstrap particle filter (below).
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
from vbpm.elbo import _stationary_dev_sigma

LOG_TEMPO_INIT = -2.66
_LOG2PI = math.log(TWO_PI)


class BarPointerVAE_B(BarPointerVAE):
    def __init__(self, h_dim: int, hidden: int = 128, num_meters: int = 4,
                 obs_mode: str = "bern", obs_dim: int | None = None, n_harm: int = 64,
                 tempo_init: float = LOG_TEMPO_INIT, sigma_bias: float = -4.0,
                 rho_bias: float = 3.5):
        super().__init__(h_dim=h_dim, hidden=hidden, num_meters=num_meters,
                         latent_only=True)
        self.obs_mode = obs_mode
        self.n_harm = n_harm
        self.obs_dim = obs_dim if obs_dim is not None else (2 if obs_mode == "bern" else h_dim)
        feat_in = 2 * n_harm + num_meters                    # NO log_tempo

        # p(b|z): rebuilt on the harmonic basis (still LATENT-ONLY)
        self.decoder = nn.Sequential(
            nn.Linear(feat_in, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        # p(h|z): the new observation decoder
        self.h_dec = nn.Sequential(
            nn.Linear(feat_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, self.obs_dim))
        if obs_mode == "gauss":
            self.h_logscale = nn.Parameter(torch.zeros(self.obs_dim))

        K = num_meters
        with torch.no_grad():
            for head in (self.prior_init_head, self.post_head):
                head[-1].bias[K + 3] = tempo_init
            self.prior_level_sigma.bias.fill_(sigma_bias)
            self.prior_dev_sigma.bias.fill_(sigma_bias)
            self.prior_phase_rho.bias.fill_(rho_bias)
            self.post_head[-1].bias[K + 4] = sigma_bias
            self.post_head[-1].bias[K + 6] = sigma_bias

    # ---- harmonic phase basis, derived from z_feat so free_run works unchanged ----
    def harm(self, z_feat: torch.Tensor) -> torch.Tensor:
        phi = torch.atan2(z_feat[..., 1], z_feat[..., 0])
        k = torch.arange(1, self.n_harm + 1, device=phi.device, dtype=phi.dtype)
        a = phi.unsqueeze(-1) * k
        return torch.cat([torch.cos(a), torch.sin(a), z_feat[..., 3:]], dim=-1)

    def decode(self, z_feat, prior_ctx_t=None):
        return self.decoder(self.harm(z_feat))

    def obs_logp(self, z_feat: torch.Tensor, o_t: torch.Tensor) -> torch.Tensor:
        """log p_theta(o_t | z_t) summed over observation channels."""
        pred = self.h_dec(self.harm(z_feat))
        if self.obs_mode == "bern":
            return -F.binary_cross_entropy_with_logits(
                pred, o_t.expand_as(pred), reduction="none").sum(-1)
        s = F.softplus(self.h_logscale) + 1e-3
        d = (o_t.expand_as(pred) - pred) / s
        return (-0.5 * d * d - torch.log(s) - 0.5 * _LOG2PI).sum(-1)


class MertFront(nn.Module):
    """Learnable softmax over the 13 frozen MERT layers -> [B,T,768], then a FIXED
    (non-affine) LayerNorm: bounded, non-degenerate Gaussian observation target."""

    def __init__(self, n_layers: int = 13, dim: int = 768):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, feats):
        w = torch.softmax(self.layer_logits, 0)
        return self.norm(torch.einsum("l,bltf->btf", w, feats))

    def weights(self):
        return torch.softmax(self.layer_logits.detach(), 0).cpu().numpy()


# ===========================================================================
# ELBO = copy of vbpm/elbo.py::strict_elbo  +  lam_h * sum_t log p(h_t | z_t)
# ===========================================================================
def elbo_b(model, h, b, db, obs, temperature=0.5, beta=1.0, lam_h=1.0):
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
    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _u, _v = model.unpack(
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

    Z = torch.stack(z_feats, dim=1)                                    # [B,T,zf]
    logits = model.decode(Z)                                           # [B,T,2]
    recon_b = F.binary_cross_entropy_with_logits(logits[..., 0], b, reduction="none").sum(1)
    recon_db = F.binary_cross_entropy_with_logits(logits[..., 1], db, reduction="none").sum(1)
    obs_lp = model.obs_logp(Z, obs).sum(1)                             # [B]

    L_kl = kl_m + kl_p + kl_lv + kl_dv
    loss = (recon_b + recon_db - lam_h * obs_lp + beta * L_kl).mean()
    info = {"loss": float(loss), "recon_beat": float(recon_b.mean()),
            "recon_db": float(recon_db.mean()), "obs_lp": float(obs_lp.mean()),
            "kl_phase": float(kl_p.mean()), "kl_level": float(kl_lv.mean()),
            "kl_dev": float(kl_dv.mean()), "kl_meter": float(kl_m.mean()),
            "n_cross": float(n_cross.mean()), "beta": float(beta)}
    return loss, info


# ===========================================================================
# DEPLOY: bootstrap particle filter
# ===========================================================================
def _student_t(dof: float, loc, scale):
    g = torch._standard_gamma(torch.full_like(loc, 0.5 * dof))
    return loc + scale * torch.randn_like(loc) / torch.sqrt(2.0 * g / dof)


def _systematic_resample(w: torch.Tensor) -> torch.Tensor:
    K = w.shape[0]
    u = (torch.rand(1, device=w.device) + torch.arange(K, device=w.device)) / K
    return torch.searchsorted(torch.cumsum(w, 0).contiguous(), u.contiguous()).clamp(max=K - 1)


@torch.no_grad()
def particle_filter(model, h, obs, K=500, alpha=1.0, diffuse_init=True,
                    tempo_spread=0.35, ess_frac=0.5):
    """Bootstrap PF on the model's OWN prior transition, weighted by p_theta(h_t|z_t).

    h [1,T,D] (drives the prior heads), obs [1,T,obs_dim] (the observed variable).
    `alpha` tempers the observation log-lik (w ~ p(h|z)^alpha) -- needed for the 768-d
    Gaussian, whose per-frame log-lik spread otherwise collapses the particle set every
    frame.  Returns the three phase read-outs + diagnostics.
    """
    assert h.shape[0] == 1
    T = h.shape[1]
    dv = h.device
    prior_ctx = model.encode_prior(h)
    dof = float(model.tempo_dof())
    a_lv = model.level_ar()

    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _u, _v = model.unpack(
        model.prior_init_head(prior_ctx.mean(1)))
    a0 = model.prior_dev_coef(prior_ctx[:, 0])
    sd0 = model.prior_dev_scale(prior_ctx[:, 0])
    dvs0 = _stationary_dev_sigma(sd0, a0).expand(K)

    if diffuse_init:
        phi = torch.rand(K, device=dv) * TWO_PI
        level = p_lv_mu.expand(K) + tempo_spread * torch.randn(K, device=dv)
    else:
        phi = sample_wrapped_cauchy(p_ph_mu.expand(K), p_ph_rho.expand(K))
        level = _student_t(dof, p_lv_mu.expand(K), p_lv_s.expand(K))
    dev_ = dvs0 * torch.randn(K, device=dv)
    meter = F.one_hot(torch.multinomial(torch.softmax(p_m, -1).expand(K, -1), 1).squeeze(-1),
                      model.K).to(h.dtype)
    log_tempo = level + dev_
    anchor = level.clone()

    zf = model.z_features(meter, phi, log_tempo)
    logw = alpha * model.obs_logp(zf, obs[0, 0].unsqueeze(0))
    logw = logw - torch.logsumexp(logw, 0)

    phi_h = torch.zeros(T, K, device=dv); lt_h = torch.zeros(T, K, device=dv)
    w_h = torch.zeros(T, K, device=dv)
    anc = torch.zeros(T, K, dtype=torch.long, device=dv)
    phi_h[0] = phi; lt_h[0] = log_tempo; w_h[0] = logw.exp()
    anc[0] = torch.arange(K, device=dv)
    n_res = 0; ess_sum = 0.0

    for t in range(1, T):
        w = torch.softmax(logw, 0)
        ess = float(1.0 / (w * w).sum()); ess_sum += ess
        if ess < ess_frac * K:
            idx = _systematic_resample(w); n_res += 1
            logw = torch.full((K,), -math.log(K), device=dv)
        else:
            idx = torch.arange(K, device=dv)
        anc[t] = idx
        phi_p = phi[idx]; meter_p = meter[idx]; lev_p = level[idx]
        dev_p = dev_[idx]; lt_p = log_tempo[idx]; anchor = anchor[idx]

        ctx = prior_ctx[0, t].unsqueeze(0).expand(K, -1)
        advance = phi_p + torch.exp(lt_p.clamp(-12.0, 6.0))
        cross = advance >= TWO_PI
        rho = model.prior_phase_conc(ctx)
        a = model.prior_dev_coef(ctx)
        s_lv = model.prior_level_scale(ctx)
        s_dv = model.prior_dev_scale(ctx)

        phi = sample_wrapped_cauchy(advance % TWO_PI, rho)
        level = _student_t(dof, anchor + a_lv * (lev_p - anchor), s_lv)
        dev_ = a * dev_p + s_dv * torch.randn(K, device=dv)
        log_tempo = level + dev_
        lp = model.meter_prior_logp(meter_p, phi, phi_p, ctx)
        draw = F.one_hot(torch.multinomial(lp.exp(), 1).squeeze(-1), model.K).to(h.dtype)
        meter = torch.where(cross.unsqueeze(-1), draw, meter_p)

        zf = model.z_features(meter, phi, log_tempo)
        logw = logw + alpha * model.obs_logp(zf, obs[0, t].unsqueeze(0))
        logw = logw - torch.logsumexp(logw, 0)

        phi_h[t] = phi; lt_h[t] = log_tempo; w_h[t] = torch.softmax(logw, 0)

    circ = torch.atan2((w_h * torch.sin(phi_h)).sum(1),
                       (w_h * torch.cos(phi_h)).sum(1)) % TWO_PI
    midx = w_h.argmax(1)
    mapphi = phi_h[torch.arange(T, device=dv), midx]

    j = int(w_h[T - 1].argmax())
    anc_phi = torch.zeros(T, device=dv); anc_lt = torch.zeros(T, device=dv)
    for t in range(T - 1, -1, -1):
        anc_phi[t] = phi_h[t, j]; anc_lt[t] = lt_h[t, j]
        if t > 0:
            j = int(anc[t, j])
    return {"circ": circ.cpu().numpy(), "map": mapphi.cpu().numpy(),
            "anc": anc_phi.cpu().numpy(), "anc_log_tempo": anc_lt.cpu().numpy(),
            "n_resample": n_res, "mean_ess": ess_sum / max(T - 1, 1)}
