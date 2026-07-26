#!/usr/bin/env python
"""PHYSICAL-PRIOR ANCHORING  (professor's tutorial S6.8.11)   ---   VBPM_reintegration/vbpm_anchor/anchor.py

    L_reg-EB  =  ELBO(theta, phi_enc, psi)  -  lambda_prior * KL( p_psi(z|x) || p_physical(z) )

Implements SPEC.md (vbpm_anchor/SPEC.md).  Nothing in vbpm/, vbpm_fix/, vbpm_arms/, vbpm_final/
is modified -- this file SUBCLASSES / COPIES.

WHAT IS WHERE
  * ELBO            : verbatim copy of vbpm_arms/arm_ii.py's training objective, i.e.
                      vbpm_fix/variant_b.py::elbo_b, PLUS the anchor accumulator (elbo_anchor).
  * emission        : the SUPERVISED, FROZEN PhaseEmission of vbpm_final/emission.py.
                      - at DEPLOY it is ALWAYS the weighting function (obs_logp is
                        swap_test.SupEmisModel's table lookup) -> emission held identical across
                        every cell, so the transition is the only manipulated variable;
                      - in the ELBO the obs term is selectable with --elbo_obs:
                          vae  (default) = arm_ii's learned h_dec  -> makes --lambda_prior 0 the
                                           exact archived R1 cell (0.592) regression;
                          sup            = the frozen supervised table, SOFT-binned so gradients
                                           still reach phi (spec arm B);
                          none           = no observation term.
  * transition      : --transition learned  = p_psi (the thing under test)
                      --transition hand     = vbpm_final/torch_pf.py::simple_pf (the 0.751 cell)
  * deploy PF       : learned -> vbpm_final/e3_pf_learned.py::particle_filter_learned
                      (ancestral backtrace; SPEC S6.2 harness fix #1).  --pf_impl legacy uses
                      vbpm_fix/variant_b.py::particle_filter (phase_path == phase_map) to
                      reproduce the archived 2x2 numbers bit-for-bit.
  * scoring         : run_exp2.score_traj / summarize / pr  -- i.e. blind_grid_controls VERBATIM.

THE ANCHORED-RESIDUAL LINKS (SPEC S3.2, mandatory: the sigmoid/softplus links cannot travel
12.1 / 8.5 pre-activation units in 1200 steps, so a naive anchor is a link-function null)
    gamma_psi(ctx) = gamma_phy * exp(v_max * tanh(g_head(ctx)))     rho = exp(-gamma)
    s_lv(ctx)      = s_lv_phy  * exp(v_max * tanh(l_head(ctx)))
    s_dv(ctx)      = s_dv_phy  * exp(v_max * tanh(d_head(ctx)))
    log pi_psi     = log_softmax( log pi_phy + v_max * tanh(m_head(ctx)) )
with every head's FINAL LAYER ZERO-INITIALISED -> training starts exactly AT p_physical and the
ELBO must pay to leave.  lambda_prior = inf is implemented as --freeze_residual (residual pinned
at 0, i.e. p_psi == p_physical exactly): the decisive "is the learned part decorative?" control.
NOTE the residual is a BOX of +/- exp(v_max) around physics; with the default v_max=3 (+/-20x) the
link CANNOT represent the archived unanchored solution (gamma 4.62 = 8300x physics).  So
`--lambda_prior 0 --links residual` is a *box-constrained* control, not the un-anchored model;
use `--links orig` (or a large --v_max, e.g. 9) for the genuine un-anchored control.  Both are
provided precisely so this confound stays visible.

DEVIATION FROM SPEC S1.3, declared: the level/dev anchors are DISPERSION-ONLY by default
(mu_psi and mu_phy set equal, exactly as happens identically for the phase factor whose means
are both phi_{t-1}+phidot_{t-1}).  Reason: with s_lv_phy = 1.25e-3 the mean term
(mu_psi-mu_phy)^2 / (2 s_phy^2) is O(1e3-1e5) nats at initialisation and would swamp every other
term -- the very cliff S3.2 exists to avoid.  SPEC S5(iii) argues the dispersions ARE the whole
intervention.  --anchor_mean turns the (specced) mean terms on.

USAGE
  regression R1 (un-anchored learned transition + supervised emission, archived 0.592):
    python anchor.py --steps 0 --links orig --init_ckpt <arm_i_ii_bern.pt> --pf_impl legacy \
                     --transition learned --K 300 --alpha 1.0
  regression R2 (hand transition + supervised emission, archived 0.725/0.765):
    python anchor.py --steps 0 --transition hand --K 600 --alpha 0.25
  the experiment:
    python anchor.py --lambda_prior 0.1 --steps 1200 --tag lam0.1
  pre-flight gradient/link audit (SPEC S4.3 / S5(ii)):
    python anchor.py --check
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time

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

import variant_b as VB                                                          # noqa: E402
from vbpm.distributions import (                                                # noqa: E402
    TWO_PI, gumbel_softmax, sample_wrapped_cauchy, sample_student_t,
    kl_categorical, kl_log_normal, kl_student_t_mc, kl_wrapped_cauchy,
)
from emission import (PhaseEmission, load_act, load_split, obs_contrast,        # noqa: E402
                      METERS, FPS, _estimate_meter)
from swap_test import SupEmisModel                                              # noqa: E402
from run_exp2 import score_traj, summarize, pr                                  # noqa: E402
from e3_pf_learned import particle_filter_learned                               # noqa: E402
from torch_pf import simple_pf                                                  # noqa: E402
from common import targets                                                      # noqa: E402

HERE = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_anchor"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
DEV = "cuda:0"

# ---------------------------------------------------------------------------------------------
# p_physical  (SPEC S2).  P-FIT = fitted on the TRAIN fold by vbpm_anchor/fit_phys.py (labels
# only, never eval).  P-DEPLOY = the hand transition that scores 0.751 (pf.py defaults).
# ---------------------------------------------------------------------------------------------
_PHYS_JSON = json.load(open(f"{HERE}/phys_params.json"))
PHYS = {
    "fit": dict(gamma_phy=float(_PHYS_JSON["gamma_cauchy"]),      # 5.5497e-4  (rho 0.999445)
                s_lv_phy=float(_PHYS_JSON["sigma_dlt"]),          # 1.2477e-3
                s_dv_phy=1e-3,
                mu_lt={int(k): float(v) for k, v in _PHYS_JSON["lt_mean"].items()},
                a_phy=0.999),
    "deploy": dict(gamma_phy=0.02,                                # rho 0.9802 ~ sigma_phi 0.03
                   s_lv_phy=0.05,                                 # pf.py --sigma_lt
                   s_dv_phy=1e-3,
                   mu_lt={int(k): float(v) for k, v in _PHYS_JSON["lt_mean"].items()},
                   a_phy=0.999),
}
GAMMA_MIN = 1e-5          # float32 hygiene: rho = exp(-1e-5) still resolves 1-rho


def wrap(x):
    return (x + math.pi) % TWO_PI - math.pi


def kl_wc_gamma(g_q, g_p):
    """KL( WC(mu, gamma_q) || WC(mu, gamma_p) ) for EQUAL means, in gamma space.

    = log[ (1 - rho_q rho_p)^2 / ((1-rho_q^2)(1-rho_p^2)) ]   with rho = exp(-gamma),
    written with -expm1(-x) = 1-exp(-x) so it is exact for gamma << 1 (rho -> 1), where the
    physical prior lives and the naive float32 form loses all its digits.
    """
    lg = lambda x: torch.log(-torch.expm1(-x))          # noqa: E731  log(1 - e^-x)
    return 2.0 * lg(g_q + g_p) - lg(2.0 * g_q) - lg(2.0 * g_p)


# ---------------------------------------------------------------------------------------------
# the model:  learned prior with the anchored-residual links  (SPEC S3.2 / S4.1)
# ---------------------------------------------------------------------------------------------
class AnchoredVAE(SupEmisModel):
    """BarPointerVAE_B (+ supervised-emission obs_logp) whose PRIOR dispersion heads are
    reparameterised as multiplicative residuals around p_physical.  Same distribution families,
    same architecture, same inputs -- only the link functions and their initialisation change."""

    def __init__(self, hidden=128, num_meters=4, links="residual", v_max=3.0,
                 phys="fit", p_switch=0.005, meter_prior=None, **kw):
        super().__init__(h_dim=2, hidden=hidden, num_meters=num_meters,
                         obs_dim=2, obs_type="bern", **kw)
        assert links in ("residual", "orig")
        self.links = links
        self.v_max = float(v_max)
        self.freeze_residual = False              # lambda_prior = inf  <=>  p_psi == p_physical
        P = PHYS[phys]
        self.phys_name = phys
        self.register_buffer("gamma_phy", torch.tensor(P["gamma_phy"]))
        self.register_buffer("s_lv_phy", torch.tensor(P["s_lv_phy"]))
        self.register_buffer("s_dv_phy", torch.tensor(P["s_dv_phy"]))
        self.register_buffer("a_phy", torch.tensor(P["a_phy"]))
        # per-meter physical log bar-advance level, indexed by CLASS j (meter value j+1)
        mu = torch.full((num_meters,), P["mu_lt"][4])
        for m, v in P["mu_lt"].items():
            if 1 <= m <= num_meters:
                mu[m - 1] = v
        self.register_buffer("mu_lt", mu)
        # physical meter kernel: sticky, switching only at bar crossings (alt_meter_model 0.0052/bar)
        pv = torch.tensor(meter_prior if meter_prior is not None
                          else [0.0] + [1.0 / 3] * (num_meters - 1), dtype=torch.float32)
        pv = pv / pv.sum()
        Pi = (1.0 - p_switch) * torch.eye(num_meters) + p_switch * pv.unsqueeze(0)
        self.register_buffer("log_pi_phy_mat", torch.log(Pi + 1e-9))
        if links == "residual":                   # start EXACTLY at physics
            for lin in (self.prior_phase_rho, self.prior_level_sigma, self.prior_dev_sigma):
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
            nn.init.zeros_(self.meter_prior[-1].weight); nn.init.zeros_(self.meter_prior[-1].bias)
        self._sat = []                            # |tanh| saturation monitor

    # ---- residual helper -------------------------------------------------------------------
    def _res(self, head, ctx_t):
        v = self.v_max * torch.tanh(head(ctx_t).squeeze(-1))
        if self.freeze_residual:
            v = torch.zeros_like(v)
        if self.training:
            self._sat.append(float((v.abs() > 0.99 * self.v_max).float().mean()))
        return v

    # ---- the four prior heads --------------------------------------------------------------
    def prior_phase_gamma(self, ctx_t):
        """wrapped-Cauchy scale gamma (rad).  frac_neg floor = (1/pi) atan(gamma/phidot)."""
        if self.links == "orig":
            return -torch.log(super().prior_phase_conc(ctx_t).clamp(min=1e-6, max=1 - 1e-6))
        return (self.gamma_phy * torch.exp(self._res(self.prior_phase_rho, ctx_t))
                ).clamp(min=GAMMA_MIN)

    def prior_phase_conc(self, ctx_t):
        if self.links == "orig":
            return super().prior_phase_conc(ctx_t)
        return torch.exp(-self.prior_phase_gamma(ctx_t)).clamp(max=1.0 - 1e-6)

    def prior_level_scale(self, ctx_t):
        if self.links == "orig":
            return super().prior_level_scale(ctx_t)
        return self.s_lv_phy * torch.exp(self._res(self.prior_level_sigma, ctx_t))

    def prior_dev_scale(self, ctx_t):
        if self.links == "orig":
            return super().prior_dev_scale(ctx_t)
        return self.s_dv_phy * torch.exp(self._res(self.prior_dev_sigma, ctx_t))

    def meter_prior_pair(self, meter_prev, phi_t, phi_prev, ctx_t):
        """-> (log pi_psi [B,K], log pi_phy [B,K]).  Row-mixed through the (soft) meter_prev."""
        log_pi_phy = torch.log(meter_prev @ self.log_pi_phy_mat.exp() + 1e-9)
        if self.links == "orig":
            return super().meter_prior_logp(meter_prev, phi_t, phi_prev, ctx_t), log_pi_phy
        feats = torch.cat([meter_prev,
                           torch.cos(phi_t).unsqueeze(-1), torch.sin(phi_t).unsqueeze(-1),
                           torch.cos(phi_prev).unsqueeze(-1), torch.sin(phi_prev).unsqueeze(-1),
                           ctx_t], dim=-1)
        R = self.v_max * torch.tanh(self.meter_prior(feats).reshape(-1, self.K, self.K))
        if self.freeze_residual:
            R = torch.zeros_like(R)
        r = torch.bmm(meter_prev.unsqueeze(1), R).squeeze(1)                  # [B,K]
        return torch.log_softmax(log_pi_phy + r, dim=-1), log_pi_phy

    def meter_prior_logp(self, meter_prev, phi_t, phi_prev, ctx_t):
        return self.meter_prior_pair(meter_prev, phi_t, phi_prev, ctx_t)[0]

    # ---- differentiable (soft-binned) supervised emission, for --elbo_obs sup ---------------
    def sup_logp_soft(self, z_feat, o_t):
        """log p_sup(o|phi,m) with LINEAR interpolation over phase bins and the SOFT meter
        mixture, so gradients reach phi and the meter (the hard-binned obs_logp used at deploy
        has zero gradient a.e. and would silently kill the phase gradient)."""
        phi = torch.atan2(z_feat[:, 1], z_feat[:, 0]) % TWO_PI
        mw = z_feat[:, 3:]                                                     # [N,K] soft meter
        v = o_t if self.emis.likelihood == "bern" else torch.log(o_t / (1 - o_t))
        ll = 0.0
        for j in range(self.K):
            nb = self.nb_t[j]
            x = phi / TWO_PI * nb
            b0 = torch.floor(x).long() % nb
            b1 = (b0 + 1) % nb
            fr = (x - torch.floor(x)).unsqueeze(-1)
            def tab(b):
                out = (self.W[j, b] * v).sum(-1) + self.C0[j, b]
                if self.emis.likelihood == "gauss":
                    out = out + (self.W2[j, b] * v ** 2).sum(-1)
                return out
            ll = ll + mw[:, j] * ((1 - fr.squeeze(-1)) * tab(b0) + fr.squeeze(-1) * tab(b1))
        return ll


# ---------------------------------------------------------------------------------------------
# ELBO + anchor   (verbatim copy of vbpm_fix/variant_b.py::elbo_b, additions marked ANCHOR)
# ---------------------------------------------------------------------------------------------
def _stationary_dev_sigma(sigma, a):
    return sigma / torch.sqrt((1.0 - a ** 2).clamp(min=1e-3))


def elbo_anchor(model, h, b, db, obs, temperature=0.5, beta=1.0, obs_w=1.0,
                lam=0.0, elbo_obs="vae", anchor_mean=False, diag=True):
    """L = -[log p(b|z) + log p(db|z) + obs_w log p(o|z)] + beta*sum KL + lam*KL(p_psi||p_phy)."""
    B, T, _ = h.shape
    post_ctx = model.encode_posterior(h, b)
    prior_ctx = model.encode_prior(h)
    dof = model.tempo_dof()

    kl_m = h.new_zeros(B); kl_p = h.new_zeros(B)
    kl_lv = h.new_zeros(B); kl_dv = h.new_zeros(B)
    A_ph = h.new_zeros(B); A_lv = h.new_zeros(B)          # ANCHOR accumulators, per factor
    A_dv = h.new_zeros(B); A_m = h.new_zeros(B)
    z_feats = []
    n_cross = h.new_zeros(B)
    neg, tot = 0.0, 0.0                                   # ANCHOR diag: prior's own frac_neg
    g_hist, slv_hist, sdv_hist = [], [], []

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
    # ANCHOR (SPEC S4.2: the t=0 dev scale)
    A_dv = A_dv + kl_log_normal(p_dv_mu.detach(), p_dv_s, p_dv_mu.detach(), model.s_dv_phy)

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
        log_pi_p, log_pi_phy = model.meter_prior_pair(meter_prev, phi, phi_prev, prior_ctx[:, t])

        kl_p = kl_p + kl_wrapped_cauchy(q_ph_mu, q_ph_rho, p_ph_mu, p_ph_rho)
        kl_lv = kl_lv + kl_student_t_mc(dof, q_lv_mu, q_lv_s, p_lv_mu, p_lv_s, level)
        kl_dv = kl_dv + kl_log_normal(q_dv_mu, q_dv_s, p_dv_mu, p_dv_s)
        kl_m = kl_m + cross * kl_categorical(torch.log_softmax(q_m, -1), log_pi_p)
        n_cross = n_cross + cross

        # ---------------- ANCHOR: KL( p_psi(.|z_{t-1},x) || p_phy(.|z_{t-1}) ) ---------------
        # every q-sampled conditioning tensor is DETACHED (SPEC S4.3): the anchor regularises
        # psi only, never the encoder.
        g_psi = model.prior_phase_gamma(prior_ctx[:, t])
        A_ph = A_ph + kl_wc_gamma(g_psi, model.gamma_phy.expand_as(g_psi))   # means identical
        mu_lv_phy = ((meter_prev.detach() * model.mu_lt).sum(-1)
                     + model.a_phy * (level_prev.detach() - (meter_prev.detach()
                                                             * model.mu_lt).sum(-1)))
        mu_q = p_lv_mu if anchor_mean else p_lv_mu.detach()
        mu_p = mu_lv_phy if anchor_mean else p_lv_mu.detach()
        A_lv = A_lv + kl_log_normal(mu_q, p_lv_s, mu_p, model.s_lv_phy)
        d_q = p_dv_mu if anchor_mean else p_dv_mu.detach()
        d_p = torch.zeros_like(d_q) if anchor_mean else p_dv_mu.detach()
        A_dv = A_dv + kl_log_normal(d_q, p_dv_s, d_p, model.s_dv_phy)
        A_m = A_m + cross * kl_categorical(log_pi_p, log_pi_phy.detach())

        if diag:                                    # monotonicity watch: the PRIOR's own steps
            with torch.no_grad():
                phi_pr = sample_wrapped_cauchy(p_ph_mu.detach(), p_ph_rho.detach())
                inc = (phi_pr - phi_prev.detach() + math.pi) % TWO_PI - math.pi
                neg += float((inc < 0).float().sum()); tot += float(inc.numel())
                g_hist.append(float(g_psi.mean()))
                slv_hist.append(float(p_lv_s.mean())); sdv_hist.append(float(p_dv_s.mean()))

        z_feats.append(model.z_features(meter, phi, log_tempo))
        meter_prev, phi_prev = meter, phi
        level_prev, dev_prev, log_tempo_prev = level, dev, log_tempo

    Z = torch.stack(z_feats, dim=1)
    logits = model.decoder(Z)
    beat_logits, db_logits = logits[..., 0], logits[..., 1]
    recon_b = F.binary_cross_entropy_with_logits(beat_logits, b, reduction="none").sum(1)
    recon_db = F.binary_cross_entropy_with_logits(db_logits, db, reduction="none").sum(1)

    if elbo_obs == "vae":
        obs_ll = model.h_dec_logp(Z.reshape(B * T, -1), obs.reshape(B * T, -1)).reshape(B, T).sum(1)
    elif elbo_obs == "sup":
        obs_ll = model.sup_logp_soft(Z.reshape(B * T, -1), obs.reshape(B * T, -1)).reshape(B, T).sum(1)
    else:
        obs_ll = torch.zeros_like(recon_b)
    recon_obs = -obs_ll

    L_kl = kl_m + kl_p + kl_lv + kl_dv
    L_anc = A_ph + A_lv + A_dv + A_m
    loss = (recon_b + recon_db + obs_w * recon_obs + beta * L_kl + lam * L_anc).mean()

    sat = float(np.mean(model._sat)) if model._sat else 0.0
    model._sat = []
    info = {"loss": float(loss), "recon_beat": float(recon_b.mean()),
            "recon_db": float(recon_db.mean()), "recon_obs": float(recon_obs.mean()),
            "kl": float(L_kl.mean()), "kl_phase": float(kl_p.mean()),
            "kl_level": float(kl_lv.mean()), "kl_dev": float(kl_dv.mean()),
            "kl_meter": float(kl_m.mean()), "n_cross": float(n_cross.mean()),
            "anchor": float(L_anc.mean()), "anchor_phase": float(A_ph.mean()),
            "anchor_level": float(A_lv.mean()), "anchor_dev": float(A_dv.mean()),
            "anchor_meter": float(A_m.mean()),
            "anchor_phase_pf": float(A_ph.mean()) / max(T - 1, 1),
            "frac_neg_prior": (neg / tot) if tot else float("nan"),
            "gamma_psi": float(np.mean(g_hist)) if g_hist else float("nan"),
            "s_lv": float(np.mean(slv_hist)) if slv_hist else float("nan"),
            "s_dv": float(np.mean(sdv_hist)) if sdv_hist else float("nan"),
            "sat_frac": sat, "tempo_dof": float(dof.detach())}
    return loss, info, (A_ph, A_lv, A_dv, A_m)


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def sample_batch(songs, acts, rng, bs, frames, dev):
    """crops of the frozen 2-ch activation (= h AND the observation, ARM (ii)) + beat targets."""
    hh, bb, dd = [], [], []
    while len(hh) < bs:
        s = songs[rng.integers(len(songs))]
        a = acts[s["stem"]]
        T = min(len(a), s["T"])
        if T <= frames:
            continue
        st = int(rng.integers(0, T - frames))
        hh.append(torch.from_numpy(np.asarray(a[st:st + frames], np.float32)))
        b, d = targets(s["beats"], s["downs"], st, frames)
        bb.append(torch.from_numpy(b)); dd.append(torch.from_numpy(d))
    return (torch.stack(hh).to(dev), torch.stack(bb).to(dev), torch.stack(dd).to(dev))


# ---------------------------------------------------------------------------------------------
# deploy = bootstrap particle filter + the MANDATORY controls
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def deploy(model, emis, songs, acts, a, prior_vec, dev):
    rows = {k: [] for k in ("mean", "map", "path", "pf_meter_path")}
    t0 = time.time()
    for i, s in enumerate(songs):
        act = acts.get(s["stem"])
        if act is None:
            continue
        T = min(len(act), s["T"])
        if a.max_frames:
            T = min(T, a.max_frames)
        ref = s["beats"][s["beats"] < T / FPS]
        dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 3:
            continue
        m_gt = _estimate_meter(s["beats"], s["downs"])
        obs = torch.from_numpy(act[:T]).to(dev)
        torch.manual_seed(a.seed + i)
        if a.transition == "hand":
            out = simple_pf(model, obs, K=a.K, alpha=a.alpha, sigma_lt=a.sigma_lt,
                            sigma_phi=a.sigma_phi, p_switch=a.p_switch,
                            meter_prior=prior_vec, seed=a.seed + i)
            mp = out["meter_path"]
        elif a.pf_impl == "legacy":
            o = obs.unsqueeze(0)
            r = VB.particle_filter(model, o, o, K=a.K, alpha=a.alpha)
            out = dict(phase_mean=r["phase_mean"].numpy(), phase_map=r["phase_map"].numpy(),
                       phase_path=r["phase_map"].numpy(), ess=r["ess"])
            mp = np.asarray(r["meter_map"]) + 1
        else:
            o = obs.unsqueeze(0)
            out = particle_filter_learned(model, o, o, K=a.K, alpha=a.alpha)
            mp = np.asarray(out["meter_path"])
        c, _ = obs_contrast(emis, [s], acts)
        base = dict(stem=s["stem"], n_true=len(ref), n_true_db=len(dref), ess=float(out["ess"]),
                    obs_contrast=float(c) if np.isfinite(c) else float("nan"),
                    meter_ok=float(int(np.bincount(mp).argmax()) == m_gt))
        for k in ("mean", "map", "path"):
            rows[k].append({**base, **score_traj(out[f"phase_{k}"], m_gt, ref, dref, T)})
        rows["pf_meter_path"].append({**base, **score_traj(
            out["phase_path"], int(np.bincount(mp).argmax()), ref, dref, T)})
        if a.verbose and i % 20 == 0:
            print(f"    {i}/{len(songs)}  {time.time()-t0:.0f}s", flush=True)
    return rows


# ---------------------------------------------------------------------------------------------
# pre-flight audits  (SPEC S4.3 gradient hygiene, S5(ii) link reachability)
# ---------------------------------------------------------------------------------------------
def preflight(model, songs, acts, a, dev):
    print("\n=== PRE-FLIGHT (a) gradient hygiene: lam>0, beta=0 -> anchor must not touch q ===")
    rng = np.random.default_rng(0)
    h, b, d = sample_batch(songs, acts, rng, 4, 64, dev)
    model.train()
    model.zero_grad()
    _, _, (A_ph, A_lv, A_dv, A_m) = elbo_anchor(model, h, b, d, h, temperature=0.7, beta=0.0,
                                                obs_w=0.0, lam=1.0, elbo_obs="none",
                                                anchor_mean=a.anchor_mean)
    (A_ph + A_lv + A_dv + A_m).mean().backward()
    def gn(mod):
        return float(sum((p.grad ** 2).sum() for p in mod.parameters() if p.grad is not None) ** 0.5)
    enc = {"post_gru": gn(model.post_gru), "post_ctx": gn(model.post_ctx),
           "post_head": gn(model.post_head), "decoder": gn(model.decoder),
           "h_dec": gn(model.h_dec)}
    psi = {"prior_phase_rho": gn(model.prior_phase_rho),
           "prior_level_sigma": gn(model.prior_level_sigma),
           "prior_dev_sigma": gn(model.prior_dev_sigma),
           "meter_prior": gn(model.meter_prior), "prior_gru": gn(model.prior_gru)}
    for k, v in enc.items():
        print(f"    q-side  {k:16s} |grad|={v:.3e}   {'OK' if v == 0.0 else 'LEAK!'}")
    for k, v in psi.items():
        print(f"    psi-side {k:16s} |grad|={v:.3e}   {'OK' if v > 0 else 'DEAD'}")
    ok_a = all(v == 0.0 for v in enc.values()) and psi["prior_phase_rho"] > 0 \
        and psi["prior_level_sigma"] > 0
    print(f"    -> gradient hygiene {'PASS' if ok_a else 'FAIL'}")
    model.zero_grad()

    print("\n=== PRE-FLIGHT (b) link reachability: can lam move gamma_psi to gamma_phy? ===")
    m2 = copy.deepcopy(model)
    # start the residual OFF physics (v = +v_max*0.9) and see whether lam pulls it back
    with torch.no_grad():
        m2.prior_phase_rho.bias.fill_(3.0)
        m2.prior_level_sigma.bias.fill_(3.0)
    opt = torch.optim.AdamW(m2.parameters(), lr=a.lr)
    g0 = s0 = None
    for st in range(1, a.check_steps + 1):
        h, b, d = sample_batch(songs, acts, rng, 8, 128, dev)
        opt.zero_grad()
        loss, info, _ = elbo_anchor(m2, h, b, d, h, temperature=0.7, beta=0.0, obs_w=0.0,
                                    lam=3.0, elbo_obs="none", anchor_mean=a.anchor_mean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m2.parameters(), 5.0)
        opt.step()
        if g0 is None:
            g0, s0 = info["gamma_psi"], info["s_lv"]
        if st % 20 == 0 or st == a.check_steps:
            print(f"    step {st:4d}  gamma_psi={info['gamma_psi']:.3e} "
                  f"(x{info['gamma_psi']/float(m2.gamma_phy):.2f} phys)  "
                  f"s_lv={info['s_lv']:.3e} (x{info['s_lv']/float(m2.s_lv_phy):.2f})  "
                  f"anchor={info['anchor']:.2f}  frac_neg_prior={info['frac_neg_prior']:.4f}",
                  flush=True)
    r = info["gamma_psi"] / float(m2.gamma_phy)
    print(f"    start x{g0/float(m2.gamma_phy):.2f} -> end x{r:.2f} physical gamma; "
          f"{'PASS (within 2x)' if 0.5 <= r <= 2.0 else 'FAIL (link is the blocker)'}")
    return ok_a, r


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # --- the manipulated variable ---
    ap.add_argument("--lambda_prior", type=float, default=0.0)
    ap.add_argument("--freeze_residual", action="store_true",
                    help="lambda_prior = inf : p_psi == p_physical exactly (decisive control)")
    ap.add_argument("--phys", default="fit", choices=["fit", "deploy"])
    ap.add_argument("--links", default="residual", choices=["residual", "orig"])
    ap.add_argument("--v_max", type=float, default=3.0)
    ap.add_argument("--anchor_mean", action="store_true",
                    help="also anchor the level/dev prior MEANS (SPEC S1.3 literal form)")
    # --- ELBO / training ---
    ap.add_argument("--elbo_obs", default="vae", choices=["vae", "sup", "none"])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--obs_w", type=float, default=1.0)
    ap.add_argument("--init_ckpt", default=None)
    ap.add_argument("--holdout_fold", type=int, default=-1,
                    help="drop this train fold from training (SPEC S3.3 dev split)")
    # --- deploy ---
    ap.add_argument("--transition", default="learned", choices=["learned", "hand"])
    ap.add_argument("--hand_transition", action="store_true", help="alias for --transition hand")
    ap.add_argument("--pf_impl", default="path", choices=["path", "legacy"])
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--sigma_lt", type=float, default=0.05)
    ap.add_argument("--sigma_phi", type=float, default=0.03)
    ap.add_argument("--p_switch", type=float, default=0.005)
    ap.add_argument("--lik", default="gauss", choices=["bern", "gauss"])
    ap.add_argument("--bpb", type=int, default=24)
    ap.add_argument("--eval_on", default="eval", choices=["eval", "holdout", "train"])
    ap.add_argument("--n_eval", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0)
    # --- misc ---
    ap.add_argument("--check", action="store_true", help="run the pre-flight audits and exit")
    ap.add_argument("--check_steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--tag", default="anchor")
    a = ap.parse_args()
    if a.hand_transition:
        a.transition = "hand"
    if a.K == 0:
        a.K = 600 if a.transition == "hand" else 300
    if a.alpha == 0.0:
        a.alpha = 0.25 if a.transition == "hand" else 1.0
    lam = a.lambda_prior

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    print("=" * 96)
    print(f"ANCHOR  lam={lam}{' (freeze_residual = inf)' if a.freeze_residual else ''}  "
          f"links={a.links} v_max={a.v_max} phys={a.phys}  elbo_obs={a.elbo_obs}  "
          f"transition={a.transition} pf={a.pf_impl} K={a.K} alpha={a.alpha}  tag={a.tag}")
    print("=" * 96, flush=True)

    tr_all = load_split("train")
    at = load_act("train")
    tr = [s for s in tr_all if s["stem"] in at and s["fold"] != a.holdout_fold]
    hold = [s for s in tr_all if s["stem"] in at and s["fold"] == a.holdout_fold]
    if a.eval_on == "eval":
        ev, ae = load_split("eval"), load_act("eval")
    elif a.eval_on == "holdout":
        ev, ae = hold, at
    else:
        ev, ae = tr, at
    if a.n_eval:
        ev = ev[:a.n_eval]
    print(f"train {len(tr)} (holdout fold {a.holdout_fold}: {len(hold)})   eval {len(ev)} "
          f"[{a.eval_on}]", flush=True)

    # ---- the SUPERVISED, FROZEN emission (fitted on the training songs only) ----
    t0 = time.time()
    emis = PhaseEmission(bins_per_beat=a.bpb, likelihood=a.lik, smooth=0.0).fit(tr, at)
    c_tr, _ = obs_contrast(emis, tr, at)
    print(f"supervised emission lik={a.lik} bpb={a.bpb} songs/meter={emis.n_used}  "
          f"obs_contrast(train)={c_tr:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    prior_vec = np.zeros(4)
    for s in tr:
        m = _estimate_meter(s["beats"], s["downs"])
        if 1 <= m <= 4:
            prior_vec[m - 1] += 1
    prior_vec = prior_vec / prior_vec.sum()

    torch.manual_seed(0)
    model = AnchoredVAE(hidden=a.hidden, num_meters=4, links=a.links, v_max=a.v_max,
                        phys=a.phys, p_switch=a.p_switch, meter_prior=prior_vec).to(DEV)
    model.freeze_residual = a.freeze_residual
    # keep the arm_ii learned emission around under a private name (deploy always uses the
    # supervised table via SupEmisModel.obs_logp)
    model.h_dec_logp = lambda zf, o: VB.BarPointerVAE_B.obs_logp(model, zf, o)
    if a.init_ckpt:
        sd = torch.load(a.init_ckpt, map_location="cpu")
        missing = model.load_state_dict(sd["model"], strict=False)
        print(f"loaded {a.init_ckpt}: missing={list(missing.missing_keys)} "
              f"unexpected={list(missing.unexpected_keys)}", flush=True)
    model.attach(emis, DEV)

    if a.check:
        preflight(model, tr, at, a, DEV)
        return

    # ------------------------------------------------------------------ TRAIN
    hist = []
    if a.steps > 0:
        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=a.lr)
        model.train()
        t0 = time.time()
        for step in range(1, a.steps + 1):
            beta = min(1.0, step / a.warmup)
            temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
            h, b, d = sample_batch(tr, at, rng, a.bs, a.frames, DEV)
            opt.zero_grad()
            loss, info, _ = elbo_anchor(model, h, b, d, h, temperature=temp, beta=beta,
                                        obs_w=a.obs_w, lam=lam, elbo_obs=a.elbo_obs,
                                        anchor_mean=a.anchor_mean)
            if not torch.isfinite(loss):
                print("NaN @", step, flush=True); break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            info["step"] = step; hist.append(info)
            if step % a.log_every == 0 or step == 1:
                print(f"  s{step:5d} loss={info['loss']:9.1f} rec_b={info['recon_beat']:7.1f} "
                      f"rec_db={info['recon_db']:7.1f} rec_o={info['recon_obs']:8.1f} "
                      f"kl={info['kl']:8.1f} (phi {info['kl_phase']:7.1f} lv {info['kl_level']:6.1f} "
                      f"dv {info['kl_dev']:6.1f} m {info['kl_meter']:5.1f}) | "
                      f"ANC={info['anchor']:9.2f} (ph {info['anchor_phase']:8.2f} "
                      f"lv {info['anchor_level']:7.2f} dv {info['anchor_dev']:7.2f} "
                      f"m {info['anchor_meter']:6.2f}) {info['anchor_phase_pf']:.3f}/fr | "
                      f"g_psi={info['gamma_psi']:.2e} s_lv={info['s_lv']:.2e} "
                      f"s_dv={info['s_dv']:.2e} sat={info['sat_frac']:.2f} "
                      f"FRAC_NEG_prior={info['frac_neg_prior']:.4f} "
                      f"{step/(time.time()-t0):.2f} it/s", flush=True)
        torch.save({"model": model.state_dict(), "config": vars(a)}, f"{HERE}/{a.tag}.pt")
    model.eval()

    # ------------------------------------------------------------------ DEPLOY
    print(f"\nDEPLOY  transition={a.transition}  emission=SUPERVISED(frozen)  "
          f"K={a.K} alpha={a.alpha}", flush=True)
    t0 = time.time()
    rows = deploy(model, emis, ev, ae, a, prior_vec, DEV)
    res = {"config": vars(a), "train_hist": hist[-1] if hist else None,
           "emission_contrast_train": c_tr}
    for k in ("mean", "map", "path", "pf_meter_path"):
        if rows[k]:
            d = summarize(rows[k], f"lam={lam} {a.transition}-trans + sup-emis [{k}]")
            pr(d); res[k] = d
    res["rows"] = rows["path"]
    print(f"({time.time()-t0:.0f}s)")
    json.dump(res, open(f"{HERE}/{a.tag}.json", "w"), indent=1, default=float)
    print("WROTE", f"{HERE}/{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
