"""The anchor as a DISCRETE TIME grid, marginalised exactly: 80 ms bins over the window."""
from __future__ import annotations

import math

import torch
from torch import nn

from .base import refuse_unsupported
from ..model import (VBPM, Encoder, bounded_kappa, sample_vonmises)


# 4 frames at 50 fps = 80 ms; see the module docstring on why this is an integer.
STRIDE_FRAMES = 4
HARMONICS = 8
K_HEAD_HIDDEN = 32


def candidate_anchors(cum, mask, stride: int):
    """(c [B, C], ok [B, C]) -- candidate anchor VALUES, read off the trajectory."""
    idx = torch.arange(0, cum.shape[1], stride, device=cum.device)
    ok = mask[:, idx] > 0
    ok[:, 0] |= ~ok.any(-1)
    return cum[:, idx], ok


class AnchorEncoder(Encoder):
    """q(phi_t | x) with the OFFSET HEAD REMOVED: the anchor is enumerated, not emitted."""

    def __init__(self, input_dim: int, *args, **kw):
        super().__init__(input_dim, *args,
                         channels=("log_phi_kappa", "log_dotphi", "residual"), **kw)

    def heads(self, trunk, mask=None, h=None):
        """Trunk -> (cum [B, T], kappa [B, T], residual_raw [B, T]). Not a sample."""
        out = self.read_out(trunk)
        kappa = bounded_kappa(torch.exp(out["log_phi_kappa"] + self.log_phi_kappa_bias) + 1e-3)

        log_dotphi = self._pool(out["log_dotphi"] + self.log_dotphi_bias, self.pool_span)
        dotphi = torch.exp(log_dotphi.clamp(math.log(0.01), math.log(0.2)))
        cum = torch.cumsum(dotphi, dim=1) - dotphi[:, :1]         # monotone, cum[:, 0] = 0

        return cum, kappa, out["residual"]


class AnchorTimeVAE(VBPM):
    """VBPM whose anchor is a categorical over candidate anchor TIMES."""

    def __init__(self, input_dim: int, stride: int = STRIDE_FRAMES,
                 harmonics: int = HARMONICS, d_model: int = 128, **kw):
        super().__init__(input_dim, d_model=d_model, **kw)
        self.encoder = AnchorEncoder(input_dim, d_model,
                                     kappa_physical=self.kappa_physical)
        self.stride = int(stride)
        self.harmonics = int(harmonics)

        # OUR evidence head, over the frontend's features. Deliberately not the frontend's
        # own activation channels: see the module docstring on attribution, and note the
        # frontend's task heads are frame-wise linears on these same features, so this form
        # can represent them exactly -- only the initialisation differs.
        self.downbeat_head = nn.Linear(input_dim, 1)
        self.k_head = nn.Sequential(nn.Linear(2 * harmonics, K_HEAD_HIDDEN),
                                    nn.GELU(),
                                    nn.Linear(K_HEAD_HIDDEN, 1))
        # The output layer keeps torch's DEFAULT init. An earlier std=1e-3 override, chosen
        # to start every candidate equally scored, combined with descriptors of ~1e-2 to
        # produce logits ~1e-5 and a permanently uniform q -- the head could not escape its
        # own initialisation. Near-uniform is not worth buying at the price of a dead head;
        # the pricing gradient does not need help starting from a flat posterior.
        nn.init.zeros_(self.k_head[-1].bias)

    # ------------------------------------------------------------------ the evidence

    def candidate_features(self, h, cum, c, mask):
        """[B, C, 2M] descriptors for every candidate, EXACTLY, in O(M T + M C)."""
        a = torch.sigmoid(self.downbeat_head(h).squeeze(-1)) * mask                 # [B, T]
        m = torch.arange(1, self.harmonics + 1, device=cum.device,
                         dtype=cum.dtype)                                      # [M]

        ang = m[None, :, None] * cum[:, None, :]                               # [B, M, T]
        evidence_mass = a.sum(1).clamp(min=1e-6)[:, None]
        s_cos = (a[:, None, :] * torch.cos(ang)).sum(-1) / evidence_mass       # [B, M]
        s_sin = (a[:, None, :] * torch.sin(ang)).sum(-1) / evidence_mass       # [B, M]

        p = m[None, :, None] * c[:, None, :]                                   # [B, M, C]
        cos_p, sin_p = torch.cos(p), torch.sin(p)
        re = s_cos[..., None] * cos_p + s_sin[..., None] * sin_p                # [B, M, C]
        im = s_sin[..., None] * cos_p - s_cos[..., None] * sin_p
        return torch.cat([re, im], dim=1).permute(0, 2, 1)                     # [B, C, 2M]

    def candidate_logits(self, h, cum, c, ok, mask):
        """[B, C] unnormalised log q(k | x); invalid candidates are -inf, not merely small."""
        feat = self.candidate_features(h, cum, c, mask)
        w = ok.unsqueeze(-1).to(feat.dtype)
        n = w.sum(1, keepdim=True).clamp(min=1.0)
        mean = (feat * w).sum(1, keepdim=True) / n
        var = (((feat - mean) ** 2) * w).sum(1, keepdim=True) / n
        feat = (feat - mean) / (var + 1e-4).sqrt()

        logits = self.k_head(feat).squeeze(-1)
        return logits.masked_fill(~ok, float("-inf"))

    def residual(self, residual_raw, cum, mask):
        """[B] a sub-bin refinement of the anchor, BOUNDED to half a bin of phase."""
        inc = (cum[:, 1:] - cum[:, :-1]) * mask[:, 1:]
        tempo = inc.sum(1) / mask[:, 1:].sum(1).clamp(min=1.0)                  # [B]
        half_bin = tempo * self.stride / 2.0
        scalar = (residual_raw * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return half_bin * torch.tanh(scalar)

    # ------------------------------------------------------------------ the objective

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0):
        """One ELBO evaluation with k marginalised exactly. Same signature as the base."""
        cum, kappa, residual_raw = self.encoder(h, mask)
        kl_traj = self.kl_jitter(cum, kappa, mask)

        c, ok = candidate_anchors(cum, mask, self.stride)                      # [B, C]
        logq = torch.log_softmax(self.candidate_logits(h, cum, c, ok, mask), dim=-1)
        q = logq.exp()

        base = cum + self.residual(residual_raw, cum, mask)[:, None]           # [B, T]
        mu_k = base[:, None, :] - c[..., None]                                 # [B, C, T]

        weight = torch.where(y > 0, torch.as_tensor(pos_weight, device=y.device,
                                                    dtype=torch.float32),
                             torch.ones((), device=y.device, dtype=torch.float32)) * mask
        target = y.float()[:, None, :].expand_as(mu_k)

        # The von Mises noise is drawn ONCE and shared across candidates: the candidates
        # differ by a deterministic constant, so common random numbers both halve the cost
        # and remove sampling noise from the DIFFERENCES between R_k -- which is the only
        # thing the pricing gradient reads.
        reward = 0.0
        for _ in range(samples):
            phi = mu_k + sample_vonmises(kappa)[:, None, :]
            per_frame = nn.functional.binary_cross_entropy_with_logits(
                self.emission_logits(phi), target, reduction="none")
            reward = reward - (per_frame * weight[:, None, :]).sum(-1)         # [B, C]
        reward = reward / samples

        recon = (q * reward.nan_to_num(0.0)).sum(-1)
        n_i = ok.sum(-1).clamp(min=1)
        neg_entropy = (q * logq.masked_fill(~ok, 0.0)).sum(-1)
        kl = kl_traj + torch.log(n_i.to(cum.dtype)) + neg_entropy

        best = reward.masked_fill(~ok, float("-inf")).argmax(-1)
        rows = torch.arange(len(best), device=cum.device)

        return {"elbo": recon - kl, "recon": recon, "kl": kl,
                "phi": mu_k[rows, best], "kappa": kappa,
                "logq": logq, "R": reward, "n_i": n_i, "ok": ok}

    # ------------------------------------------------------------------ deployment

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        """Deployment: the mean path of the candidate q likes best. Audio only."""
        assert not self.training, "deployment path must run in eval mode"
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)

        cum, _kappa, residual_raw = self.encoder(h, mask)
        c, ok = candidate_anchors(cum, mask, self.stride)
        best = self.candidate_logits(h, cum, c, ok, mask).argmax(-1)
        rows = torch.arange(len(best), device=cum.device)

        base = cum + self.residual(residual_raw, cum, mask)[:, None]
        return base - c[rows, best][:, None]


# ----------------------------------------------------------------- run.py hooks

# Config keys this variant adds on top of the mainline schema, with their defaults.
DEFAULTS = dict(
    anchor_stride_frames=STRIDE_FRAMES,   # 80 ms at 50 fps; INTEGER frames keeps candidates
                                          # exact. See the module docstring for why the
                                          # width is set by the model's error budget rather
                                          # than by the tolerance alone.
    anchor_harmonics=HARMONICS,           # M = 1 with fixed weights would be the closed-form
                                          # circular mean; M > 1 with a learned head is the
                                          # part that is not already the frontend's.
)


def build_model(cfg, input_dim: int) -> AnchorTimeVAE:
    """The time-anchored model. REQUIRES an elementwise emission -- see the assert."""
    refuse_unsupported(cfg, "anchor_time")
    assert cfg.emission in ("triangle", "cosine"), (
        f"anchor_time needs an elementwise emission, got {cfg.emission!r}: the "
        f"reconstruction is evaluated at every candidate, so a transformer emission would "
        f"run B x C sequences per step")
    return AnchorTimeVAE(input_dim, stride=cfg.anchor_stride_frames,
                         harmonics=cfg.anchor_harmonics, emission=cfg.emission,
                         emission_layers=cfg.emission_layers,
                         emission_positional=cfg.emission_positional,
                         kappa_physical=cfg.kappa_physical)


def optimizer(model, cfg):
    """(optimizer, params-to-clip). One Adam group; everything clipped, as base."""
    from . import base
    return base.optimizer(model, cfg)


def objective(out, beta: float, cfg):
    """The base ELBO unchanged: ``kl`` already carries both the trajectory and anchor terms."""
    from . import base
    return base.objective(out, beta, cfg)


def on_epoch(model, cfg, epoch: int) -> None:
    """Same emission-sharpness schedule as the base recipe."""
    from . import base
    base.on_epoch(model, cfg, epoch)


def epoch_note(model, probe) -> str:
    """Hq and agree -- the two numbers that read this mechanism."""
    with torch.no_grad():
        out = model(probe["h"], probe["mask"], probe["y"])
    logq, ok = out["logq"], out["ok"]
    q = logq.exp()
    entropy = -(q * logq.masked_fill(~ok, 0.0)).sum(-1)      # see forward: no -inf, ever
    agree = (q.argmax(-1) == out["R"].masked_fill(~ok, float("-inf")).argmax(-1))

    # Average over SCORABLE items only. A fully-masked backstop window keeps exactly one
    # candidate (candidate_anchors' guard), so log C_i = log 1 = 0 and its normalised
    # entropy is 0/0: nan, or ~1e6 under a clamped denominator. Either value from one
    # backstop item destroys the batch mean, and Hq is the number this whole variant is
    # read by. If nothing is scorable, say so rather than printing a fabricated average.
    live = out["n_i"] > 1
    if not bool(live.any()):
        return "  Hq    n/a  agree   n/a"
    normalised = entropy[live] / torch.log(out["n_i"][live].to(entropy.dtype))
    return (f"  Hq {float(normalised.mean()):5.3f}  "
            f"agree {float(agree[live].float().mean()):5.1%}")
