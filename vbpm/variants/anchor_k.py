"""The categorical-anchor variant: q(k | x) over enumerated bar shifts, exact sum."""
from __future__ import annotations

import math

import torch
from torch import nn

from ..constants import TWO_PI
from ..model import VBPM
from ..vonmises import sample_vonmises
from .base import (common_kwargs, objective, on_epoch,
                   refuse_unsupported)

DEFAULTS = {"anchor_slots": 64}


class AnchorKVAE(VBPM):
    """VBPM plus a categorical anchor-slot head on phase-binned activations."""

    def __init__(self, input_dim: int, anchor_slots: int = 64, d_model: int = 128,
                 k_hidden: int = 256, **kw):
        super().__init__(input_dim, d_model=d_model, **kw)
        assert self.emission_net is None, \
            "anchor_k vectorises the closed-form emission over [B, C, T]; " \
            "the transformer emission is not supported here"
        self.anchor_slots = anchor_slots
        self.k_head = nn.Sequential(nn.Linear(anchor_slots, k_hidden), nn.ReLU(),
                                    nn.Linear(k_hidden, anchor_slots))
        # same birth rule as Encoder.out: small but NOT zero (dead-subnetwork lesson)
        nn.init.normal_(self.k_head[-1].weight, std=1e-2)
        nn.init.zeros_(self.k_head[-1].bias)
        shifts = TWO_PI * torch.arange(anchor_slots, dtype=torch.float32) / anchor_slots
        # wrapped to (-pi, pi]; C=1 degenerates to a single zero shift (= base model)
        self.register_buffer("slot_shifts",
                             torch.atan2(torch.sin(shifts), torch.cos(shifts)))

    def bin_downbeat(self, a, mu, mask=None):
        """[B, T] evidence, [B, T] phase -> [B, C]: a_t pooled into C phase bins."""
        B, T = mu.shape
        C = self.anchor_slots
        w = torch.ones(B, T, device=mu.device) if mask is None else mask
        bins = torch.remainder(mu.detach(), TWO_PI).div(TWO_PI).mul(C) \
                    .long().clamp(max=C - 1)
        flat = (torch.arange(B, device=mu.device)[:, None] * C + bins).reshape(-1)
        sums = torch.zeros(B * C, device=mu.device).index_add_(
            0, flat, (a * w).reshape(-1))
        cnt = torch.zeros(B * C, device=mu.device).index_add_(0, flat, w.reshape(-1))
        return (sums / cnt.clamp(min=1.0)).reshape(B, C)

    def slot_logits(self, a, mu, mask=None):
        """[B, T], [B, T] -> [B, C]: slot scores from the binned histogram."""
        return self.k_head(self.bin_downbeat(a, mu, mask))

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0):
        """ELBO with k marginalised exactly; same contract as the base forward."""
        trunk = self.encoder.features(h, mask)
        post = self.encoder.heads(trunk, mask, h)
        mu, kappa = post["phase"]["mu"], post["phase"]["kappa"]
        kl = self.kl_jitter(mu, kappa, mask)

        C = self.anchor_slots
        w_mask = torch.ones_like(mu) if mask is None else mask
        a_t = self.encoder.downbeat_scores(trunk, self.encoder.read_out(trunk), w_mask, h)
        a_bin = self.bin_downbeat(a_t, mu, mask)
        if self.training and C > 1:
            # rotation augmentation: mu_eff = mu + c_r, exactly (see module doc)
            r = torch.randint(0, C, (mu.shape[0],), device=mu.device)
            ar = torch.arange(C, device=mu.device)
            a_bin = torch.gather(a_bin, 1, (ar[None] - r[:, None]) % C)
            recon_gather = (ar[None] + r[:, None]) % C
        else:
            recon_gather = None
        log_q = nn.functional.log_softmax(self.k_head(a_bin), dim=-1)
        q_k = log_q.exp()

        weight = torch.where(y > 0, torch.as_tensor(pos_weight, device=y.device,
                                                    dtype=torch.float32),
                             torch.ones((), device=y.device, dtype=torch.float32)) * mask
        y_c = y.float().unsqueeze(1).expand(-1, C, -1)

        recon_k = 0.0
        for _ in range(samples):
            phi = mu + sample_vonmises(kappa)                        # shared across k
            phi_c = phi.unsqueeze(1) + self.slot_shifts[None, :, None]
            per_frame = nn.functional.binary_cross_entropy_with_logits(
                self.emission_logits(phi_c), y_c, reduction="none")
            recon_k = recon_k - (per_frame * weight.unsqueeze(1)).sum(-1)
        recon_k = recon_k / samples                                  # [B, C]
        if recon_gather is not None:
            recon_k = torch.gather(recon_k, 1, recon_gather)         # R'_k = R_{k+r}

        recon = (q_k * recon_k).sum(-1)
        kl_k = math.log(C) + (q_k * log_q).sum(-1)                   # KL(q(k) || U)

        return {"elbo": recon - kl - kl_k, "recon": recon, "kl": kl + kl_k,
                "phi": mu, "kappa": kappa,
                "recon_k": recon_k.detach(), "log_q": log_q.detach()}

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        """Deployment: mu shifted by the argmax slot. Reads audio (+ length mask) only."""
        assert not self.training, "deployment path must run in eval mode"
        trunk = self.encoder.features(h, mask)
        mu = self.encoder.heads(trunk, mask, h)["phase"]["mu"]
        w_mask = torch.ones_like(mu) if mask is None else mask
        a_t = self.encoder.downbeat_scores(trunk, self.encoder.read_out(trunk), w_mask, h)
        k = self.slot_logits(a_t, mu, mask).argmax(-1)
        return mu + self.slot_shifts[k].unsqueeze(-1)


def build_model(cfg, input_dim: int) -> AnchorKVAE:
    """The base recipe's model with the categorical anchor head."""
    # kl_k rides out["kl"], so beta-annealing would scale the k-entropy pull too; at
    # beta 1 throughout (warmup 0) the objective is the exact ELBO. Revisit before
    # enabling annealing with this variant (pre-launch review, lens 1).
    assert cfg.beta_warmup == 0, "anchor_k folds kl_k into kl; run it at beta=1"
    refuse_unsupported(cfg, "anchor_k",
                       supported=("downbeat_source", "detector_layers", "unified_bar_tempo"))
    kw = common_kwargs(cfg)
    for k in ("readout",):
        kw.pop(k, None)
    return AnchorKVAE(input_dim, anchor_slots=cfg.anchor_slots, **kw)


def optimizer(model, cfg):
    """Two Adam groups: k_head at the lab-validated 1e-3, rest at cfg.lr.

    q-side only; same clipping set as base (everything).
    """
    head_ids = {id(p) for p in model.k_head.parameters()}
    rest = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.Adam([
        {"params": rest, "lr": cfg.lr},
        {"params": list(model.k_head.parameters()), "lr": 1e-3},
    ])
    return opt, list(model.parameters())


def epoch_note(model, probe) -> str:
    """Epoch telemetry: Hk (categorical entropy) and agree (argmax match)."""
    if model.anchor_slots == 1:
        return ""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(probe["h"], probe["mask"], probe["y"])
        log_q = out["log_q"]
        q = log_q.exp()
        entropy = -(q * log_q).sum(-1).mean() / math.log(model.anchor_slots)
        agree = (q.argmax(-1) == out["recon_k"].argmax(-1)).float().mean()
    if was_training:
        model.train()
    return f"  Hk {float(entropy):.2f}  agree {float(agree):.2f}"
