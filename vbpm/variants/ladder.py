"""Isolation ladder: baseline -> anchor_k v2, ONE change per rung."""
from __future__ import annotations

import math

import torch
from torch import nn

from ..model import TWO_PI, VBPM, sample_vonmises
from .base import objective, on_epoch  # noqa: F401  -- re-exported hooks
from .base import refuse_unsupported

DEFAULTS = {"k_slots": 0, "head_input": "trunk_frame0", "head_arch": "linear",
            "head_lr": 0.0, "rotation_aug": False, "k_hidden": 256}


class LadderVAE(VBPM):
    """One model, five switches. Every rung of the ladder is an instance of this."""

    def __init__(self, input_dim, k_slots=0, head_input="trunk_frame0",
                 head_arch="linear", rotation_aug=False, k_hidden=256,
                 d_model=128, **kw):
        super().__init__(input_dim, d_model=d_model, **kw)
        assert self.emission_net is None, "ladder uses the closed-form emission"
        self.k_slots = int(k_slots)
        self.head_input = head_input
        self.rotation_aug = bool(rotation_aug)
        self.slots = max(self.k_slots, 64)      # bins for folding, even when k is off

        n_out = self.k_slots if self.k_slots > 0 else 2   # slot logits, or (cos, sin)
        n_in = {"trunk_mean": d_model, "folded_acts": 2 * self.slots}.get(head_input)
        if head_input == "trunk_frame0":
            self.head = None                    # the base encoder's own offset head
        elif head_arch == "linear":
            self.head = nn.Linear(n_in, n_out)
        else:
            self.head = nn.Sequential(nn.Linear(n_in, k_hidden), nn.ReLU(),
                                      nn.Linear(k_hidden, n_out))
        if self.head is not None:
            last = self.head if head_arch == "linear" else self.head[-1]
            nn.init.normal_(last.weight, std=1e-2)   # small but NOT zero
            nn.init.zeros_(last.bias)

        if self.k_slots > 0:
            shifts = TWO_PI * torch.arange(self.k_slots, dtype=torch.float32) / self.k_slots
            self.register_buffer("slot_shifts",
                                 torch.atan2(torch.sin(shifts), torch.cos(shifts)))

    # ---- head input representations -------------------------------------------------
    def bin_activations(self, h, mu, mask=None):
        """[B,T,D],[B,T] -> [B,C,2]: activations masked-mean pooled into C phase bins.

        Identical arithmetic to anchor_k.bin_activations (mu detached).
        """
        B, T = mu.shape
        C = self.slots
        acts = torch.sigmoid(h[..., -2:])
        w = torch.ones(B, T, device=mu.device) if mask is None else mask
        bins = torch.remainder(mu.detach(), TWO_PI).div(TWO_PI).mul(C).long().clamp(max=C - 1)
        flat = (torch.arange(B, device=mu.device)[:, None] * C + bins).reshape(-1)
        sums = torch.zeros(B * C, 2, device=mu.device).index_add_(
            0, flat, acts.reshape(-1, 2) * w.reshape(-1, 1))
        cnt = torch.zeros(B * C, device=mu.device).index_add_(0, flat, w.reshape(-1))
        return (sums / cnt.clamp(min=1.0)[:, None]).reshape(B, C, 2)

    def head_features(self, h, trunk, mu, mask):
        if self.head_input == "trunk_mean":
            w = torch.ones(mu.shape, device=mu.device) if mask is None else mask
            return (trunk * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp(min=1.0)
        return self.bin_activations(h, mu, mask).flatten(1)

    # ---- trajectory -----------------------------------------------------------------
    def base_path(self, h, mask=None):
        """(trunk, mu, kappa) from the encoder. mu still carries its own offset head."""
        trunk = self.encoder.features(h, mask)
        post = self.encoder.heads(trunk, mask, h)
        mu, kappa = post["phase"]["mu"], post["phase"]["kappa"]
        return trunk, mu, kappa

    def scalar_offset(self, h, trunk, mu, mask):
        """k_slots=0 with a NON-frame0 input: (cos,sin)->atan2 offset, added to the ramp."""
        feats = self.head_features(h, trunk, mu, mask)
        cs = self.head(feats)
        return torch.atan2(cs[:, 0], cs[:, 1])

    def trajectory(self, h, mask=None):
        trunk, mu, kappa = self.base_path(h, mask)
        if self.k_slots > 0:
            k = self.head(self.head_features(h, trunk, mu, mask)).argmax(-1)
            return mu + self.slot_shifts[k].unsqueeze(-1), kappa
        if self.head is None:
            return mu, kappa                                  # baseline: frame-0 offset
        return mu + self.scalar_offset(h, trunk, mu, mask).unsqueeze(-1), kappa

    # ---- objective ------------------------------------------------------------------
    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0):
        if self.k_slots == 0:
            return super().forward(h, mask, y, samples, pos_weight)

        trunk, mu, kappa = self.base_path(h, mask)
        kl = self.kl_jitter(mu, kappa, mask)
        C = self.k_slots

        feats = self.head_features(h, trunk, mu, mask)
        recon_gather = None
        if self.training and self.rotation_aug and self.head_input == "folded_acts":
            a_bin = feats.reshape(feats.shape[0], self.slots, 2)
            r = torch.randint(0, C, (mu.shape[0],), device=mu.device)
            ar = torch.arange(C, device=mu.device)
            a_bin = torch.gather(a_bin, 1,
                                 ((ar[None] - r[:, None]) % C)[..., None].expand(-1, -1, 2))
            feats = a_bin.flatten(1)
            recon_gather = (ar[None] + r[:, None]) % C

        log_q = nn.functional.log_softmax(self.head(feats), dim=-1)
        q_k = log_q.exp()

        weight = torch.where(y > 0, torch.as_tensor(pos_weight, device=y.device,
                                                    dtype=torch.float32),
                             torch.ones((), device=y.device, dtype=torch.float32)) * mask
        y_c = y.float().unsqueeze(1).expand(-1, C, -1)

        recon_k = 0.0
        for _ in range(samples):
            phi = mu + sample_vonmises(kappa)
            phi_c = phi.unsqueeze(1) + self.slot_shifts[None, :, None]
            per_frame = nn.functional.binary_cross_entropy_with_logits(
                self.emission_logits(phi_c), y_c, reduction="none")
            recon_k = recon_k - (per_frame * weight.unsqueeze(1)).sum(-1)
        recon_k = recon_k / samples
        if recon_gather is not None:
            recon_k = torch.gather(recon_k, 1, recon_gather)

        recon = (q_k * recon_k).sum(-1)
        kl_k = math.log(C) + (q_k * log_q).sum(-1)
        return {"elbo": recon - kl - kl_k, "recon": recon, "kl": kl + kl_k,
                "phi": mu, "kappa": kappa,
                "recon_k": recon_k.detach(), "log_q": log_q.detach()}


def build_model(cfg, input_dim: int) -> LadderVAE:
    refuse_unsupported(cfg, "ladder")
    assert cfg.beta_warmup == 0, "ladder folds kl_k into kl; run at beta=1"
    return LadderVAE(input_dim, k_slots=cfg.k_slots, head_input=cfg.head_input,
                     head_arch=cfg.head_arch, rotation_aug=cfg.rotation_aug,
                     k_hidden=cfg.k_hidden, emission=cfg.emission,
                     emission_layers=cfg.emission_layers,
                     emission_positional=cfg.emission_positional,
                     kappa_physical=cfg.kappa_physical)


def optimizer(model, cfg):
    """One group, or a second group for the head when head_lr > 0 (the R3->R4 change)."""
    if cfg.head_lr <= 0 or model.head is None:
        params = list(model.parameters())
        return torch.optim.Adam(params, lr=cfg.lr), params
    head_ids = {id(p) for p in model.head.parameters()}
    rest = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.Adam([{"params": rest, "lr": cfg.lr},
                            {"params": list(model.head.parameters()), "lr": cfg.head_lr}])
    return opt, list(model.parameters())


def epoch_note(model, probe) -> str:
    if model.k_slots == 0:
        return ""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(probe["h"], probe["mask"], probe["y"])
        q = out["log_q"].exp()
        hk = -(q * out["log_q"]).sum(-1).mean() / math.log(model.k_slots)
        agree = (q.argmax(-1) == out["recon_k"].argmax(-1)).float().mean()
    if was_training:
        model.train()
    return f"  Hk {float(hk):.2f}  agree {float(agree):.2f}"
