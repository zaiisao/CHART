"""The categorical-anchor variant: q(k | x) over enumerated bar shifts, exact sum.

What it fixes (2026-08-07 workflow verdicts): the anchor landscape over constant
shifts of mu is UNIMODAL with a flat top (~330 ms of the bar within 20 nats), so the
pathwise gradient trains the scalar offset head only to "somewhere on the top"
(train-fold median 70 ms, gtzan 137 ms) while the VALUE-argmax of the same trained
objective is precise (6-9 ms, 92% < 70 ms). The estimator, not the objective, was
the bottleneck: this variant gives q a head whose gradient reads objective VALUES.

Construction (auxiliary-variable ELBO; p is bitwise unchanged): extend the latent
with k ~ p(k) = Uniform(C), p(y | phi, k) = p(y | phi). Then

    L = sum_k q(k|x) E_{q(phi|x)}[ log p(y | phi + c_k) ]  -  KL_trajectory
        - ( log C - H[q(k|x)] )

is a standard ELBO on the SAME marginal p(y). The trajectory KL needs no per-k term:
kl_to_physical_prior reads increments only and phi_1 ~ Uniform, so it is exactly
invariant under the constant shift c_k. Only the reconstruction varies with k, and k
is marginalised EXACTLY -- C closed-form emission evaluations, no REINFORCE, no
Gumbel; common random numbers (one eps shared across k) keep the R_k differences
low-variance. dL/dlogit_j = q_j (R_j - sum_k q_k R_k): every slot is priced every
step, so anchor discovery is by construction rather than by search.

v2 head (2026-08-08 head-lab verdicts). v1's linear+time-mean trunk head could not
amortize (deployed slot 115 ms from value-argmax; gtzan F 0.278). The lab winner is
the PHASE-BINNED ACTIVATION head: the two frontend activation channels
(h[..., -2:], the beat/downbeat logits) are masked-mean pooled into C phase bins
under the encoder's own mu, and an MLP maps the [C, 2] histogram to slot logits
(lab: 27 ms / 70% <70 ms vs value-argmax bound 11 ms / 92%; v1-style head 141 ms).
Trunk features added nothing under augmentation and were dropped on parsimony.

Rotation augmentation (train only): the raw anchor of mu is unreliable, and the
un-augmented head FAILS the rotation-composition test (a slot-16 rotation of mu
costs 26 -> 135 ms). Fix: draw r ~ Uniform(C) per example and use mu_eff = mu + c_r
everywhere k-related -- implemented exactly by rotating the binned histogram by r
and re-indexing recon_k by (k + r) mod C. Because the slots tile the full circle
and the trajectory KL is shift-invariant, the objective for each r is the SAME
exact ELBO evaluated under a reparameterised q (q(k|x, r) with independent noise
r; the average over r is still a valid lower bound on log p(y)). p is untouched.
Lab: augmentation makes the head rotation-robust (26-28 ms at every tested r) AND
improves the un-rotated grade (27 ms / 70%).

Precedent heeded: psi's K>1 rotation mixture never learned the anchor -- but it was
trained by stop-gradient distillation, pathwise. Watch the epoch log's Hk (normalised
categorical entropy) and agree (argmax q == argmax R_k on the probe batch): the
mechanism is working exactly when Hk falls while agree rises.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..model import TWO_PI, BarPhaseVAE, sample_vonmises
from .base import objective, on_epoch  # noqa: F401  -- re-exported hooks

DEFAULTS = {"anchor_slots": 64}


class AnchorKVAE(BarPhaseVAE):
    """BarPhaseVAE plus a categorical anchor-slot head on phase-binned activations."""

    def __init__(self, input_dim: int, anchor_slots: int = 64, hidden: int = 128,
                 k_hidden: int = 256, **kw):
        super().__init__(input_dim, hidden=hidden, **kw)
        assert self.emission_net is None, \
            "anchor_k vectorises the closed-form emission over [B, C, T]; " \
            "the transformer emission is not supported here"
        self.anchor_slots = anchor_slots
        # phase-binned activation histogram [C, 2] -> slot logits [C]
        self.k_head = nn.Sequential(nn.Linear(2 * anchor_slots, k_hidden), nn.ReLU(),
                                    nn.Linear(k_hidden, anchor_slots))
        # same birth rule as Encoder.out: small but NOT zero (dead-subnetwork lesson)
        nn.init.normal_(self.k_head[-1].weight, std=1e-2)
        nn.init.zeros_(self.k_head[-1].bias)
        shifts = TWO_PI * torch.arange(anchor_slots, dtype=torch.float32) / anchor_slots
        # wrapped to (-pi, pi]; C=1 degenerates to a single zero shift (= base model)
        self.register_buffer("slot_shifts",
                             torch.atan2(torch.sin(shifts), torch.cos(shifts)))

    def bin_activations(self, h, mu, mask=None):
        """[B, T, D], [B, T] -> [B, C, 2]: activations pooled into C phase bins.

        The last two input dims (the frontend's beat/downbeat channels, sigmoided) are
        masked-mean pooled into C phase bins under mu.

        Bin indices are integer functions of mu (no gradient path), and the pooled
        values come from the frozen frontend: the k-head's ELBO gradient prices
        slots without back-pressure on the encoder through the binning.
        """
        B, T = mu.shape
        C = self.anchor_slots
        acts = torch.sigmoid(h[..., -2:])
        w = torch.ones(B, T, device=mu.device) if mask is None else mask
        bins = torch.remainder(mu.detach(), TWO_PI).div(TWO_PI).mul(C) \
                    .long().clamp(max=C - 1)
        flat = (torch.arange(B, device=mu.device)[:, None] * C + bins).reshape(-1)
        sums = torch.zeros(B * C, 2, device=mu.device).index_add_(
            0, flat, acts.reshape(-1, 2) * w.reshape(-1, 1))
        cnt = torch.zeros(B * C, device=mu.device).index_add_(0, flat, w.reshape(-1))
        return (sums / cnt.clamp(min=1.0)[:, None]).reshape(B, C, 2)

    def slot_logits(self, h, mu, mask=None):
        """[B, T, D], [B, T] -> [B, C]: slot scores from the binned histogram.

        The mask keeps pad frames out of the pool: without it, untrained pad-frame
        content (1/3 of every gtzan window) corrupts the histogram and flips the
        deployed anchor by whole slots (v1 lesson; the pad-vote gate enforces this).
        """
        return self.k_head(self.bin_activations(h, mu, mask).flatten(1))

    def forward(self, h, delta, mask, y, samples: int = 1, pos_weight: float = 1.0):
        """ELBO with k marginalised exactly; same contract as the base forward."""
        trunk = self.encoder.features(h)
        mu, kappa = self.encoder.heads(trunk, delta)
        kl = self.kl_to_physical_prior(mu, kappa, delta, mask)

        C = self.anchor_slots
        a_bin = self.bin_activations(h, mu, mask)
        if self.training and C > 1:
            # rotation augmentation: mu_eff = mu + c_r, exactly (see module doc)
            r = torch.randint(0, C, (mu.shape[0],), device=mu.device)
            ar = torch.arange(C, device=mu.device)
            a_bin = torch.gather(a_bin, 1,
                                 ((ar[None] - r[:, None]) % C)[..., None].expand(-1, -1, 2))
            recon_gather = (ar[None] + r[:, None]) % C
        else:
            recon_gather = None
        log_q = nn.functional.log_softmax(self.k_head(a_bin.flatten(1)), dim=-1)
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
                "mu": mu, "kappa": kappa,
                "recon_k": recon_k.detach(), "log_q": log_q.detach()}

    @torch.no_grad()
    def infer_phase(self, h, delta=None, mask=None):
        """Deployment: mu shifted by the argmax slot. Reads audio (+ length mask) only."""
        assert not self.training, "deployment path must run in eval mode"
        trunk = self.encoder.features(h)
        mu, _kappa = self.encoder.heads(trunk, delta)
        k = self.slot_logits(h, mu, mask).argmax(-1)
        return mu + self.slot_shifts[k].unsqueeze(-1)


def build_model(cfg, input_dim: int) -> AnchorKVAE:
    """The base recipe's model with the categorical anchor head."""
    # kl_k rides out["kl"], so beta-annealing would scale the k-entropy pull too; at
    # beta 1 throughout (warmup 0) the objective is the exact ELBO. Revisit before
    # enabling annealing with this variant (pre-launch review, lens 1).
    assert cfg.beta_warmup == 0, "anchor_k folds kl_k into kl; run it at beta=1"
    return AnchorKVAE(input_dim, anchor_slots=cfg.anchor_slots,
                      emission=cfg.emission, emission_layers=cfg.emission_layers,
                      emission_positional=cfg.emission_positional,
                      drift_bound=cfg.drift_bound, bar_rate=cfg.bar_rate,
                      kappa_physical=cfg.kappa_physical)


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
    """Epoch telemetry: Hk (categorical entropy) and agree (argmax match).

    Hk = normalised H[q(k)] (1 = uniform, 0 = collapsed); agree = argmax q vs argmax
    R_k on the probe batch (the mechanism is learning exactly when Hk falls while
    agree rises).
    """
    if model.anchor_slots == 1:
        return ""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(probe["h"], probe["delta"], probe["mask"], probe["y"])
        log_q = out["log_q"]
        q = log_q.exp()
        entropy = -(q * log_q).sum(-1).mean() / math.log(model.anchor_slots)
        agree = (q.argmax(-1) == out["recon_k"].argmax(-1)).float().mean()
    if was_training:
        model.train()
    return f"  Hk {float(entropy):.2f}  agree {float(agree):.2f}"
