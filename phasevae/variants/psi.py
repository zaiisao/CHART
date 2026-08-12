"""The §9 variant: q(z|h,b) scaffolding + conditional prior p_psi(z|h), Sohn-orthodox."""
from __future__ import annotations

import torch
from torch import nn

from ..model import (TWO_PI, BarPhaseVAE, Encoder, sample_vonmises,
                     vonmises_log_density)


class PosteriorEncoder(Encoder):
    """q(z | h, b): the target enters as frame t's SECOND variable (h_t, y_t)."""

    reads_target = True

    def __init__(self, input_dim: int, **kw):
        super().__init__(input_dim + 1, **kw)

    def forward(self, h, mask=None, y=None):
        """[B, T, D] audio + [B, T] target -> (mu, kappa); same heads, richer input."""
        assert y is not None, "posterior encoder requires the target input y"
        x = torch.cat([h, y.unsqueeze(-1).to(h.dtype)], dim=-1)
        return self.heads(self.features(x, mask))


class RotationPrior(Encoder):
    """p_psi(z | h): a base path plus (for K > 1) a whole-trajectory rotation mixture."""

    def __init__(self, input_dim: int, rotations: int = 1, **kw):
        super().__init__(input_dim, **kw)
        self.rotations = rotations
        self.rot_head = (nn.Linear(self.proj.out_features, rotations)
                         if rotations > 1 else None)

    def forward(self, h, mask=None):
        """[B, T, D] -> (mu, kappa) or, for K > 1, (mu, kappa, rot_logits [B, K])."""
        trunk = self.features(h, mask)
        mu, kappa = self.heads(trunk)
        if self.rot_head is None:
            return mu, kappa
        return mu, kappa, self.rot_head(trunk).mean(dim=1)


class PsiBarPhaseVAE(BarPhaseVAE):
    """BarPhaseVAE with the third parameter set psi: deploys the conditional prior."""

    def __init__(self, input_dim: int, rotations: int = 1, d_model: int = 128, **kw):
        super().__init__(input_dim, d_model=d_model, **kw)
        self.encoder = PosteriorEncoder(input_dim, d_model=d_model)
        self.prior_net = RotationPrior(input_dim, rotations, d_model=d_model)
        self.psi_rotations = rotations

    @property
    def deployed_net(self):
        """Sohn-orthodox: the conditional prior is what inference reads."""
        return self.prior_net

    def kl_to_conditional_prior(self, z, mu_q, kappa_q, mu_p, kappa_p, rot_logits,
                                mask):
        """MC estimate of KL( q(z|h,b) || p_psi(z|h) ) at the sampled trajectory z."""
        log_q = (vonmises_log_density(z, mu_q, kappa_q) * mask).sum(1)
        if rot_logits is None:
            log_p = (vonmises_log_density(z, mu_p, kappa_p) * mask).sum(1)
            return log_q - log_p

        rotations = TWO_PI * torch.arange(self.psi_rotations,
                                          device=z.device) / self.psi_rotations
        per_component = (vonmises_log_density(
            z.unsqueeze(1), mu_p.unsqueeze(1) + rotations[None, :, None],
            kappa_p.unsqueeze(1)) * mask.unsqueeze(1)).sum(2)          # [B, K]
        log_p = torch.logsumexp(
            torch.log_softmax(rot_logits, dim=1) + per_component, dim=1)
        return log_q - log_p

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0):
        """ELBO plus the psi terms; q trains against the PHYSICAL prior only."""
        mu, kappa = self.encoder(h, mask, y)
        prior_out = self.prior_net(h, mask)
        mu_p, kappa_p = prior_out[0], prior_out[1]
        rot_logits = prior_out[2] if len(prior_out) == 3 else None

        kl = self.kl_to_physical_prior(mu, kappa, mask)

        weight = torch.where(y > 0, torch.as_tensor(pos_weight, device=y.device,
                                                    dtype=torch.float32),
                             torch.ones((), device=y.device, dtype=torch.float32)) * mask

        recon, distill, anchor = 0.0, None, None
        for s_i in range(samples):
            phi = mu + sample_vonmises(kappa)
            if s_i == 0:
                distill = self.kl_to_conditional_prior(
                    phi.detach(), mu.detach(), kappa.detach(), mu_p, kappa_p,
                    rot_logits, mask)
                # Eq. 27 physics anchoring: keep p_psi NEAR the physical random walk,
                # not an unconstrained learned prior (the old codebase's psi ran away).
                anchor = self.kl_to_physical_prior(mu_p, kappa_p, mask)
            per_frame = nn.functional.binary_cross_entropy_with_logits(
                self.emission_logits(phi), y.float(), reduction="none")
            recon = recon - (per_frame * weight).sum(1)
        recon = recon / samples

        return {"elbo": recon - kl, "recon": recon, "kl": kl, "mu": mu,
                "kappa": kappa, "distill": distill, "prior_anchor": anchor}

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        """Sohn-orthodox deployment: the conditional prior, never the posterior."""
        assert not self.training, "deployment path must run in eval mode"
        prior_out = self.prior_net(h, mask)
        if len(prior_out) == 2:
            return prior_out[0]

        mu_p, _kappa_p, rot_logits = prior_out
        best = rot_logits.argmax(dim=1, keepdim=True)
        return mu_p + TWO_PI * best.to(mu_p.dtype) / self.psi_rotations


# ----------------------------------------------------------------- run.py hooks

# Config keys this variant adds on top of the mainline schema, with their defaults.
DEFAULTS = dict(
    rotations=1,          # K=1 (no mixture) is the parity configuration; K>1 mixture
                          # never learned the anchor -- see module docstring
    lambda_prior=1.0,     # Eq. 27 weight anchoring p_psi to the physical prior
)


def build_model(cfg, input_dim: int) -> PsiBarPhaseVAE:
    """The §9 model; ``rotations``/``lambda_prior`` come from the config."""
    return PsiBarPhaseVAE(input_dim, rotations=cfg.rotations, emission=cfg.emission,
                          emission_layers=cfg.emission_layers,
                          emission_positional=cfg.emission_positional)


def optimizer(model, cfg):
    """Rotation head (if any): own group, 10x lr, EXEMPT from the clip that starved it.

    Everything else is the base single group.
    """
    rot_params = (list(model.prior_net.rot_head.parameters())
                  if model.prior_net.rot_head is not None else [])
    rot_ids = {id(p) for p in rot_params}
    main_params = [p for p in model.parameters() if id(p) not in rot_ids]
    opt = torch.optim.Adam([{"params": main_params, "lr": cfg.lr},
                            {"params": rot_params, "lr": cfg.lr * 10}])
    return opt, main_params


def objective(out, beta: float, cfg):
    """ELBO minus the distillation and physics-anchor terms (Eq. 27, weight lambda)."""
    return (out["recon"] - beta * out["kl"]
            - out["distill"] - cfg.lambda_prior * out["prior_anchor"])


def on_epoch(model, cfg, epoch: int) -> None:
    """Same sharpness schedule as the base recipe."""
    from . import base
    base.on_epoch(model, cfg, epoch)


def epoch_note(model, probe) -> str:
    """Rotation-head vitality: a flat spread means the argmax decode is arbitrary."""
    if model.prior_net.rot_head is None:
        return ""
    with torch.no_grad():
        _m, _k, rot = model.prior_net(probe["h"], probe["mask"])
    return f"  rot-spread {float(rot.std()):6.4f}"
