"""Marginal-carriage continuous-phase smoother (attempt-3 next_modification).

q(phi_{1:T} | x, c) is the chain-structured posterior obtained by combining the
per-frame von Mises walk prior (kappa_physical per FRAME, stride 1) with
per-frame recognition potentials psi_t(phi) read locally from the 512-dim
features (band-limited Fourier, H harmonics). Inference is exact message
passing evaluated by N-point circular quadrature -- inference numerics over a
continuous latent, not state tiling. No path is ever sampled: C_commit = 0.

ELBO (per rate component c):
    q_c(path) = p_c(path) exp(sum_t psi_t(phi_t)) / Z_c
    KL(q_c || p_c) = E_q[sum psi] - log Z_c        (exact, >= 0)
    recon_c       = sum_t E_{q_t}[log p(y_t | phi_t)]  (tied subdiv emission)
mixture over the discrete rate symmetry class handled as in archain
(categorical head, exact expectation + KL). Deployment is label-free: the
potentials are functions of x only; decode = Viterbi on the same quadrature.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..constants import TWO_PI
from . import archain

N_GRID = 128
N_HARM = 12


class SmoothVBPM(archain.VBPM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.emission.recon == "tied"
        assert self.rate.posterior == "categorical" and self.rate.resid == 0.0
        d_model = self.rate_head.in_features
        self.psi_head = nn.Linear(d_model, 2 * N_HARM)
        nn.init.zeros_(self.psi_head.weight)
        nn.init.zeros_(self.psi_head.bias)
        grid = torch.arange(N_GRID) * (TWO_PI / N_GRID)
        j = torch.arange(1, N_HARM + 1, dtype=torch.float32)
        basis = torch.cat([torch.cos(grid[:, None] * j),
                           torch.sin(grid[:, None] * j)], dim=1)
        self.register_buffer("grid", grid)
        self.register_buffer("psi_basis", basis)
        diff = grid[None, :, None] - grid[None, None, :] \
            + self.rates[:, None, None]
        logk = self.walk.kappa_physical * (torch.cos(diff) - 1.0)
        logk = logk - torch.logsumexp(logk, dim=2, keepdim=True)
        self.register_buffer("log_k", logk)
        self.register_buffer("k_lin", logk.exp())

    def _psi(self, feats, mask):
        coef = self.psi_head(feats)
        return (coef @ self.psi_basis.T) * mask[..., None]

    def _grid_loglik(self, cls, mask):
        e_db, e_bt = self.tied_logits(self.grid)
        is_db = (cls == 2).float()[..., None]
        is_bt = (cls >= 1).float()[..., None]
        ll = (is_db * nn.functional.logsigmoid(e_db)
              + (1.0 - is_db) * nn.functional.logsigmoid(-e_db)
              + is_bt * nn.functional.logsigmoid(e_bt)
              + (1.0 - is_bt) * nn.functional.logsigmoid(-e_bt))
        return ll * mask[..., None]

    def _smooth(self, psi):
        """Forward-backward on the quadrature grid, all components at once.

        psi [B,T,N] -> (q [B,C,T,N], log_z [B,C]).  Linear-domain recursions
        with per-step max subtraction; a log-floor keeps zero kernel mass from
        producing NaN gradients (those grid points are unreachable under the
        walk, which is the physics, not an artifact).
        """
        B, T, N = psi.shape
        C = self.rates.shape[0]
        floor = 1e-38
        alpha = (psi[:, 0] - math.log(N)).unsqueeze(1).expand(B, C, N)
        alphas = [alpha]
        for t in range(1, T):
            m = alpha.max(dim=-1, keepdim=True).values
            a = torch.exp(alpha - m)
            prop = torch.einsum("bcm,cmn->bcn", a, self.k_lin)
            alpha = torch.log(prop.clamp_min(floor)) + m + psi[:, t].unsqueeze(1)
            alphas.append(alpha)
        log_z = torch.logsumexp(alpha, dim=-1)
        beta = torch.zeros_like(alpha)
        betas = [beta]
        for t in range(T - 1, 0, -1):
            b = beta + psi[:, t].unsqueeze(1)
            m = b.max(dim=-1, keepdim=True).values
            prop = torch.einsum("cmn,bcn->bcm", self.k_lin, torch.exp(b - m))
            beta = torch.log(prop.clamp_min(floor)) + m
            betas.append(beta)
        betas.reverse()
        post = torch.stack(alphas, dim=2) + torch.stack(betas, dim=2)
        q = torch.softmax(post, dim=-1)
        return q, log_z

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0,
                raw=None):
        assert raw is not None
        cls = raw["cls"].to(h.device)
        feats = self.encoder.features(h, mask)
        pooled = self._pooled(feats, mask)
        psi = self._psi(feats, mask)
        q, log_z = self._smooth(psi)
        ll = self._grid_loglik(cls, mask)
        recon_c = torch.einsum("bctn,btn->bc", q, ll)
        e_psi = torch.einsum("bctn,btn->bc", q, psi)
        kl_c = e_psi - log_z
        logits = self.rate_head(pooled)
        log_prior = self.rate_log_prior[None].expand_as(logits)
        log_qc = torch.log_softmax(logits + log_prior, dim=-1)
        qc = log_qc.exp()
        kl_mix = (qc * (log_qc - log_prior)).sum(-1)
        recon = (qc * recon_c).sum(-1)
        kl = (qc * kl_c).sum(-1) + kl_mix
        best = log_qc.argmax(-1)
        return {"elbo": recon - kl, "recon": recon, "kl": kl,
                "kl_mix": kl_mix.mean(), "kl0": kl_c.mean(),
                "rate": self.rates[best],
                "recon_c": recon_c, "kl_c": kl_c, "log_qc": log_qc,
                "q": q, "log_z": log_z}

    @torch.no_grad()
    def infer_phase(self, h, mask=None, decode: str = "mean"):
        """Label-free deployment: the potentials read x only.

        decode='mean' -- unwrapped per-frame circular mean of the exact
        smoother marginal (continuous, no grid quantization); 'viterbi' --
        MAP grid path, kept for comparison.
        """
        assert not self.training
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        feats = self.encoder.features(h, mask)
        pooled = self._pooled(feats, mask)
        psi = self._psi(feats, mask)
        logits = self.rate_head(pooled) + self.rate_log_prior[None]
        best = torch.log_softmax(logits, dim=-1).argmax(-1)
        B, T, N = psi.shape
        if decode == "mean":
            q, _ = self._smooth(psi)
            qb = q[torch.arange(B), best]
            re = (qb * torch.cos(self.grid)).sum(-1)
            im = (qb * torch.sin(self.grid)).sum(-1)
            ang = torch.atan2(im, re)
            d = ang[:, 1:] - ang[:, :-1]
            d = torch.remainder(d + math.pi, TWO_PI) - math.pi
            return torch.cat([ang[:, :1],
                              ang[:, :1] + torch.cumsum(d, 1)], dim=1)
        out = psi.new_zeros(B, T)
        for b in range(B):
            logk = self.log_k[best[b]]
            v = psi[b, 0] - math.log(N)
            ptrs = []
            for t in range(1, T):
                s = v[:, None] + logk
                v, p = s.max(dim=0)
                v = v + psi[b, t]
                ptrs.append(p)
            idx = int(v.argmax())
            path = [idx]
            for p in reversed(ptrs):
                idx = int(p[idx])
                path.append(idx)
            path.reverse()
            ang = self.grid[torch.tensor(path, device=psi.device)]
            d = ang[1:] - ang[:-1]
            d = torch.remainder(d + math.pi, TWO_PI) - math.pi
            out[b] = torch.cat([ang[:1], ang[0] + torch.cumsum(d, 0)])
        return out


def build_model(cfg, input_dim: int) -> SmoothVBPM:
    from .base import common_kwargs
    from ..specs import ChainSpec, RateSpec
    return SmoothVBPM(input_dim,
                      rate=RateSpec(grid=cfg.chain_rate_grid, lo=cfg.ar_rate_lo,
                                    hi=cfg.ar_rate_hi, posterior="categorical",
                                    resid=0.0),
                      chain=ChainSpec(stride=1, phase_kernel="vonmises"),
                      **common_kwargs(cfg))
