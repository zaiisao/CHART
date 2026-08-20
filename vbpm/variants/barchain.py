"""Smoother with a per-BAR tempo transition inside the TRAINING objective.

Difference from smooth.SmoothVBPM: the rate is no longer a fixed label per mixture
component. It is part of the chain state, and it may change ONLY through a wrapping
phase transition (a downbeat), paying a wrapped-Cauchy cost on log-rate at the corpus
per-bar scale. That is the handcrafted-DBN rule ("no tempo change between beats")
transposed to bars, which is the level this model has.

The rate nodes are QUADRATURE points over a continuous log-rate, not a state tiling:
refining 36 -> 288 nodes leaves F, CMLt and rho unchanged (verified at decode time),
so the represented tempo density is continuous.

q(path | x) = p_prior(path) * exp(sum_t psi_t(phi_t) + head(c_0)) / Z, so
    KL(q || p) = E_q[sum_t psi_t + head] - log Z        (exact, >= 0)
    recon      = sum_t E_{q_t(phi)}[log p(y_t | phi_t)]
Nothing is sampled: C_commit = 0.
"""
from __future__ import annotations
import math
import torch
from . import smoother as smooth
from .archain import (DEFAULTS, epoch_note, objective,  # noqa: F401
                      on_epoch, optimizer)
from ..constants import TWO_PI

GAMMA = 0.0363          # corpus median per-bar |dlog rate|, the Cauchy scale


class BarChainVBPM(smooth.SmoothVBPM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lr = torch.log(self.rates)
        sw = 1.0 / (1.0 + ((lr[None, :] - lr[:, None]) / GAMMA) ** 2)
        self.register_buffer("switch", sw / sw.sum(dim=1, keepdim=True))
        n = torch.arange(self.grid.shape[0])
        wrap = (n[None, :] < n[:, None]).to(self.k_lin.dtype)
        self.register_buffer("k_wrap", self.k_lin * wrap)
        self.register_buffer("k_stay", self.k_lin - self.k_lin * wrap)

    def _fwd(self, a):
        """[B,C,N] -> propagated. Rate may change only through a wrap."""
        stay = torch.einsum("bcm,cmn->bcn", a, self.k_stay)
        wm = torch.einsum("bcm,cmn->bcn", a, self.k_wrap)
        return stay + torch.einsum("bdn,dc->bcn", wm, self.switch)

    def _bwd(self, b):
        stay = torch.einsum("bcn,cmn->bcm", b, self.k_stay)
        sw = torch.einsum("bcn,dc->bdn", b, self.switch)
        return stay + torch.einsum("bdn,dmn->bdm", sw, self.k_wrap)

    def _smooth_joint(self, psi, head):
        B, T, N = psi.shape
        C = self.rates.shape[0]
        FL = 1e-30
        E = psi.exp()
        p0 = torch.softmax(self.rate_log_prior, 0)[None, :, None]
        a = p0 * head.exp()[:, :, None] * E[:, 0][:, None, :] / N
        s = a.sum(dim=(1, 2), keepdim=True).clamp_min(FL)
        logz = s.squeeze(-1).squeeze(-1).log()
        a = a / s
        A = [a]
        for t in range(1, T):
            a = self._fwd(a) * E[:, t][:, None, :]
            s = a.sum(dim=(1, 2), keepdim=True).clamp_min(FL)
            logz = logz + s.squeeze(-1).squeeze(-1).log()
            a = a / s
            A.append(a)
        b = torch.ones_like(a) / (C * N)
        Bl = [b]
        for t in range(T - 1, 0, -1):
            b = self._bwd(b * E[:, t][:, None, :])
            b = b / b.sum(dim=(1, 2), keepdim=True).clamp_min(FL)
            Bl.append(b)
        Bl.reverse()
        post = torch.stack(A, 1) * torch.stack(Bl, 1)               # [B,T,C,N]
        post = post / post.sum(dim=(2, 3), keepdim=True).clamp_min(FL)
        return post, logz

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0, raw=None):
        assert raw is not None
        cls = raw["cls"].to(h.device)
        feats = self.encoder.features(h, mask)
        pooled = self._pooled(feats, mask)
        psi = self._psi(feats, mask)
        head = torch.log_softmax(self.rate_head(pooled), dim=-1)
        post, logz = self._smooth_joint(psi, head)
        qn = post.sum(2)                                            # [B,T,N] phase
        qc = post.sum(3)                                            # [B,T,C] rate
        ll = self._grid_loglik(cls, mask)
        recon = torch.einsum("btn,btn->b", qn, ll)
        e_psi = torch.einsum("btn,btn->b", qn, psi) \
            + torch.einsum("bc,bc->b", qc[:, 0], head)
        kl = e_psi - logz
        g = self.grid
        re = (qn * g.cos()).sum(-1)
        im = (qn * g.sin()).sum(-1)
        ang = torch.atan2(im, re)
        d = torch.remainder(ang[:, 1:] - ang[:, :-1] + math.pi, TWO_PI) - math.pi
        phi = torch.cat([ang[:, :1], ang[:, :1] + torch.cumsum(d, 1)], 1)
        R = (re ** 2 + im ** 2).sqrt().clamp(1e-6, 1 - 1e-6)
        rate = (qc * self.rates[None, None, :]).sum(-1)
        return {"elbo": recon - kl, "recon": recon, "kl": kl,
                "kl_mix": kl.mean() * 0, "kl0": kl.mean() * 0,
                "phi": phi, "kappa": R * (2 - R ** 2) / (1 - R ** 2),
                "rate": rate.mean(-1), "q": post, "rate_traj": rate}

    @torch.no_grad()
    def infer_phase(self, h, mask=None, decode: str = "mean"):
        assert not self.training
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        feats = self.encoder.features(h, mask)
        psi = self._psi(feats, mask)
        head = torch.log_softmax(self.rate_head(self._pooled(feats, mask)), dim=-1)
        post, _ = self._smooth_joint(psi, head)
        qn = post.sum(2)
        g = self.grid
        re = (qn * g.cos()).sum(-1)
        im = (qn * g.sin()).sum(-1)
        ang = torch.atan2(im, re)
        d = torch.remainder(ang[:, 1:] - ang[:, :-1] + math.pi, TWO_PI) - math.pi
        return torch.cat([ang[:, :1], ang[:, :1] + torch.cumsum(d, 1)], 1)


def build_model(cfg, input_dim):
    m = smooth.build_model(cfg, input_dim)
    m.__class__ = BarChainVBPM
    lr = torch.log(m.rates)
    sw = 1.0 / (1.0 + ((lr[None, :] - lr[:, None]) / GAMMA) ** 2)
    m.register_buffer("switch", (sw / sw.sum(dim=1, keepdim=True)).to(m.rates.device))
    n = torch.arange(m.grid.shape[0], device=m.rates.device)
    wrap = (n[None, :] < n[:, None]).to(m.k_lin.dtype)
    m.register_buffer("k_wrap", m.k_lin * wrap)
    m.register_buffer("k_stay", m.k_lin - m.k_lin * wrap)
    return m
