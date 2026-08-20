"""Continuous phase, discrete tempo: von Mises messages instead of a phase grid.

tchain represents phase as 96 bins. The circle has a conjugate family that makes
that unnecessary: von Mises is closed under BOTH operations the chain needs.

    rotation:        vM(mu, kappa) advanced by v  =  vM(mu + v, kappa)
    evidence:        vM(mu1, k1) * vM(mu2, k2)    propto vM of the vector sum

So the phase belief stays exactly von Mises through predict and update, with no
grid and no unwrapping -- the winding number never has to be chosen. Tempo stays
discrete, because the octave ambiguity is genuinely multimodal and a unimodal
family cannot hold mass at v and 2v at once.

Two approximations, both named:
  process noise -- convolving two von Mises is not von Mises; matched by the
      standard resultant product A(kappa) = A(k1) A(k2), exact in the resultant.
  tempo mixing -- a tempo step makes each bin a MIXTURE over source bins; it is
      collapsed back to one von Mises by circular moment matching.
This is an assumed-density filter: the family is continuous, the recursion is
approximate. tchain is the reverse -- discrete family, exact recursion.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..constants import TWO_PI
from ..nets import Encoder
from ..specs import WalkSpec

DEFAULTS = {"tempo_bins": 24, "tempo_lo": 0.030, "tempo_hi": 0.180,
            "chain_sigma": 0.03, "tempo_band": 3, "tempo_revert": True,
            "kappa_process": 2000.0, "quad_nodes": 256, "kappa_max": 500.0, "use_graphs": False,
            "psi_warmup": 40}


def a_ratio(kappa):
    """A(kappa) = I1(kappa)/I0(kappa), the resultant length, exact via i1e/i0e."""
    k = kappa.clamp(min=1e-8)
    return (torch.special.i1e(k) / torch.special.i0e(k)).clamp(1e-8, 1 - 1e-8)


KAPPA_CEIL = 500.0
R_CEIL = float(a_ratio(torch.tensor(KAPPA_CEIL)))


def inv_a(r, steps: int = 3):
    """Invert A: resultant -> kappa. Banerjee start, then Newton on the exact A.

    r is capped at A(KAPPA_CEIL) because the inverse is genuinely stiff as r -> 1
    (dkappa/dr ~ 2 kappa^2), and an uncapped Newton step divides by a vanishing
    A'(kappa). kappa = 500 is a phase sd of 0.045 rad, far finer than the metric.
    """
    r = r.clamp(1e-8, R_CEIL)
    k = (r * (2.0 - r ** 2) / (1.0 - r ** 2).clamp(min=1e-8)).clamp(1e-4, KAPPA_CEIL)
    for _ in range(steps):
        a = a_ratio(k)
        denom = (1.0 - a ** 2 - a / k).clamp(min=1e-4)
        k = (k - (a - r) / denom).clamp(1e-4, KAPPA_CEIL)
    return k


def _graph(loop, sample_args, grad_inputs):
    """Capture loop over sample_args, warming cuBLAS on those exact shapes first.

    Capture fails at some batch shapes because the first matmul of that shape
    allocates a cuBLAS workspace, which is not permitted mid-capture. Running the
    real shapes on a side stream first forces the allocation to happen early.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            out = loop(*sample_args)
            torch.autograd.backward(sum(o.sum() for o in out), inputs=grad_inputs)
            for g in grad_inputs:
                g.grad = None
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.make_graphed_callables(loop, sample_args, num_warmup_iters=11)


class _PredictStep(nn.Module):
    """One prior step: rotate by tempo, widen, then mix across the tempo grid."""

    def forward(self, mu, kappa, logw, rates, log_te, te_exp, a_proc):
        """Advance the belief one frame under the prior alone."""
        mu = mu + rates
        kappa = inv_a(a_ratio(kappa) * a_proc)
        ar = a_ratio(kappa)
        re = logw.exp()[..., None] * torch.stack([ar * torch.cos(mu),
                                                  ar * torch.sin(mu)], -1)
        w_new = torch.logsumexp(logw[:, :, None] + log_te[None], dim=1)
        mix = torch.einsum("bmk,mn->bnk", re, te_exp)
        sq = (mix ** 2).sum(-1)
        r = torch.sqrt(sq + 1e-12) / w_new.exp().clamp(min=1e-12)
        live = sq > 1e-20
        mu = torch.atan2(torch.where(live, mix[..., 1], torch.zeros_like(mu)),
                         torch.where(live, mix[..., 0], torch.ones_like(mu)))
        return mu, inv_a(r), w_new - torch.logsumexp(w_new, -1, keepdim=True)


class _UpdateStep(nn.Module):
    """One evidence step: multiply the belief by this frame's von Mises potential."""

    def __init__(self, kappa_max: float):
        super().__init__()
        self.kappa_max = float(kappa_max)

    def forward(self, mu, kappa, logw, mu_t, k_psi_t, m_t):
        """Fold in frame evidence; returns the new belief and this frame's log scale."""
        k_t = k_psi_t * m_t
        x = kappa * torch.cos(mu) + k_t * torch.cos(mu_t)
        y = kappa * torch.sin(mu) + k_t * torch.sin(mu_t)
        k_post = torch.sqrt(x ** 2 + y ** 2 + 1e-12).clamp(max=self.kappa_max)
        inc = (torch.log(torch.special.i0e(k_post)) + k_post
               - torch.log(torch.special.i0e(kappa)) - kappa
               - torch.log(torch.special.i0e(k_t)) - k_t
               - math.log(TWO_PI))
        logw = logw + inc * m_t
        step = torch.logsumexp(logw, -1)
        logw = logw - step[:, None]
        mu = torch.where(m_t > 0, torch.atan2(y, x), mu)
        kappa = torch.where(m_t > 0, k_post, kappa)
        return mu, kappa, logw, step


class _FilterLoop(nn.Module):
    """The whole forward recursion as one callable, so it graphs without aliasing.

    make_graphed_callables reuses static output buffers per replay, so graphing a
    single frame and calling it T times makes every frame alias the last. Capturing
    the entire loop keeps one buffer set for the whole recursion.
    """

    def __init__(self, predict, update):
        super().__init__()
        self.predict, self.update = predict, update

    def forward(self, mu_psi, kappa_psi, mask, rates, log_te, te_exp, a_proc, logw0):
        """Run every frame; returns stacked beliefs and the running log scale."""
        b, t = mu_psi.shape
        mu = torch.zeros_like(logw0)
        kappa = torch.full_like(logw0, 1e-4)
        logw = logw0
        mus, kappas, logws = [], [], []
        logZ = torch.zeros(b, device=mu_psi.device, dtype=mu_psi.dtype)
        for i in range(t):
            if i > 0:
                mu, kappa, logw = self.predict(mu, kappa, logw, rates, log_te,
                                               te_exp, a_proc)
            mu, kappa, logw, step = self.update(mu, kappa, logw, mu_psi[:, i:i + 1],
                                                kappa_psi[:, i:i + 1], mask[:, i:i + 1])
            logZ = logZ + step * mask[:, i]
            mus.append(mu)
            kappas.append(kappa)
            logws.append(logw)
        return (torch.stack(mus, 1), torch.stack(kappas, 1),
                torch.stack(logws, 1), logZ)


class VonMisesChain(nn.Module):
    """Bar-pointer posterior with continuous von Mises phase and a discrete tempo grid."""

    def __init__(self, input_dim: int, tempo_bins: int = 24, tempo_lo: float = 0.030,
                 tempo_hi: float = 0.180, sigma: float = 0.03, band: int = 3,
                 tempo_revert: bool = True, kappa_process: float = 2000.0,
                 quad_nodes: int = 256, kappa_max: float = KAPPA_CEIL, use_graphs: bool = False,
                 walk: WalkSpec | None = None, d_model: int = 128, **kw):
        super().__init__()
        self.walk = walk or WalkSpec()
        self.tempo_bins, self.sigma, self.band = int(tempo_bins), float(sigma), int(band)
        self.tempo_revert = bool(tempo_revert)
        self.kappa_process = float(kappa_process)
        self.kappa_max = float(kappa_max)

        self.encoder = Encoder(input_dim, d_model=d_model, **kw)
        self.psi_head = nn.Linear(d_model, 3)
        nn.init.zeros_(self.psi_head.weight)
        with torch.no_grad():
            # (cos, sin) = (1, 0) keeps atan2 differentiable at init. The kappa
            # channel must start NEAR-uniform but not saturated: softplus(0)=0.69
            # pins every frame to one direction so the phase never advances, while
            # softplus(-10) has gradient 5e-5 and never learns to be informative.
            self.psi_head.bias.copy_(torch.tensor([1.0, 0.0, -3.0]))

        self.emission_a = nn.Parameter(torch.tensor(-3.0))
        self.emission_b_raw = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("emission_b_floor", torch.tensor(0.0))

        rates = torch.exp(torch.linspace(math.log(tempo_lo), math.log(tempo_hi),
                                         self.tempo_bins))
        self.register_buffer("rates", rates)
        z = (torch.log(rates) - self.walk.tempo_mu) / self.walk.tempo_sigma
        lp = -0.5 * z ** 2
        self.register_buffer("tempo_log_prior", lp - torch.logsumexp(lp, 0))
        nodes = torch.arange(quad_nodes) * (TWO_PI / quad_nodes)
        self.register_buffer("nodes", nodes)
        self.register_buffer("a_proc",
                             a_ratio(torch.tensor(float(kappa_process))))
        self._filter_loop = _FilterLoop(_PredictStep(), _UpdateStep(self.kappa_max))
        self.use_graphs = bool(use_graphs)
        self.kappa_ceiling = float(kappa_max)
        self.kappa_target = float(kappa_max)
        self._graphs: dict = {}

    @property
    def emission_b(self):
        """Amplitude of the emission bump, positive by softplus."""
        return self.emission_b_floor + nn.functional.softplus(self.emission_b_raw)

    def psi(self, h, mask):
        """Per-frame von Mises potential (mu, kappa) and its log scale.

        kappa is capped by a scheduled ceiling. A constant-direction potential of
        moderate strength is a STABLE fixed point: the belief's concentration settles
        where the evidence pull-back kappa_psi/kappa exactly cancels the prior's
        rotation, so the phase never advances (measured 0.0297 against a rate of
        0.0300). Holding the evidence weak early leaves rotation as the only thing
        that can move phase, so that fixed point is not reachable from the start.
        """
        feats = self.encoder.features(h, mask)
        out = self.psi_head(feats)
        mu = torch.atan2(out[..., 1], out[..., 0])
        kappa = nn.functional.softplus(out[..., 2]).clamp(1e-4, float(self.kappa_ceiling))
        return mu, kappa

    def log_tempo_kernel(self):
        """[M, M] banded log transition over the discrete tempo grid."""
        lr = torch.log(self.rates)
        if self.tempo_revert:
            s_prior = float(self.walk.tempo_sigma)
            a = math.sqrt(max(1.0 - min((self.sigma / s_prior) ** 2, 1.0), 0.0))
            mu = float(self.walk.tempo_mu)
            d = lr[None, :] - (mu + a * (lr - mu))[:, None]
        else:
            d = lr[None, :] - lr[:, None]
        lt = -0.5 * (d / self.sigma) ** 2
        if self.band > 0:
            idx = torch.arange(self.tempo_bins, device=lr.device)
            lt = lt.masked_fill((idx[None, :] - idx[:, None]).abs() > self.band, -1e9)
        return lt - torch.logsumexp(lt, dim=1, keepdim=True)

    def emission_logits(self, phi):
        """log-odds of a downbeat at continuous phase phi (triangle bump)."""
        d = torch.atan2(torch.sin(phi), torch.cos(phi)).abs()
        return self.emission_a + self.emission_b * (1.0 - d / math.pi)

    def _loop(self, b, t):
        """The recursion callable, CUDA-graphed per (batch, length) when enabled."""
        if not self.use_graphs or not torch.is_grad_enabled() or not self.rates.is_cuda:
            return self._filter_loop
        key = (b, t, self.tempo_bins)
        if key not in self._graphs:
            m, dev = self.tempo_bins, self.rates.device
            sa = (torch.zeros(b, t, device=dev, requires_grad=True),
                  torch.full((b, t), 1.0, device=dev, requires_grad=True),
                  torch.ones(b, t, device=dev),
                  self.rates[None, :].detach(), self.log_tempo_kernel().detach(),
                  self.log_tempo_kernel().exp().detach(), self.a_proc.detach(),
                  torch.zeros(b, m, device=dev, requires_grad=True))
            self._graphs[key] = _graph(self._filter_loop, sa, [sa[0], sa[1], sa[7]])
        return self._graphs[key]

    def filter(self, mu_psi, kappa_psi, mask):
        """Forward pass: per-tempo von Mises phase belief and log weights."""
        b, t = mu_psi.shape
        log_te = self.log_tempo_kernel()
        logw0 = self.tempo_log_prior[None].expand(b, self.tempo_bins).contiguous()
        return self._loop(b, t)(mu_psi, kappa_psi, mask, self.rates[None, :],
                                log_te, log_te.exp(), self.a_proc, logw0)

    def phase_density(self, mu, kappa, logw):
        """Mixture density over phase at the quadrature nodes: [B, T, Q]."""
        d = torch.cos(self.nodes[None, None, None] - mu[..., None]) * kappa[..., None]
        norm = torch.log(torch.special.i0e(kappa))[..., None] + math.log(TWO_PI)
        lp = d - kappa[..., None] - norm
        return torch.logsumexp(logw[..., None] + lp, dim=2).exp()

    def mean_path(self, mu, kappa, logw):
        """Circular mean of the tempo mixture, unwrapped into a rising path."""
        w = logw.exp() * a_ratio(kappa)
        re = (w * torch.cos(mu)).sum(-1)
        im = (w * torch.sin(mu)).sum(-1)
        wrapped = torch.atan2(im, re)
        step = torch.diff(wrapped, dim=-1)
        step = torch.atan2(torch.sin(step), torch.cos(step))
        path = torch.cat([wrapped[:, :1], wrapped[:, :1] + torch.cumsum(step, -1)], -1)
        return path, torch.sqrt(re ** 2 + im ** 2 + 1e-12)

    def forward(self, h, mask, y, pos_weight: float = 1.0):
        """One ELBO evaluation; the phase expectation is numerical quadrature."""
        mu_psi, kappa_psi = self.psi(h, mask)
        mu, kappa, logw, logZ = self.filter(mu_psi, kappa_psi, mask)
        dens = self.phase_density(mu, kappa, logw) * (TWO_PI / len(self.nodes))

        e = self.emission_logits(self.nodes)[None, None]
        ll = (pos_weight * y[..., None] * -nn.functional.softplus(-e)
              + (1.0 - y)[..., None] * -nn.functional.softplus(e))
        recon = ((dens * ll).sum(-1) * mask).sum(-1)

        cos_e = (logw.exp() * a_ratio(kappa)
                 * torch.cos(mu - mu_psi[..., None])).sum(-1)
        cross = (kappa_psi * cos_e - torch.log(torch.special.i0e(kappa_psi))
                 - kappa_psi - math.log(TWO_PI))
        kl = (cross * mask).sum(-1) - logZ

        phi, resultant = self.mean_path(mu, kappa, logw)
        rate = (logw.exp() * self.rates).sum(-1)
        return {"elbo": recon - kl, "recon": recon, "kl": kl, "phi": phi,
                "kappa": resultant * float(self.walk.kappa_physical),
                "resultant": resultant.mean(1), "rate": rate,
                "rate_sd": (logw.exp() * (self.rates[None, None] - rate[..., None]) ** 2
                            ).sum(-1).sqrt().mean(1)}

    def mean_dir(self, mu, logw):
        """Circular mean direction over the tempo mixture: [B, T]."""
        w = logw.exp()
        return torch.atan2((w * torch.sin(mu)).sum(-1), (w * torch.cos(mu)).sum(-1))

    @property
    def deployed_net(self):
        """The inference network read at test time."""
        return self.encoder

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        """Deployment: the mean phase path from x alone."""
        assert not self.training, "deployment path must run in eval mode"
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        mu_psi, kappa_psi = self.psi(h, mask)
        mu, kappa, logw, _z = self.filter(mu_psi, kappa_psi, mask)
        return self.mean_path(mu, kappa, logw)[0]

    @torch.no_grad()
    def emission_probs(self, h, mask=None):
        """Alternative D: the emission read through the phase marginal."""
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        mu_psi, kappa_psi = self.psi(h, mask)
        mu, kappa, logw, _z = self.filter(mu_psi, kappa_psi, mask)
        dens = self.phase_density(mu, kappa, logw) * (TWO_PI / len(self.nodes))
        return (dens * torch.sigmoid(self.emission_logits(self.nodes))[None, None]).sum(-1)


def on_epoch(model, cfg, epoch: int) -> None:
    """Ramp the potential's concentration ceiling; see VonMisesChain.psi."""
    warm = max(int(getattr(cfg, "psi_warmup", 0)), 0)
    if warm:
        frac = min(1.0, epoch / warm)
        lo = math.log(0.05)
        hi = math.log(model.kappa_target)
        model.kappa_ceiling = math.exp(lo + frac * (hi - lo))


def build_model(cfg, input_dim: int) -> VonMisesChain:
    """Config -> VonMisesChain, mirroring tchain's constructor contract."""
    kw = {k: getattr(cfg, k) for k in ("d_model", "nhead", "layers", "dropout")
          if hasattr(cfg, k)}
    return VonMisesChain(input_dim, tempo_bins=cfg.tempo_bins, tempo_lo=cfg.tempo_lo,
                         tempo_hi=cfg.tempo_hi, sigma=cfg.chain_sigma,
                         band=cfg.tempo_band, tempo_revert=cfg.tempo_revert,
                         kappa_process=cfg.kappa_process, quad_nodes=cfg.quad_nodes,
                         use_graphs=cfg.use_graphs,
                         walk=WalkSpec(), **kw)
