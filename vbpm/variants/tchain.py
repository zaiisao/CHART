"""Bar phase AND tempo in the chain state, marginalised exactly.

rate_grid and schain both enumerate ONE tempo for the whole window, so neither can
follow a performance that breathes -- on rubato they saturate the constant-rate ceiling
and stop. Here the state is a pair,

    z_t = (phase bin k, tempo bin m)

and forward-backward sums over tempo PATHS rather than picking a tempo. Nothing descends
a gradient into tempo, which is the property that made enumeration work; what changes is
that the enumeration is over trajectories instead of constants.

The transition factorises, which is what keeps it affordable:

    p(k', m' | k, m) = vM(theta_k' - theta_k - rate_m ; kappa)
                       * N(log rate_m' - log rate_m ; sigma^2)

The phase part depends on (k' - k) only, so for each tempo it is a circular shift; the
tempo part is a small band around the diagonal, since tempo moves slowly. Cost per frame
is M phase-shifts plus one M x M mix, not a dense (K M) x (K M) matrix.

sigma is the walk scale -- the per-song quantity the corpus says spans 20x and that a
single constant cannot serve. Here it finally has something to act on: the width of the
tempo kernel.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .base import epoch_note, objective, on_epoch, optimizer  # noqa: F401
from ..constants import TWO_PI
from ..nets import Encoder
from ..specs import EmissionSpec, WalkSpec
from ..vonmises import log_i0


DEFAULTS = {"chain_bins": 96, "tempo_bins": 24, "tempo_lo": 0.030, "tempo_hi": 0.180,
            "chain_sigma": 0.01, "tempo_band": 3,
            "phase_kernel": "vonmises", "tempo_kernel": "gauss",
            "tempo_revert": True, "phase_nu": 4.0, "use_graphs": False}


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


class _SweepLoop(nn.Module):
    """The whole forward-backward sweep as one callable, so it graphs without aliasing.

    make_graphed_callables reuses static output buffers per replay, so graphing a
    single frame and calling it T times makes every frame alias the last. Capturing
    the entire sweep keeps one buffer set for the whole recursion.
    """

    def forward(self, psi, mask, ph_f, te, a0):
        """Run both passes; returns the pair marginal in logs and the log evidence."""
        b, t, k = psi.shape
        m = te.shape[-1]
        a = a0
        alphas = []
        logZ = torch.zeros(b, device=psi.device, dtype=psi.dtype)
        for i in range(t):
            if i > 0:
                moved = torch.einsum("bkm,bmkj->bjm", a, ph_f)
                a = torch.einsum("bkm,bmn->bkn", moved, te)
            live = mask[:, i][:, None, None]
            a = a * (live * psi[:, i][:, :, None] + (1.0 - live))
            norm = a.reshape(b, -1).sum(-1).clamp(min=1e-30)
            a = a / norm[:, None, None]
            logZ = logZ + torch.log(norm)
            alphas.append(a)

        beta = torch.ones(b, k, m, device=psi.device, dtype=psi.dtype) / (k * m)
        betas = [None] * t
        betas[t - 1] = beta
        for i in range(t - 1, 0, -1):
            live = mask[:, i][:, None, None]
            msg = beta * (live * psi[:, i][:, :, None] + (1.0 - live))
            mixed = torch.einsum("bkm,bnm->bkn", msg, te)
            beta = torch.einsum("bjm,bmkj->bkm", mixed, ph_f)
            beta = beta / beta.reshape(b, -1).sum(-1).clamp(min=1e-30)[:, None, None]
            betas[i - 1] = beta

        g = torch.stack(alphas, 1) * torch.stack(betas, 1)
        g = g / g.reshape(b, t, -1).sum(-1).clamp(min=1e-30)[:, :, None, None]
        return torch.log(g.clamp(min=1e-30)), logZ


class TempoChainVBPM(nn.Module):
    """The tutorial's generative model with (phase, tempo) as the chain state."""

    wants_raw = True
    emission_net = None

    def __init__(self, input_dim: int, d_model: int = 128, bins: int = 96,
                 tempo_bins: int = 24, tempo_lo: float = 0.030, tempo_hi: float = 0.180,
                 sigma: float = 0.01, band: int = 3,
                 phase_kernel: str = "vonmises", tempo_kernel: str = "gauss",
                 tempo_revert: bool = True, phase_nu: float = 4.0,
                 use_graphs: bool = False, harmonics: int = 6,
                 emission: EmissionSpec | str = "triangle",
                 walk: WalkSpec | None = None, encoder_pe: bool = False):
        super().__init__()
        emission = EmissionSpec.coerce(emission)
        self.walk = walk or WalkSpec()
        self.bins, self.tempo_bins = int(bins), int(tempo_bins)
        self.sigma, self.band = float(sigma), int(band)
        self.phase_kernel, self.tempo_kernel = phase_kernel, tempo_kernel
        self.tempo_revert = bool(tempo_revert)
        self.phase_nu = float(phase_nu)
        self.emission_kind = emission.kind
        self.bump_kappa = float(emission.bump_kappa)

        self.encoder = Encoder(input_dim, d_model,
                               kappa_physical=self.walk.kappa_physical,
                               use_pe=encoder_pe)
        self.psi_head = nn.Linear(d_model, self.bins)
        nn.init.zeros_(self.psi_head.weight)
        nn.init.zeros_(self.psi_head.bias)
        self.emission_a = nn.Parameter(torch.tensor(-3.0))
        self.emission_b_raw = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("emission_b_floor", torch.tensor(0.0))

        self.harmonics = int(harmonics)
        coef = torch.zeros(3, 2 * self.harmonics)
        coef[2, 0] = 1.0
        self.emission_coef = nn.Parameter(coef)
        self.emission_bias = nn.Parameter(torch.tensor([0.0, -2.5, -3.6]))

        self.kappa_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.kappa_head.weight)
        nn.init.constant_(self.kappa_head.bias, math.log(self.walk.kappa_physical))
        self.sigma_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.constant_(self.sigma_head.bias, math.log(self.sigma))
        self.phi0_head = nn.Linear(d_model, self.bins)
        nn.init.zeros_(self.phi0_head.weight)
        nn.init.zeros_(self.phi0_head.bias)

        rates = torch.exp(torch.linspace(math.log(tempo_lo), math.log(tempo_hi),
                                         self.tempo_bins))
        self.register_buffer("rates", rates)
        self.register_buffer("theta", torch.arange(self.bins) * (TWO_PI / self.bins))
        j = torch.arange(1, self.harmonics + 1, dtype=torch.float32)
        angle = self.theta[:, None] * j[None, :]
        self.register_buffer("fourier", torch.cat([angle.cos(), angle.sin()], dim=1))
        self._sweep_loop = _SweepLoop()
        self.use_graphs = bool(use_graphs)
        self._graphs: dict = {}
        z = (torch.log(rates) - self.walk.tempo_mu) / self.walk.tempo_sigma
        lp = -0.5 * z ** 2
        self.register_buffer("tempo_log_prior", lp - torch.logsumexp(lp, 0))

    @property
    def emission_b(self):
        """Amplitude, positive by softplus, never below the scheduled floor."""
        return self.emission_b_floor + nn.functional.softplus(self.emission_b_raw)

    @property
    def deployed_net(self):
        """The inference network read at test time; controls assert ITS target-blindness."""
        return self.encoder

    def class_logits_at_bins(self):
        """[K, 3]: the tutorial's emission p(b | phi) over (non-beat, beat, downbeat).

        Each class is a truncated Fourier series on the circle, so the reader stays dumb
        -- three biases and 2H coefficients per class -- and the beat class has to find
        its own subdivisions rather than be told the meter. Only the downbeat class is
        seeded, with the first cosine, because phi = 0 IS the downbeat landmark by the
        coordinate's definition and not by an assumption about how many beats fill a bar.
        """
        return self.emission_bias[None, :] + self.fourier @ self.emission_coef.T

    def emission_logits_at_bins(self):
        """log-odds of a downbeat at each phase bin, read off the three-way emission."""
        lp = torch.log_softmax(self.class_logits_at_bins(), dim=-1)
        return lp[:, 2] - torch.logsumexp(lp[:, :2], dim=-1)

    def emission_logits(self, phi, mask=None):
        """The same log-odds read at arbitrary phases rather than at bin centres."""
        j = torch.arange(1, self.harmonics + 1, device=phi.device, dtype=phi.dtype)
        angle = phi[..., None] * j
        basis = torch.cat([angle.cos(), angle.sin()], dim=-1)
        logits = self.emission_bias + basis @ self.emission_coef.T
        lp = torch.log_softmax(logits, dim=-1)
        return lp[..., 2] - torch.logsumexp(lp[..., :2], dim=-1)

    def log_phase_shift(self, kappa):
        """[M, K, K]: for tempo m, the phase advances by rate_m and jitters.

        The jitter's shape is a modelling choice, and the corpus prefers a heavy tail:
        fitted on train-fold annotations, per-song wrapped Cauchy beats von Mises on the
        phase innovation by 0.77 nats per bar. vonmises is the shape the tutorial's
        section 9.9 recommends; cauchy is the corpus's answer.
        """
        diff = self.theta[None, :] - self.theta[:, None]
        d = (diff[None] - self.rates[:, None, None])[None]
        kappa = kappa[:, None, None, None]
        if self.phase_kernel == "student":
            # wrapped Student-t: nu dials the tail between Cauchy (nu=1) and the
            # Gaussian/von Mises limit (nu -> inf). The scale is matched to vM(kappa),
            # so nu alone controls how hard the chain prunes a competing hypothesis.
            scale = kappa.rsqrt()
            chord = 2.0 * torch.sin(d / 2.0)
            nu = float(self.phase_nu)
            lt = -0.5 * (nu + 1.0) * torch.log1p((chord / scale) ** 2 / nu)
        elif self.phase_kernel == "cauchy":
            # match the CORE width, not the mean resultant: a wrapped Cauchy with the
            # same resultant as vM(kappa) has a core ~40x narrower at kappa=383, which
            # is a spike rather than a heavier-tailed version of the same kernel.
            gamma = kappa.rsqrt()
            rho = torch.exp(-gamma).clamp(1e-4, 1.0 - 1e-6)
            denom = (1.0 - rho) ** 2 + 4.0 * rho * torch.sin(d / 2.0) ** 2
            lt = torch.log1p(-rho ** 2) - torch.log(denom) - math.log(TWO_PI)
        else:
            lt = kappa * torch.cos(d) - math.log(TWO_PI) - log_i0(kappa)
        return lt - torch.logsumexp(lt, dim=3, keepdim=True)

    def log_tempo_kernel(self, sigma=None):
        """[M, M]: how log tempo may move in one frame, banded to +-band.

        A driftless walk has a UNIFORM stationary distribution over the grid, so it
        expresses no preference for plausible tempos and the only mention of tempo level
        is the initial state -- one frame against a whole window of likelihood, which is
        what lets a run drift to a 20 s bar and stay there. Pulling the step toward the
        corpus mean makes the level priced at every frame instead:

            log r' ~ N( mu + a (log r - mu),  sigma^2 ),   a = sqrt(1 - (sigma/s_prior)^2)

        so the stationary spread is the corpus spread s_prior by construction, and the
        step is locally identical to the old walk. The corpus supports the form
        independently: tempo increments are negatively autocorrelated (-0.26 at lag 1,
        -0.11 at lags 2-3), which a driftless walk cannot produce.
        """
        lr = torch.log(self.rates)
        sigma = sigma[:, None, None]
        if self.tempo_revert:
            s_prior = float(self.walk.tempo_sigma)
            a = (1.0 - (sigma / s_prior) ** 2).clamp(min=0.0).sqrt()
            mu = float(self.walk.tempo_mu)
            target = mu + a * (lr[None, :, None] - mu)
            d = lr[None, None, :] - target
        else:
            d = (lr[None, :] - lr[:, None])[None]
        if self.tempo_kernel == "cauchy":
            lt = -torch.log1p((d / sigma) ** 2)
        elif self.tempo_kernel == "laplace":
            lt = -d.abs() / sigma
        else:
            lt = -0.5 * (d / sigma) ** 2
        if self.band > 0:
            idx = torch.arange(self.tempo_bins, device=lr.device)
            far = (idx[None, :] - idx[:, None]).abs() > self.band
            lt = lt.masked_fill(far[None], -1e9)
        return lt - torch.logsumexp(lt, dim=2, keepdim=True)

    def _sweep(self, b, t, k):
        """The sweep callable, CUDA-graphed per (batch, length) shape when enabled."""
        if not self.use_graphs or not torch.is_grad_enabled() or not self.rates.is_cuda:
            return self._sweep_loop
        key = (b, t, k, self.tempo_bins)
        if key not in self._graphs:
            m, dev = self.tempo_bins, self.rates.device
            kap = torch.full((b,), float(self.walk.kappa_physical), device=dev)
            sig = torch.full((b,), float(self.sigma), device=dev)
            sa = (torch.rand(b, t, k, device=dev).add(0.1).requires_grad_(),
                  torch.ones(b, t, device=dev),
                  self.log_phase_shift(kap).exp().detach(),
                  self.log_tempo_kernel(sig).exp().detach(),
                  torch.rand(b, k, m, device=dev).add(0.1).requires_grad_())
            self._graphs[key] = _graph(self._sweep_loop, sa, [sa[0], sa[4]])
        return self._graphs[key]

    def prior_params(self, feats, mask):
        """(kappa, sigma, log phi0) for each song: the prior's own heads on the audio.

        Option T-b of the phase question and its tempo counterpart. The heads are zeroed
        at init and their biases carry the corpus constants, so an untrained model is
        exactly the constant-prior model and every departure from it is learned. The
        initial phase head starts uniform, which is Option I-1.
        """
        w = mask[..., None]
        pooled = (feats * w).sum(1) / w.sum(1).clamp(min=1.0)
        kappa = self.kappa_head(pooled)[:, 0].clamp(-8.0, 12.0).exp()
        sigma = self.sigma_head(pooled)[:, 0].clamp(-8.0, 2.0).exp()
        return kappa, sigma, torch.log_softmax(self.phi0_head(pooled), dim=-1)

    def forward_backward(self, log_psi, mask, kappa, sigma, log_phi0):
        """Scaled linear-domain recursion, equivalent to log-space message passing.

        alpha is renormalised every frame anyway, so carrying it in the linear domain
        and accumulating log of the normaliser is equivalent to log-space message
        passing, and replaces two logsumexp reductions per frame with two matmuls.
        The recursion is launch-bound rather than arithmetic-bound, so this is where
        the wall clock goes -- and why the sweep is worth CUDA-graphing.
        """
        b, t, k = log_psi.shape
        ph_f = self.log_phase_shift(kappa).exp()               # [B, M, k, k'] per tempo
        te = self.log_tempo_kernel(sigma).exp()                # [B, M, M]
        a0 = (log_phi0.exp()[:, :, None]
              * self.tempo_log_prior.exp()[None, None, :]).contiguous()
        return self._sweep(b, t, k)(log_psi.exp(), mask, ph_f, te, a0)

    def marginals(self, h, mask, sigma=None):
        """(log psi, log gamma, log Z, kappa): the encoder pass and one sweep."""
        feats = self.encoder.features(h, mask)
        log_psi = torch.log_softmax(self.psi_head(feats), dim=-1)
        kappa, sigma_x, log_phi0 = self.prior_params(feats, mask)
        if sigma is not None:
            sigma_x = torch.full_like(sigma_x, float(sigma))
        log_g, logZ = self.forward_backward(log_psi, mask, kappa, sigma_x, log_phi0)
        return log_psi, log_g, logZ, kappa

    def mean_path(self, gamma_phase):
        """Circular mean of the phase marginals, unwrapped: the discrete analogue of mu(x)."""
        re = (gamma_phase * torch.cos(self.theta)).sum(-1)
        im = (gamma_phase * torch.sin(self.theta)).sum(-1)
        wrapped = torch.atan2(im, re)
        step = torch.diff(wrapped, dim=-1)
        step = torch.atan2(torch.sin(step), torch.cos(step))
        path = torch.cat([wrapped[:, :1], wrapped[:, :1] + torch.cumsum(step, -1)], -1)
        return path, torch.sqrt(re ** 2 + im ** 2 + 1e-12)

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0, raw=None):
        """One ELBO evaluation: exact expectations under the chain, no sampling."""
        assert raw is not None, "the three-way emission needs the batch's class targets"
        cls = raw["cls"].to(h.device)
        log_psi, log_g, logZ, kappa = self.marginals(h, mask)
        gamma = log_g.exp()
        gamma_phase = gamma.sum(-1)
        gamma_tempo = gamma.sum(-2)

        lp = torch.log_softmax(self.class_logits_at_bins(), dim=-1)
        ll = lp[:, cls].permute(1, 2, 0)
        recon = ((gamma_phase * ll).sum(-1) * mask).sum(-1)
        kl = ((gamma_phase * log_psi).sum(-1) * mask).sum(-1) - logZ

        phi, resultant = self.mean_path(gamma_phase)
        rate = (gamma_tempo * self.rates).sum(-1)
        return {"elbo": recon - kl, "recon": recon, "kl": kl, "phi": phi,
                "kappa": resultant * kappa[:, None],
                "resultant": resultant.mean(1), "rate": rate,
                "rate_sd": (gamma_tempo * (self.rates[None, None] - rate[..., None]) ** 2
                            ).sum(-1).sqrt().mean(1)}

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        """Deployment: the mean phase path from x alone."""
        assert not self.training, "deployment path must run in eval mode"
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        _psi, log_g, _z, _k = self.marginals(h, mask)
        return self.mean_path(log_g.exp().sum(-1))[0]

    @torch.no_grad()
    def emission_probs(self, h, mask=None):
        """Alternative D: the emission read through the posterior's phase marginal."""
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        _psi, log_g, _z, _k = self.marginals(h, mask)
        return (log_g.exp().sum(-1) * torch.sigmoid(self.emission_logits_at_bins())).sum(-1)


def build_model(cfg, input_dim: int) -> TempoChainVBPM:
    """The hooks entry point: one TempoChainVBPM from a config."""
    from .base import common_kwargs
    kw = common_kwargs(cfg)
    kw.pop("phase_init", None)
    return TempoChainVBPM(input_dim, bins=cfg.chain_bins, tempo_bins=cfg.tempo_bins,
                          tempo_lo=cfg.tempo_lo, tempo_hi=cfg.tempo_hi,
                          sigma=cfg.chain_sigma, band=cfg.tempo_band,
                          phase_kernel=cfg.phase_kernel, tempo_kernel=cfg.tempo_kernel,
                          tempo_revert=cfg.tempo_revert, phase_nu=cfg.phase_nu,
                          use_graphs=cfg.use_graphs,
                          **kw)
