"""Interval emission: the observation is the downbeat TIMES, not a per-frame indicator.

The Bernoulli emission grades a prediction hit-or-miss and never how far off it is, so
the rate axis it induces is corrugated and octave-degenerate (30 single-song runs, zero
retained solutions, endpoints always harmonics). Here the observation is the annotated
downbeat times t_1..t_N inside the window, and the likelihood factorises into two
distance-aware rulers over the phase trajectory:

    placement  phi(t_1) should be 0 mod 2pi                 -> vM(kappa_place)
    interval   r_i = (phi(t_{i+1}) - phi(t_i)) / 2pi = 1     -> Laplace/Huber(b_ratio)

The map (t_1..t_N) -> (phi_1, log r_1..log r_{N-1}) is a bijection while phi increases,
so N coordinates carry N density factors (one vM, N-1 interval) plus the log-Jacobian
sum_i log dotphi(t_i) - sum_i log(2 pi r_i). Scoring a vM at EVERY annotation as well
put 2N-1 factors on N coordinates and left the "density" unnormalised with a
latent-dependent normaliser (measured Z = 0.013, drifting 3.6 nats with the model's own
rate); that is why only the first annotation carries a placement factor here.

dotphi is read from the sampled path over a long baseline, never from a one-frame
difference: at kappa 383 the per-frame jitter sd (0.072) exceeds the bar rate (0.051),
so a one-frame difference measures noise, ~23% of its slopes come out negative, and it
donated 23-35 nats of the octave margin to the 2x harmonic.

Labels enter the loss only. The encoder is unchanged and still reads audio alone.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .base import common_kwargs, epoch_note, objective, on_epoch as _base_on_epoch  # noqa: F401
from .base import optimizer  # noqa: F401
from ..model import TWO_PI, BarPhaseVAE, log_i0, sample_vonmises

DEFAULTS = {"b_ratio": 0.1, "kappa_place": 100.0, "kappa_anneal": "3,300,0.7",
            "sigma_ceil": 0.01, "phase_half": 0, "interval_kind": "laplace"}


def interp_phase(phi, frames_f):
    """phi [B, T] sampled at integer frames -> its interpolation at frames_f [B, N]."""
    length = phi.shape[1]
    f = frames_f.clamp(min=0.0, max=float(length - 1) - 1e-4)
    i0 = f.floor().long()
    i1 = (i0 + 1).clamp(max=length - 1)
    frac = f - i0.to(f.dtype)
    return torch.gather(phi, 1, i0) * (1.0 - frac) + torch.gather(phi, 1, i1) * frac


def path_dotphi(phi, half: int = 25):
    """dphi/dt from the sampled path over +-half frames, so jitter averages out."""
    pad = torch.nn.functional.pad(phi[:, None, :], (half, half), mode="replicate")[:, 0]
    return ((pad[:, 2 * half:] - pad[:, :-2 * half]) / (2.0 * half)).clamp(min=1e-8)


def smooth_phase(phi, half: int):
    """Boxcar the phase before reading it at an annotation; the ramp is locally linear,
    so this is unbiased and cuts the jitter by sqrt(2*half+1)."""
    if half <= 0:
        return phi
    pad = torch.nn.functional.pad(phi[:, None, :], (half, half), mode="replicate")
    return torch.nn.functional.avg_pool1d(pad, 2 * half + 1, stride=1)[:, 0]


def interval_penalty(log_ratio, b_ratio: float, kind: str = "laplace", delta: float = 0.05):
    """-log density of log r. Laplace votes +-1/b at any size; Huber's force is
    proportional to the error inside delta, so its mean survives averaging."""
    if kind == "laplace":
        return log_ratio.abs() / b_ratio + math.log(2.0 * b_ratio)
    size = log_ratio.abs()
    shaped = torch.where(size <= delta, 0.5 * size ** 2 / delta, size - 0.5 * delta)
    return shaped / b_ratio + math.log(2.0 * b_ratio)


def annotation_frames(raw, device):
    """[B, Nmax] annotated downbeat positions in frames, and their 0/1 validity."""
    fps = raw["fps"].numpy()
    t0 = raw["t0"].numpy()
    per_item = [(np.asarray(t, dtype=np.float64) - t0[i]) * fps[i]
                for i, t in enumerate(raw["downbeat_times"])]
    width = max(1, max(len(a) for a in per_item))
    frames = np.zeros((len(per_item), width), dtype=np.float32)
    valid = np.zeros((len(per_item), width), dtype=np.float32)
    for i, a in enumerate(per_item):
        frames[i, :len(a)] = a
        valid[i, :len(a)] = 1.0
    return torch.from_numpy(frames).to(device), torch.from_numpy(valid).to(device)


def interval_loglik(phi, ann_f, ann_valid, kappa_place: float, b_ratio: float,
                    phase_half: int = 0, kind: str = "laplace", phi_place=None):
    """log p(annotation times | phi) [B].

    ``phi_place`` carries the placement factor's phase. It exists because the two
    factors own different degrees of freedom -- placement the rotation, interval the
    rate -- and the split is not automatic: phi_1 = mu0(t_1) + theta, so
    d(kappa cos phi_1)/d log k = -kappa sin(phi_1) * mu0(t_1), and with the first
    annotation 42 frames in that leaks +-476 nats per unit log-rate into the rate
    channel, against the interval ruler's +-110. Measured, it reversed the net gradient
    at k = 1.25 and k = 2.65 -- the harmonics runs actually park on. Detaching the
    ramp inside this factor keeps its gradient to theta (which trains the evidence
    head) and returns the rate to the term that measures it.
    """
    kappa = torch.as_tensor(kappa_place, device=phi.device, dtype=phi.dtype)
    at = interp_phase(smooth_phase(phi, phase_half), ann_f)
    first = ann_valid.cumsum(1).eq(1.0).to(phi.dtype) * ann_valid

    at_place = at if phi_place is None else interp_phase(smooth_phase(phi_place, phase_half), ann_f)
    place = ((kappa * torch.cos(at_place) - math.log(TWO_PI) - log_i0(kappa)) * first).sum(1)
    pair = ann_valid[:, 1:] * ann_valid[:, :-1]
    ratio = ((at[:, 1:] - at[:, :-1]) / TWO_PI).clamp(min=1e-6)
    interval = (-interval_penalty(torch.log(ratio), b_ratio, kind) * pair).sum(1)

    d_at = interp_phase(path_dotphi(phi), ann_f)
    jac = (torch.log(d_at) * ann_valid).sum(1) - (torch.log(TWO_PI * ratio) * pair).sum(1)
    return {"loglik": place + interval + jac, "place": place, "interval": interval}


class IntervalVAE(BarPhaseVAE):
    """BarPhaseVAE with the time/interval observation model in place of the Bernoulli."""

    wants_raw = True

    def __init__(self, input_dim: int, b_ratio: float = 0.1, kappa_place: float = 100.0,
                 phase_half: int = 0, interval_kind: str = "laplace", **kw):
        super().__init__(input_dim, **kw)
        self.b_ratio = float(b_ratio)
        self.kappa_place = float(kappa_place)
        self.phase_half = int(phase_half)
        self.interval_kind = interval_kind

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0, raw=None):
        assert raw is not None, "the interval emission needs the batch's downbeat_times"
        post = self.encoder(h, mask)
        phase_kappa = post["phase"]["kappa"]
        tempo_mu, tempo_sigma = post["tempo"]["mu"], post["tempo"]["sigma"]
        rotation_weight = post["rotation"]["weight"]

        w = torch.ones_like(phase_kappa) if mask is None else mask
        pair_mask = (w[:, 1:] > 0) & (w[:, :-1] > 0)
        pair_w = pair_mask.to(w.dtype)
        ann_f, ann_valid = annotation_frames(raw, phase_kappa.device)

        h_tempo = ((0.5 * math.log(2.0 * math.pi * math.e)
                    + torch.log(tempo_sigma)) * w).sum(1)

        recon = 0.0
        logp_tempo = 0.0
        kl_phase = 0.0
        phi = None

        for _ in range(samples):
            dotphi = tempo_mu * torch.exp(tempo_sigma * torch.randn_like(tempo_sigma))
            step = dotphi[:, :-1] + sample_vonmises(phase_kappa[:, 1:])
            ramp = torch.cat([torch.zeros_like(dotphi[:, :1]),
                              torch.cumsum(step * pair_w, dim=1)], dim=1)

            theta, _ = self.encoder._anchor(ramp.detach(), rotation_weight)
            phi = ramp + theta[:, None]
            phi_place = ramp.detach() + theta[:, None]

            logp_tempo = logp_tempo + self.tempo_log_prior(dotphi, w)
            kl_phase = kl_phase + self.kl_jitter(ramp[:, 1:], phase_kappa[:, 1:], pair_w)
            recon = recon + interval_loglik(phi, ann_f, ann_valid, self.kappa_place,
                                            self.b_ratio, self.phase_half,
                                            self.interval_kind, phi_place)["loglik"]

        recon = recon / samples
        logp_tempo = logp_tempo / samples
        kl_phase = kl_phase / samples
        kl = kl_phase - h_tempo - logp_tempo

        return {"elbo": recon - kl, "recon": recon, "kl": kl,
                "mu": phi, "kappa": phase_kappa,
                "tempo_prior": logp_tempo, "tempo_entropy": h_tempo,
                "kl_phase": kl_phase}


def build_model(cfg, input_dim: int) -> IntervalVAE:
    return IntervalVAE(input_dim, b_ratio=cfg.b_ratio, kappa_place=cfg.kappa_place,
                       phase_half=cfg.phase_half, interval_kind=cfg.interval_kind,
                       **common_kwargs(cfg))


def on_epoch(model, cfg, epoch: int) -> None:
    """Placement precision anneals; the interval ruler carries the rate meanwhile."""
    _base_on_epoch(model, cfg, epoch)
    if not cfg.kappa_anneal:
        return
    lo, hi, frac = (float(v) for v in cfg.kappa_anneal.split(","))
    ramp = min(1.0, epoch / max(1.0, frac * cfg.epochs))
    model.kappa_place = math.exp(math.log(lo) + ramp * (math.log(hi) - math.log(lo)))
