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
from torch import nn

from .base import common_kwargs, epoch_note, objective, on_epoch as _base_on_epoch  # noqa: F401
from .base import optimizer  # noqa: F401
from ..model import (TWO_PI, VBPM, IntervalVAE, WALK_MIX_SIGMA, WALK_MIX_W,  # noqa: F401
                     annotation_frames, interp_phase, interval_loglik,  # noqa: F401
                     interval_penalty, log_i0, path_dotphi, sample_vonmises,  # noqa: F401
                     smooth_phase)  # noqa: F401

DEFAULTS = {"b_ratio": 0.1, "kappa_place": 100.0, "kappa_anneal": "3,300,0.7",
            "phase_half": 0, "interval_kind": "laplace", "disp_weight": 0.0,
            "dec_dim": 32, "knot_stride": 25, "dec_warmup": 15,
            "encoder_pe": False,
            "walk_kind": "gauss"}

def build_model(cfg, input_dim: int) -> IntervalVAE:
    return IntervalVAE(input_dim, b_ratio=cfg.b_ratio, kappa_place=cfg.kappa_place,
                       phase_half=cfg.phase_half, interval_kind=cfg.interval_kind,
                       disp_weight=cfg.disp_weight, dec_dim=cfg.dec_dim,
                       knot_stride=cfg.knot_stride, walk_kind=cfg.walk_kind,
                       encoder_pe=cfg.encoder_pe,
                       **common_kwargs(cfg))


def on_epoch(model, cfg, epoch: int) -> None:
    """Placement precision anneals; the interval ruler carries the rate meanwhile."""
    _base_on_epoch(model, cfg, epoch)
    if not cfg.kappa_anneal:
        return
    lo, hi, frac = (float(v) for v in cfg.kappa_anneal.split(","))
    ramp = min(1.0, epoch / max(1.0, frac * cfg.epochs))
    model.kappa_place = math.exp(math.log(lo) + ramp * (math.log(hi) - math.log(lo)))
    live = epoch >= cfg.dec_warmup
    for p in model.zdec.parameters():
        p.requires_grad_(live)
