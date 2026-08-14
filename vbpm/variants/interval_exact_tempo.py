"""Fixed-dimension tempo latent: one log-rate per block of W frames, bars emerge from phase.

The bar-segmented tempo latent makes the latent's own DIMENSION a function of the latent:
a fast rate cuts the window into more bars, so it carries more log-rate values, more
entropy terms and more walk factors than a slow one. Neither available measure survives
that.

    per BAR      the count of terms is proportional to the rate -- a bar-count subsidy,
                 measured at ~1.7 nats per harvested bar, which once let a rail beat
                 perfectly aligned truth by 17 nats of ELBO.
    per FRAME    the count is rate-free but wrong by the ratio frames/bars: a 45 s window
                 holds ~16 distinct bar-pooled values and 2250 frames, so the entropy is
                 charged 141x. Widening sigma from 0.15 to the 0.25 ceiling then buys
                 +1149 nats against an emission that spans +-150; sigma ended >= 0.245 in
                 38 of 40 runs.

Here the latent lives on a FIXED grid instead: one log-rate per block of ``block_frames``
frames, K = ceil(T / W) of them whatever the tempo is. The per-frame log-rate is the block
value, held constant across the block or interpolated linearly between block centres
(``block_interp``). Bars are no longer a construction of the latent -- they are crossings
of 2 pi in the phase the latent integrates to.

Because K does not depend on the rate, both q's entropy and the walk prior are exact sums
over the SAME K terms, and the two scale together:

    scheme                    entropy terms   walk pairs   sigma that maximises prior + H
    per-bar latent/per-frame        2250           15         0.2500  (the ceiling)
    per-bar latent/per-bar            16           15         0.0258
    block W = 50                      45           44         0.0080
    block W = 25                      90           89         0.0040

and widening sigma from 0.15 to 0.25 moves the entropy by +23.0 nats (W = 50) rather than
+1149, against a walk that simultaneously charges -72.4. See BLOCK_WALK_MIX for how that
walk was measured; ``python -m vbpm.variants.interval_exact_tempo`` re-derives it.
"""
from __future__ import annotations

import math

import torch

from .base import common_kwargs, objective  # noqa: F401
from .base import optimizer  # noqa: F401
from .interval import DEFAULTS as INTERVAL_DEFAULTS, IntervalVAE, on_epoch  # noqa: F401
from ..model import (Encoder, TEMPO_PRIOR_EPS, TEMPO_PRIOR_MU, TEMPO_PRIOR_SIGMA,
                     TEMPO_HI, TEMPO_LO, TWO_PI, bounded_kappa)

DEFAULTS = dict(INTERVAL_DEFAULTS, block_frames=50, block_interp=True, sigma_ceil=0.25)

# (weight, b_coast, b_change) of the block-to-block log-rate step, fit by EM over the
# 2893 catalog songs that carry four or more downbeats. The tempo curve is read as linear
# between bar CENTRES: holding it piecewise constant per bar instead puts an atom of 49%
# at step 0 (W = 25), which is a property of measuring tempo once per bar and not of
# tempo, and it drives b_coast to a degenerate 0.0000. The same estimator run on the
# per-BAR steps returns (0.661, 0.0185, 0.1760) against model.py's banked
# TEMPO_WALK_MIX = (0.665, 0.019, 0.180), which is what licenses these.
BLOCK_WALK_MIX = {25: (0.608, 0.0032, 0.0407),
                  50: (0.616, 0.0061, 0.0711),
                  100: (0.612, 0.0092, 0.1057)}

LOG_SQRT_2PI_E = 0.5 * math.log(2.0 * math.pi * math.e)


def block_means(x, w, width: int):
    """[B, T] -> ([B, K] mask-weighted block means, [B, K] block validity), K = ceil(T/W)."""
    b, t = x.shape
    pad = (-t) % width
    xp = torch.nn.functional.pad(x, (0, pad)).reshape(b, -1, width)
    wp = torch.nn.functional.pad(w, (0, pad)).reshape(b, -1, width)
    counts = wp.sum(-1)
    live = counts > 0
    means = torch.where(live, (xp * wp).sum(-1) / counts.clamp(min=1e-6), xp.mean(-1))
    return means, live


def expand_blocks(block, length: int, width: int, interp: bool):
    """[B, K] -> [B, T]: the block value held across its block, or interpolated."""
    if not interp:
        held = block[:, :, None].expand(-1, -1, width)
        return held.reshape(block.shape[0], -1)[:, :length]
    k = block.shape[1]
    frame = torch.arange(length, device=block.device, dtype=block.dtype)
    u = ((frame + 0.5) / width - 0.5).clamp(min=0.0, max=float(k - 1))
    i0 = u.floor().long()
    i1 = (i0 + 1).clamp(max=k - 1)
    frac = u - i0.to(block.dtype)
    return block[:, i0] * (1.0 - frac) + block[:, i1] * frac


def walk_log_prob(step, mix):
    """log of the two-Laplace step law, elementwise."""
    weight, coast, change = mix
    size = step.abs()
    stay = math.log(weight) - size / coast - math.log(2.0 * coast)
    move = math.log(1.0 - weight) - size / change - math.log(2.0 * change)
    return torch.logaddexp(stay, move)


class BlockEncoder(Encoder):
    """``Encoder.heads`` with the tempo latent on a fixed block grid, not on bars."""

    def __init__(self, *args, block_frames: int = 50, block_interp: bool = True, **kw):
        super().__init__(*args, **kw)
        self.block_frames = int(block_frames)
        self.block_interp = bool(block_interp)
        assert self.block_frames in BLOCK_WALK_MIX, (
            f"no measured step law for block_frames={self.block_frames}; run "
            f"`python -m vbpm.variants.interval_exact_tempo --widths "
            f"{self.block_frames}` and add the fit to BLOCK_WALK_MIX")
        self.walk_mix = BLOCK_WALK_MIX[self.block_frames]

    def heads(self, trunk, mask=None, h=None):
        """Trunk -> (mu [B, T], kappa [B, T], aux): q's parameters off the block grid."""
        channels = self.output_channels(trunk)
        kappa = bounded_kappa(
            torch.exp(channels["log_phi_kappa"] + self.log_phi_kappa_bias) + 1e-3)

        w = torch.ones(trunk.shape[:2], device=trunk.device, dtype=trunk.dtype) \
            if mask is None else mask
        width = self.block_frames

        block, live = block_means(channels["log_dotphi"], w, width)
        sigma, _live = block_means(channels["log_sigma_dotphi"], w, width)
        sigma = self.sigma_ceil * torch.sigmoid(
            sigma + math.log(self.sigma_init / (self.sigma_ceil - self.sigma_init)))

        weight = live.to(block.dtype)
        tempo_entropy = ((LOG_SQRT_2PI_E + torch.log(sigma)) * weight).sum(1)
        if self.training:
            block = block + sigma * torch.randn_like(sigma)

        log_dotphi = expand_blocks(block, trunk.shape[1], width, self.block_interp)
        _dotphi, mu0 = self._ramp(log_dotphi)
        a = torch.sigmoid(channels["downbeat_logit"]) * w
        offset, resultant = self._anchor(mu0.detach(), a)

        return {"phase": {"mu": mu0 + offset[:, None], "kappa": kappa},
            "tempo": {"log_mu": block, "sigma": sigma, "seg": None,
                      "log_prior": self._block_log_prior(block, weight),
                      "entropy": tempo_entropy},
            "offset": {"mu": offset}, "sigma": sigma,
            "resultant": resultant, "log_rate": block}

    def _block_log_prior(self, block, weight):
        """log p(rate_1) + sum_k log p(rate_k | rate_{k-1}) over BLOCKS. [B]."""
        z = (block[:, 0] - TEMPO_PRIOR_MU) / TEMPO_PRIOR_SIGMA
        log_gauss = -0.5 * z ** 2 - math.log(TEMPO_PRIOR_SIGMA) \
            - 0.5 * math.log(2.0 * math.pi)
        log_unif = -math.log(TEMPO_HI - TEMPO_LO)
        floor = torch.full_like(log_gauss, math.log(TEMPO_PRIOR_EPS) + log_unif)
        init = torch.logaddexp(math.log(1.0 - TEMPO_PRIOR_EPS) + log_gauss, floor)

        pair = weight[:, 1:] * weight[:, :-1]
        walk = walk_log_prob(block[:, 1:] - block[:, :-1], self.walk_mix) * pair
        return init * weight[:, 0] + walk.sum(1)


class BlockIntervalVAE(IntervalVAE):
    """IntervalVAE whose tempo latent has a fixed dimension T / block_frames."""

    wants_raw = True

    def __init__(self, input_dim: int, d_model: int = 128, block_frames: int = 50,
                 block_interp: bool = True, sigma_ceil: float = 0.0, **kw):
        super().__init__(input_dim, d_model=d_model, **kw)
        self.encoder = BlockEncoder(input_dim, d_model,
                                    kappa_physical=self.kappa_physical,
                                    block_frames=block_frames,
                                    block_interp=block_interp)


def build_model(cfg, input_dim: int) -> BlockIntervalVAE:
    """The interval emission with the tempo latent on a fixed block grid."""
    return BlockIntervalVAE(input_dim, b_ratio=cfg.b_ratio, kappa_place=cfg.kappa_place,
                            phase_half=cfg.phase_half, interval_kind=cfg.interval_kind,
                            block_frames=cfg.block_frames, block_interp=cfg.block_interp,
                            **common_kwargs(cfg))


def epoch_note(model, probe) -> str:
    """Per-epoch log: q's tempo sd and the block count it is charged on."""
    trunk = model.encoder.features(probe["h"], probe["mask"])
    aux = model.encoder.heads(trunk, probe["mask"])
    return (f"  sigma {float(aux['sigma'].mean()):6.4f}"
            f"  blocks {aux['sigma'].shape[1]:3d}"
            f"  H_tempo {float(aux['tempo']['entropy'].mean()):8.2f}"
            f"  walk {float(aux['tempo']['log_prior'].mean()):9.2f}")


def measure_walk_mix(widths=(25, 50, 100), fps: float = 50.0, iters: int = 400):
    """Re-derive BLOCK_WALK_MIX from the catalog annotations. Prints one fit per width."""
    import numpy as np

    from ..data.dataset import load_catalog

    songs = sorted(sum(load_catalog().values(), []), key=lambda s: s.song_id)
    curves = []
    for song in songs:
        db = np.asarray(song.beats()[1], dtype=np.float64)
        if len(db) < 4:
            continue
        period = np.diff(db)
        good = (period > 0.2) & (period < 8.0)
        rate = np.log(TWO_PI / (np.where(good, period, 1.0) * fps))
        lo, hi = int(math.floor(db[0] * fps)), int(math.ceil(db[-1] * fps))
        t = np.arange(lo, hi) / fps
        index = np.clip(np.searchsorted(db, t, side="right") - 1, 0, len(period) - 1)
        curves.append((np.interp(t, 0.5 * (db[:-1] + db[1:]), rate), good[index]))

    for width in widths:
        steps = []
        for rate, good in curves:
            n = (len(rate) // width) * width
            if n < 2 * width:
                continue
            means = rate[:n].reshape(-1, width).mean(1)
            live = good[:n].reshape(-1, width).all(1)
            steps.append(np.diff(means)[live[1:] & live[:-1]])
        x = np.abs(np.concatenate(steps))
        weight, coast, change = 0.6, 0.005, 0.05
        for _ in range(iters):
            stay = math.log(weight) - x / coast - math.log(2.0 * coast)
            move = math.log(1.0 - weight) - x / change - math.log(2.0 * change)
            top = np.maximum(stay, move)
            g = np.exp(stay - top) / (np.exp(stay - top) + np.exp(move - top))
            weight = float(g.mean())
            coast = float(max((g * x).sum() / max(g.sum(), 1e-9), 1e-6))
            change = float(max(((1 - g) * x).sum() / max((1 - g).sum(), 1e-9), 1e-6))
        print(f"W={width:4d} ({width / fps:.2f} s)  n={len(x)}  "
              f"(weight, b_coast, b_change) = ({weight:.3f}, {coast:.4f}, {change:.4f})",
              flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", default="25,50,100")
    measure_walk_mix([int(v) for v in parser.parse_args().widths.split(",")])
