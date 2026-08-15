"""phi_0 sampled by a denoising loop instead of read off the fold in closed form.

The anchor is a ONE-SHOT estimator: theta = -arg sum_t a_t e^{i mu_t}, computed once
and never revisited. Everything downstream inherits whatever it produced, and the
deployed offset is thereafter a free parameter that diffuses inside its tolerance.

Here theta starts as NOISE -- uniform over the circle at diffuse_t0 = 1, or the anchor
plus a controlled displacement below that -- and the autoregressive decoder walks it
home. At every knot the decoder reads (audio memory, cos phi, sin phi) for the phase it
has actually reached and emits an ADDITIVE phase shift, so a rotation error is repaired
by rotating; the multiplicative rate correction it used to emit cannot do that at any
affordable price (erasing 0.24 rad over one knot costs ~174,000 nats under the walk
prior, against a 0.14 nat reward).

The shifts are innovations and are charged as such: each is scored under the phase
prior, which the kappa-gate makes cheap at bar crossings and dear inside a bar. So the
loop is free to jump at bar lines and pays to wander mid-bar, which is the same physics
the walk gate encodes for tempo.
"""
from __future__ import annotations

import math

import torch

from .base import common_kwargs, epoch_note, objective, optimizer  # noqa: F401
from .interval import DEFAULTS as INTERVAL_DEFAULTS
from .interval import on_epoch  # noqa: F401
from ..constants import KAPPA_INTER, TWO_PI
from ..model import IntervalVAE
from ..observation import annotation_frames, interval_loglik
from ..specs import DecoderSpec, PlacementSpec
from ..vonmises import log_i0, sample_vonmises

DEFAULTS = {**INTERVAL_DEFAULTS, "diffuse_t0": 1.0, "shift_scale": 0.5}


class Phi0DiffusionVAE(IntervalVAE):

    def __init__(self, input_dim: int, diffuse_t0: float = 1.0,
                 shift_scale: float = 0.5, **kw):
        super().__init__(input_dim, **kw)
        self.diffuse_t0 = float(diffuse_t0)
        self.shift_scale = float(shift_scale)

    def theta_start(self, anchor, training: bool):
        """t0 = 1 is uniform over the circle; 0 is the anchor; between, a partial kick."""
        if not training or self.diffuse_t0 <= 0.0:
            return anchor
        noise = (torch.rand_like(anchor) - 0.5) * TWO_PI
        return anchor + self.diffuse_t0 * noise

    def denoise_scan(self, dotphi, jitter, memory, theta, pair_w, sample_noise=True):
        """The AR loop, but the decoder's output is an additive phase shift."""
        T = dotphi.shape[1]
        stride = self.decoder.knot_stride
        phase = theta
        segments = [phase[:, None]]
        shift_frames = []
        knots = []
        tokens = []
        start = 1
        while start < T:
            stop = min(start + stride, T)
            tokens.append(self._token(memory[:, start - 1], phase))
            delta = self.shift_scale * torch.tanh(
                self.zdec.next_correction(torch.stack(tokens, dim=1)))
            knots.append(delta)

            steps = dotphi[:, start - 1:stop - 1].clone()
            if sample_noise:
                steps = steps + jitter[:, start - 1:stop - 1]
            steps[:, 0] = steps[:, 0] + delta

            segment = phase[:, None] + torch.cumsum(
                steps * pair_w[:, start - 1:stop - 1], dim=1)
            segments.append(segment)
            shift_frames.append(delta[:, None].expand(-1, stop - start))
            phase = segment[:, -1]
            start = stop

        sh = torch.cat(shift_frames, dim=1)
        return torch.cat(segments, dim=1), torch.cat([sh, sh[:, -1:]], dim=1), knots

    def shift_log_prior(self, knots, crossing_at_knot):
        """Each shift is a phase innovation: cheap at a bar line, dear inside a bar."""
        d = torch.stack(knots, dim=1)
        kp = torch.full_like(d, self.walk.kappa_physical)
        if crossing_at_knot is not None and self.walk.kappa_gate:
            kp = torch.where(crossing_at_knot[:, :d.shape[1]],
                             torch.full_like(d, self.mix_kappa_or_inter()), kp)
        return (kp * torch.cos(d) - math.log(TWO_PI) - log_i0(kp)).sum(1)

    def mix_kappa_or_inter(self):
        return float(getattr(self, "mix_kappa", KAPPA_INTER))

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0, raw=None):
        assert raw is not None, "the interval emission needs the batch's downbeat_times"
        post, memory = self.encoder(h, mask)
        phase_kappa = post["phase"]["kappa"]
        tempo_mu, tempo_sigma = post["tempo"]["mu"], post["tempo"]["sigma"]
        rotation_weight = post["rotation"]["weight"]

        w = torch.ones_like(phase_kappa) if mask is None else mask
        pair_w = ((w[:, 1:] > 0) & (w[:, :-1] > 0)).to(w.dtype)
        ann_f, ann_valid = annotation_frames(raw, phase_kappa.device)
        h_tempo = ((0.5 * math.log(2.0 * math.pi * math.e)
                    + torch.log(tempo_sigma)) * w).sum(1)

        mean_ramp = torch.cumsum(tempo_mu, dim=1) - tempo_mu[:, :1]
        anchor, _ = self.encoder._anchor(mean_ramp.detach(), rotation_weight)
        crossing = None
        if self.walk.kind == "gated" or self.walk.kappa_gate:
            mean_phi = (mean_ramp + anchor[:, None]).detach()
            crossing = torch.div(mean_phi[:, 1:], TWO_PI, rounding_mode="floor") \
                != torch.div(mean_phi[:, :-1], TWO_PI, rounding_mode="floor")
        cross_knot = None
        if crossing is not None:
            s = self.decoder.knot_stride
            cross_knot = torch.stack(
                [crossing[:, i:i + s].any(1) for i in range(0, crossing.shape[1], s)], dim=1)

        recon = logp_tempo = kl_phase = resultant = shift_abs = 0.0
        phi = None
        for _ in range(samples):
            dotphi = tempo_mu * torch.exp(tempo_sigma * torch.randn_like(tempo_sigma))
            jitter = sample_vonmises(phase_kappa[:, 1:])
            theta0 = self.theta_start(anchor, self.training)
            phi, shifts, knots = self.denoise_scan(dotphi, jitter, memory, theta0, pair_w)

            logp_tempo = logp_tempo + self.walk_log_prior(dotphi, w, crossing) \
                + self.shift_log_prior(knots, cross_knot)
            kl_phase = kl_phase + self.kl_jitter(
                phi[:, 1:], phase_kappa[:, 1:], pair_w,
                crossing if self.walk.kappa_gate else None)
            em = interval_loglik(phi, ann_f, ann_valid, self.kappa_place, self.b_ratio,
                                 self.phase_half, self.interval_kind, None,
                                 self.disp_weight, self.placement.coord)
            recon = recon + em["loglik"]
            resultant = resultant + em["resultant"]
            shift_abs = shift_abs + torch.stack(knots, 1).abs().mean()

        recon, logp_tempo = recon / samples, logp_tempo / samples
        kl_phase = kl_phase / samples
        kl = kl_phase - h_tempo - logp_tempo
        return {"elbo": recon - kl, "recon": recon, "kl": kl, "phi": phi,
                "kappa": phase_kappa, "tempo_prior": logp_tempo,
                "tempo_entropy": h_tempo, "kl_phase": kl_phase,
                "resultant": resultant / samples, "corr": shifts,
                "corr_nodes": tuple(knots), "corr_abs": shift_abs / samples}

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        assert not self.training, "deployment path must run in eval mode"
        post, memory = self.encoder(h, mask)
        dotphi = post["tempo"]["mu"]
        mean_ramp = torch.cumsum(dotphi, dim=1) - dotphi[:, :1]
        anchor, _ = self.encoder._anchor(mean_ramp, post["rotation"]["weight"])
        pair_w = torch.ones_like(dotphi[:, 1:]) if mask is None else \
            ((mask[:, 1:] > 0) & (mask[:, :-1] > 0)).to(dotphi.dtype)
        phi, _s, _k = self.denoise_scan(dotphi, None, memory, anchor, pair_w,
                                        sample_noise=False)
        return phi


def build_model(cfg, input_dim: int) -> Phi0DiffusionVAE:
    return Phi0DiffusionVAE(input_dim, diffuse_t0=cfg.diffuse_t0,
                            shift_scale=cfg.shift_scale, b_ratio=cfg.b_ratio,
                            kappa_place=cfg.kappa_place, phase_half=cfg.phase_half,
                            interval_kind=cfg.interval_kind, disp_weight=cfg.disp_weight,
                            decoder=DecoderSpec(dim=cfg.dec_dim,
                                                knot_stride=cfg.knot_stride),
                            placement=PlacementSpec(coord=cfg.place_coord,
                                                    lift=cfg.place_lift,
                                                    attach=cfg.place_attach),
                            encoder_pe=cfg.encoder_pe,
                            **common_kwargs(cfg))
