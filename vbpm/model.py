"""The bar-pointer CVAE of the tutorial's section 7, with a periodic latent (section 9.9)."""
from __future__ import annotations

import math

import torch
from torch import nn

from .constants import (CORRECTION_MAX, DELTA_MAX, KAPPA_INTER, TEMPO_PRIOR_MU,
                        TEMPO_PRIOR_SIGMA, TEMPO_WALK_SIGMA, TWO_PI, WALK_INTER_SIGMA,
                        WALK_INTER_W, WALK_INTRA_SIGMA, WALK_MIX_SIGMA, WALK_MIX_W)
from .nets import Encoder, EmissionTransformer, ZDecoder, vonmises_log_density
from .observation import annotation_frames, interval_loglik, event_recon
from .specs import DecoderSpec, EmissionSpec, PlacementSpec, UpdateSpec, WalkSpec
from .vonmises import kl_vonmises, mean_resultant, sample_vonmises


class VBPM(nn.Module):
    """Encoder + physical prior + latent-only emission. Learnable: theta and phi only."""

    def __init__(self, input_dim: int, d_model: int = 128,
                 emission: EmissionSpec | str = "cosine",
                 walk: WalkSpec | None = None,
                 placement: PlacementSpec | None = None,
                 update: UpdateSpec | None = None,
                 decoder: DecoderSpec | None = None,
                 encoder_pe: bool = False):
        super().__init__()
        emission = EmissionSpec.coerce(emission)
        self.walk = walk or WalkSpec()
        self.placement = placement or PlacementSpec()
        self.update = update or UpdateSpec()
        self.decoder = decoder or DecoderSpec()

        self.encoder = Encoder(input_dim, d_model,
                               kappa_physical=self.walk.kappa_physical, use_pe=encoder_pe)

        self.emission_kind = emission.kind
        # ONLY the arm in use gets parameters. Registering both left the unused pair with
        # no gradient, which the audit correctly refused -- and an audit that has to be
        # weakened to accommodate dead parameters stops being an audit.
        self.emission_net = None
        if emission.kind == "transformer":
            self.emission_net = EmissionTransformer(emission.dim, emission.layers,
                                                    use_positional=emission.positional)
        else:
            # JA: emission_a and emission_b_raw are scalars used by the Bernoulli distribution
            self.emission_a = nn.Parameter(torch.tensor(-3.0))
            self.emission_b_raw = nn.Parameter(torch.tensor(1.0))

        self.register_buffer("emission_b_floor", torch.tensor(0.0))

        self.zdec = ZDecoder(d_model, d=self.decoder.dim)
        if self.update.delta_on:
            self.delta_head = nn.Linear(
                self.zdec.out.in_features + (1 if self.update.gate_cond else 0), 1)
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)

    @property
    def emission_b(self):
        """Amplitude, positive by softplus, never below the scheduled floor."""
        return self.emission_b_floor + nn.functional.softplus(self.emission_b_raw)

    def emission_logits(self, phi, mask=None):
        """Downbeat logits from the LATENT alone (Point 1) -- never from h."""
        if self.emission_net is not None:
            return self.emission_net(phi, mask)

        if self.emission_kind == "triangle":
            wrapped = torch.atan2(torch.sin(phi), torch.cos(phi))   # (-pi, pi]
            return self.emission_a + self.emission_b * (1.0 - 2.0 * wrapped.abs() / math.pi)
        if self.emission_kind == "bump":
            peak = torch.exp(float(self.emission.bump_kappa) * (torch.cos(phi) - 1.0))
            return self.emission_a + self.emission_b * (2.0 * peak - 1.0)

        return self.emission_a + self.emission_b * torch.cos(phi)

    @torch.no_grad()
    def phase_ablation_gap(self, phi, mask=None):
        """How much the emission actually depends on phi. Near 0 means it has cheated."""
        live = self.emission_logits(phi, mask)
        frozen = self.emission_logits(torch.zeros_like(phi), mask)
        weight = torch.ones_like(live) if mask is None else mask
        return float(((live - frozen).abs() * weight).sum() / weight.sum())

    def _token(self, memory_frame, phase):
        return torch.cat([memory_frame, torch.cos(phase)[:, None],
                          torch.sin(phase)[:, None]], dim=-1)

    def walk_log_prior(self, dot_eff, w, crossing=None):
        if self.walk.kind == "gauss":
            return self.tempo_log_prior(dot_eff, w)

        log_dotphi = torch.log(dot_eff)
        z = (log_dotphi[:, 0] - TEMPO_PRIOR_MU) / TEMPO_PRIOR_SIGMA
        init = -0.5 * z ** 2 - math.log(TEMPO_PRIOR_SIGMA) - 0.5 * math.log(2.0 * math.pi)
        step = log_dotphi[:, 1:] - log_dotphi[:, :-1]
        pair = (w[:, 1:] > 0) & (w[:, :-1] > 0)
        if self.walk.kind == "gated":
            intra = -0.5 * (step / WALK_INTRA_SIGMA) ** 2 - math.log(WALK_INTRA_SIGMA) \
                - 0.5 * math.log(2.0 * math.pi)
            comps = [math.log(wt) - 0.5 * (step / sg) ** 2 - math.log(sg)
                     - 0.5 * math.log(2.0 * math.pi)
                     for wt, sg in zip(WALK_INTER_W, WALK_INTER_SIGMA)]
            calibration = -math.log(WALK_INTRA_SIGMA) \
                - math.log(sum(wt / sg for wt, sg in zip(WALK_INTER_W, WALK_INTER_SIGMA)))
            inter = torch.logsumexp(torch.stack(comps), dim=0) + calibration
            lp = torch.where(crossing, inter, intra)
        else:
            comps = [math.log(wt) - 0.5 * (step / sg) ** 2 - math.log(sg)
                     - 0.5 * math.log(2.0 * math.pi)
                     for wt, sg in zip(WALK_MIX_W, WALK_MIX_SIGMA)]
            lp = torch.logsumexp(torch.stack(comps), dim=0)
        return init + (lp * pair).sum(1)

    def _knot_delta(self, tokens, on=None):
        x = self.zdec.proj(tokens) + self.zdec.pe[:tokens.shape[1]]
        head_in = self.zdec.blocks(x)[:, -1]
        if self.update.gate_cond:
            g = (torch.zeros_like(head_in[:, :1]) if on is None
                 else on.to(head_in.dtype)[:, None])
            head_in = torch.cat([head_in, g], dim=1)
        raw = self.delta_head(head_in)[:, 0]
        return DELTA_MAX * torch.tanh(raw / DELTA_MAX)

    def _scan(self, dotphi, jitter, memory, theta, pair_w, sample_noise=True,
              crossing=None, kappa_q=None):
        T = dotphi.shape[1]
        stride = self.decoder.knot_stride
        phase = theta
        segments = [phase[:, None]]
        corr_frames = []
        knots = []
        tokens = []
        deltas = []
        kl_terms = []
        shift = torch.zeros_like(theta)
        shifts = [shift[:, None]]
        start = 1
        while start < T:
            stop = min(start + stride, T)

            token = self._token(memory[:, start - 1], phase)
            tokens.append(token)
            stack = torch.stack(tokens, dim=1)
            correction = self.zdec.next_correction(stack)
            correction = CORRECTION_MAX * torch.tanh(correction / CORRECTION_MAX)
            knots.append(correction)

            on = None
            if crossing is not None and self.walk.kappa_gate:
                on = crossing[:, start - 1:stop - 1].any(1)
            d_k = (self._knot_delta(stack, on) if self.update.delta_on
                   else torch.zeros_like(correction))
            deltas.append(d_k)
            if self.update.delta_on and kappa_q is not None:
                kp = torch.full_like(d_k, self.walk.kappa_physical)
                if on is not None:
                    kp = torch.where(on, torch.full_like(d_k, KAPPA_INTER), kp)
                kl_terms.append(mean_resultant(kappa_q[:, start - 1]) * kp
                                * (1.0 - torch.cos(d_k)))

            steps = dotphi[:, start - 1:stop - 1] * torch.exp(correction)[:, None]
            if sample_noise:
                steps = steps + jitter[:, start - 1:stop - 1]
            if self.update.delta_on:
                steps = torch.cat([steps[:, :1] + d_k[:, None], steps[:, 1:]], dim=1)

            segment = phase[:, None] + torch.cumsum(
                steps * pair_w[:, start - 1:stop - 1], dim=1)
            segments.append(segment)
            corr_frames.append(correction[:, None].expand(-1, stop - start))

            lift = dotphi[:, start - 1:stop - 1].detach() \
                * (torch.exp(correction) - 1.0)[:, None]
            span = shift[:, None] + torch.cumsum(
                lift * pair_w[:, start - 1:stop - 1], dim=1)
            shifts.append(span)
            shift = span[:, -1]

            phase = segment[:, -1]
            start = stop

        corr = torch.cat(corr_frames, dim=1)
        corr_full = torch.cat([corr, corr[:, -1:]], dim=1)

        kl_delta = (torch.stack(kl_terms, dim=1).sum(1) if kl_terms
                    else torch.zeros_like(theta))
        return (torch.cat(segments, dim=1), corr_full, knots,
                torch.cat(shifts, dim=1), torch.stack(deltas, dim=1), kl_delta)

    def kl_phase_step(self, delta, kappa_q, mask):
        kp = torch.full_like(kappa_q, self.walk.kappa_physical)
        return (kl_vonmises(delta, kappa_q, torch.zeros_like(delta), kp) * mask).sum(1)

    def tempo_log_prior(self, dotphi, w):
        log_dotphi = torch.log(dotphi)

        z = (log_dotphi[:, 0] - TEMPO_PRIOR_MU) / TEMPO_PRIOR_SIGMA
        init = -0.5 * z ** 2 - math.log(TEMPO_PRIOR_SIGMA) - 0.5 * math.log(2.0 * math.pi)

        step = log_dotphi[:, 1:] - log_dotphi[:, :-1]
        pair = (w[:, 1:] > 0) & (w[:, :-1] > 0)

        lp = -0.5 * (step / TEMPO_WALK_SIGMA) ** 2 - math.log(TEMPO_WALK_SIGMA) \
            - 0.5 * math.log(TWO_PI)

        return init + (lp * pair).sum(1)

    def phase_log_prior(self, phi, dotphi, w, kappa_p):
        pair = (w[:, 1:] > 0) & (w[:, :-1] > 0)
        predicted = phi[:, :-1] + dotphi[:, :-1]

        return (vonmises_log_density(phi[:, 1:], predicted, kappa_p) * pair).sum(1)

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0):
        post, memory = self.encoder(h, mask)
        phase, tempo = post["phase"], post["tempo"]

        phase_mu_offset, phase_kappa = phase["mu_offset"], phase["kappa"]
        tempo_mu, tempo_sigma = tempo["mu"], tempo["sigma"]

        w = torch.ones_like(phase_kappa) if mask is None else mask
        pair_w = ((w[:, 1:] > 0) & (w[:, :-1] > 0)).to(w.dtype)

        entropy_norm = 0.5 * math.log(2.0 * math.pi * math.e)
        h_tempo = ((entropy_norm + torch.log(tempo_sigma)) * w).sum(1)

        recon = 0.0
        logp_tempo = 0.0
        kl_phase = 0.0
        phi = None

        for _ in range(samples):
            phi_1 = torch.rand_like(tempo_mu[:, 0]) * TWO_PI
            dotphi = tempo_mu * torch.exp(tempo_sigma * torch.randn_like(tempo_sigma))
            eps = sample_vonmises(phase_kappa[:, 1:])

            steps = torch.cumsum(
                (dotphi[:, :-1] + phase_mu_offset[:, 1:] + eps) * pair_w, dim=1)
            phi = phi_1[:, None] + torch.cat([torch.zeros_like(steps[:, :1]), steps], dim=1)

            logp_tempo = logp_tempo + self.tempo_log_prior(dotphi, w)
            kl_phase = kl_phase + self.kl_phase_step(
                phase_mu_offset[:, 1:], phase_kappa[:, 1:], pair_w)

            recon = recon + event_recon(self.emission_logits(phi, mask), y, w, pos_weight)

        recon = recon / samples
        logp_tempo = logp_tempo / samples

        kl_phase = kl_phase / samples
        kl_tempo = h_tempo + logp_tempo
        kl = kl_phase - kl_tempo

        return {"elbo": recon - kl, "recon": recon, "kl": kl,
                "phi": phi, "kappa": phase_kappa,
                "tempo_prior": logp_tempo, "tempo_entropy": h_tempo,
                "kl_phase": kl_phase}

    @property
    def deployed_net(self):
        """The inference network read at test time; controls assert ITS target-blindness."""
        return self.encoder

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        assert not self.training, "deployment path must run in eval mode"
        post, memory = self.encoder(h, mask)
        dotphi = post["tempo"]["mu"]
        mean_ramp = torch.cumsum(dotphi, dim=1) - dotphi[:, :1]
        theta, _ = self.encoder._anchor(mean_ramp, post["rotation"]["weight"])
        T = dotphi.shape[1]
        stride = self.decoder.knot_stride
        phase = theta
        segments = [phase[:, None]]
        tokens = []
        start = 1
        while start < T:
            stop = min(start + stride, T)
            tokens.append(self._token(memory[:, start - 1], phase))
            correction = self.zdec.next_correction(torch.stack(tokens, dim=1))
            correction = CORRECTION_MAX * torch.tanh(correction / CORRECTION_MAX)
            segment = phase[:, None] + torch.cumsum(
                dotphi[:, start - 1:stop - 1] * torch.exp(correction)[:, None], dim=1)
            segments.append(segment)
            phase = segment[:, -1]
            start = stop
        return torch.cat(segments, dim=1)

    @torch.no_grad()
    def emission_probs(self, h, mask=None):
        """Alternative D (8.3.4): the emission evaluated at the deployed mean path."""
        return torch.sigmoid(self.emission_logits(self.infer_phase(h, mask), mask))


class IntervalVAE(VBPM):
    """VBPM with the time/interval observation model in place of the Bernoulli."""

    wants_raw = True

    def __init__(self, input_dim: int, b_ratio: float = 0.1, kappa_place: float = 100.0,
                 phase_half: int = 0, interval_kind: str = "laplace",
                 disp_weight: float = 0.0, **kw):
        super().__init__(input_dim, **kw)
        self.b_ratio = float(b_ratio)
        self.kappa_place = float(kappa_place)
        self.phase_half = int(phase_half)
        self.interval_kind = interval_kind
        self.disp_weight = float(disp_weight)

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0, raw=None):
        assert raw is not None, "the interval emission needs the batch's downbeat_times"
        post, memory = self.encoder(h, mask)
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
        resultant = 0.0
        phi = None

        mean_ramp = torch.cumsum(tempo_mu, dim=1) - tempo_mu[:, :1]
        theta, _ = self.encoder._anchor(mean_ramp.detach(), rotation_weight)
        crossing = None
        if self.walk.kind == "gated" or self.walk.kappa_gate:
            mean_phi = (mean_ramp + theta[:, None]).detach()
            crossing = torch.div(mean_phi[:, 1:], TWO_PI, rounding_mode="floor") \
                != torch.div(mean_phi[:, :-1], TWO_PI, rounding_mode="floor")
        corr_full = None
        knots = None
        corr_abs = 0.0

        for _ in range(samples):
            dotphi = tempo_mu * torch.exp(tempo_sigma * torch.randn_like(tempo_sigma))
            jitter = sample_vonmises(phase_kappa[:, 1:])
            phi, corr_full, knots, lift, deltas, kl_delta = self._scan(
                dotphi, jitter, memory, theta, pair_w, crossing=crossing,
                kappa_q=phase_kappa)
            dot_eff = dotphi * torch.exp(corr_full)
            phi_place = None if self.placement.attach else (
                phi.detach() + (theta - theta.detach())[:, None]
                + (lift - lift.detach()) * self.placement.lift)

            logp_tempo = logp_tempo + self.walk_log_prior(dot_eff, w, crossing)
            kl_phase = kl_phase + self.kl_jitter(
                phi[:, 1:], phase_kappa[:, 1:], pair_w,
                crossing if self.walk.kappa_gate else None) + kl_delta
            em = interval_loglik(phi, ann_f, ann_valid, self.kappa_place,
                                 self.b_ratio, self.phase_half,
                                 self.interval_kind, phi_place, self.disp_weight,
                                 self.placement.coord)
            recon = recon + em["loglik"]
            resultant = resultant + em["resultant"]
            corr_abs = corr_abs + corr_full.abs().mean()

        recon = recon / samples
        logp_tempo = logp_tempo / samples
        kl_phase = kl_phase / samples
        kl = kl_phase - h_tempo - logp_tempo

        return {"elbo": recon - kl, "recon": recon, "kl": kl,
                "phi": phi, "kappa": phase_kappa,
                "tempo_prior": logp_tempo, "tempo_entropy": h_tempo,
                "kl_phase": kl_phase, "resultant": resultant / samples,
                "corr": corr_full, "corr_nodes": tuple(knots),
                "corr_abs": corr_abs / samples,
                "delta": deltas, "kl_delta": kl_delta}


def downbeat_times(mu, mask=None):
    """Fractional frame times where the phase crosses multiples of 2 pi, per item.
    Linear interpolation between frames removes downbeat_frames' up-to-one-frame
    early bias."""
    r = torch.remainder(mu, 2.0 * math.pi)
    drop = torch.diff(r, dim=-1) < -math.pi
    if mask is not None:
        drop = drop & (mask[:, 1:] > 0)
    out = []
    for b in range(mu.shape[0]):
        idx = torch.nonzero(drop[b], as_tuple=False)[:, 0]
        r0 = r[b, idx]
        r1 = r[b, idx + 1] + 2.0 * math.pi
        frac = (2.0 * math.pi - r0) / (r1 - r0).clamp(min=1e-9)
        out.append(idx.to(mu.dtype) + frac)
    return out


def downbeat_frames(mu, mask=None):
    """Rule g (8.1.2): a downbeat is where the phase crosses ZERO. Deterministic."""
    zero_to_two_pi = torch.remainder(mu, 2.0 * math.pi)
    crossing = torch.diff(zero_to_two_pi, dim=-1) < -math.pi
    if mask is not None:
        crossing = crossing & (mask[:, 1:] > 0)
    return crossing
