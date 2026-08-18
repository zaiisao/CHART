"""The bar-pointer CVAE of the tutorial's section 7, with a periodic latent (section 9.9)."""
from __future__ import annotations

import math

import torch
from torch import nn

from .constants import (CORRECTION_MAX, DELTA_MAX, KAPPA_INTER, TWO_PI, WALK_INTER_SIGMA,
                        WALK_INTER_W, WALK_INTRA_SIGMA, WALK_MIX_SIGMA, WALK_MIX_W)
from .nets import Encoder, EmissionTransformer, ZDecoder, vonmises_log_density
from .observation import (annotation_frames, count_loglik, gauss_time_loglik,
                          interval_loglik, recon_term)
from .specs import DecoderSpec, EmissionSpec, PlacementSpec, UpdateSpec, WalkSpec
from .vonmises import kl_vonmises, log_i0, mean_resultant, sample_vonmises


class VBPM(nn.Module):
    """Encoder + physical prior + latent-only emission. Learnable: theta and phi only."""

    def __init__(self, input_dim: int, d_model: int = 128,
                 emission: EmissionSpec | str = "cosine",
                 walk: WalkSpec | None = None,
                 placement: PlacementSpec | None = None,
                 update: UpdateSpec | None = None,
                 phase_init: str = "anchor",
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
        self.phase_init = phase_init
        self.bump_kappa = float(emission.bump_kappa)
        self.recon_kind = emission.recon
        self.recon_term = recon_term(emission.recon)
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

        self.harmonics = int(emission.harmonics)
        if self.recon_kind == "class":
            self.wants_raw = True
            coef = torch.zeros(3, 2 * self.harmonics)
            coef[2, 0] = 1.0
            self.emission_coef = nn.Parameter(coef)
            self.emission_bias = nn.Parameter(torch.tensor([0.0, -2.5, -3.6]))

        self.kappa_p_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.kappa_p_head.weight)
        nn.init.constant_(self.kappa_p_head.bias, math.log(self.walk.kappa_physical))
        self.walk_sigma_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.walk_sigma_head.weight)
        nn.init.constant_(self.walk_sigma_head.bias, math.log(self.walk.walk_sigma))
        self.phi0_prior_head = nn.Linear(d_model, 3)
        nn.init.zeros_(self.phi0_prior_head.weight)
        with torch.no_grad():
            self.phi0_prior_head.bias.copy_(torch.tensor([1.0, 0.0, -6.0]))

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

    def class_logits(self, phi):
        """[..., 3] logits over (non-beat, beat, downbeat) at the sampled phase.

        Each class is a truncated Fourier series on the circle, so the reader stays dumb
        and the beat class has to find its own subdivisions instead of being told the
        meter. Only the downbeat class is seeded, with the first cosine, because phi = 0
        is the downbeat by the coordinate's definition rather than by an assumption
        about how many beats fill a bar.
        """
        j = torch.arange(1, self.harmonics + 1, device=phi.device, dtype=phi.dtype)
        angle = phi[..., None] * j
        basis = torch.cat([angle.cos(), angle.sin()], dim=-1)
        return self.emission_bias + basis @ self.emission_coef.T

    def prior_params(self, memory, mask):
        """(kappa_p [B,T], walk sigma [B,T], mu0 [B], kappa0 [B]) from the audio alone.

        Options T-b and I-2 of the phase note: the transition's concentration and the
        velocity walk's width are read per frame, and the initial phase gets a direction
        and a concentration of its own. Every head is zeroed at init with the corpus
        constant in its bias, and the initial concentration starts at softplus(-6), so an
        untrained model is the constant-prior model under a flat initial phase -- which
        is Option I-1 -- and every departure from it is learned. The direction's cosine
        starts at one rather than zero: atan2(0, 0) has the value the flat prior wants
        and a gradient that is 0/0, which is nan on the first backward.
        """
        kappa_p = self.kappa_p_head(memory)[..., 0].clamp(-4.0, 12.0).exp()
        walk_sigma = self.walk_sigma_head(memory)[..., 0].clamp(-9.0, 0.0).exp()
        if mask is not None:
            live = mask > 0
            kappa_p = torch.where(live, kappa_p, torch.full_like(kappa_p,
                                                                self.walk.kappa_physical))
            walk_sigma = torch.where(live, walk_sigma,
                                     torch.full_like(walk_sigma, self.walk.walk_sigma))
        w = mask[..., None] if mask is not None else torch.ones_like(memory[..., :1])
        pooled = (memory * w).sum(1) / w.sum(1).clamp(min=1.0)
        a1, a2, u = self.phi0_prior_head(pooled).unbind(-1)
        return kappa_p, walk_sigma, torch.atan2(a2, a1), nn.functional.softplus(u)

    def initial_phase_log_prior(self, phi_1, mu0, kappa0):
        """Log p_eta(phi_1 | x): the term the Dirac q used to drop on the floor."""
        return kappa0 * torch.cos(phi_1 - mu0) - math.log(TWO_PI) - log_i0(kappa0)

    def emission_logits(self, phi, mask=None):
        """Downbeat logits from the LATENT alone (Point 1) -- never from h."""
        if self.emission_net is not None:
            return self.emission_net(phi, mask)

        if self.emission_kind == "triangle":
            wrapped = torch.atan2(torch.sin(phi), torch.cos(phi))   # (-pi, pi]
            return self.emission_a + self.emission_b * (1.0 - 2.0 * wrapped.abs() / math.pi)
        if self.emission_kind == "bump":
            peak = torch.exp(self.bump_kappa * (torch.cos(phi) - 1.0))
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
        z = (log_dotphi[:, 0] - self.walk.tempo_mu) / self.walk.tempo_sigma
        init = -0.5 * z ** 2 - math.log(self.walk.tempo_sigma) \
            - 0.5 * math.log(2.0 * math.pi)
        pair = (w[:, 1:] > 0) & (w[:, :-1] > 0)
        step = log_dotphi[:, 1:] - log_dotphi[:, :-1]
        step = torch.where(pair, step, torch.zeros_like(step))
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

    def kl_jitter(self, mu, kappa, mask, crossing=None):
        """Per-frame KL( vM(mu,kappa) || vM(mu,kappa_p) ): prices concentration only."""
        kp = torch.full_like(kappa, self.walk.kappa_physical)
        if crossing is not None:
            kp = torch.where(crossing, torch.full_like(kappa, KAPPA_INTER), kp)
        return (kl_vonmises(mu, kappa, mu, kp) * mask).sum(1)

    def kl_phase_step(self, delta, kappa_q, mask, kappa_p=None):
        kp = (torch.full_like(kappa_q, self.walk.kappa_physical) if kappa_p is None
              else kappa_p)
        return (kl_vonmises(delta, kappa_q, torch.zeros_like(delta), kp) * mask).sum(1)

    def tempo_log_prior(self, dotphi, w, walk_sigma=None):
        log_dotphi = torch.log(dotphi)
        sigma = (torch.full_like(log_dotphi, self.walk.walk_sigma)
                 if walk_sigma is None else walk_sigma)

        z = (log_dotphi[:, 0] - self.walk.tempo_mu) / self.walk.tempo_sigma
        init = -0.5 * z ** 2 - math.log(self.walk.tempo_sigma) \
            - 0.5 * math.log(2.0 * math.pi)

        pair = (w[:, 1:] > 0) & (w[:, :-1] > 0)
        step = log_dotphi[:, 1:] - log_dotphi[:, :-1]
        step = torch.where(pair, step, torch.zeros_like(step))

        s = sigma[:, 1:]
        lp = -0.5 * (step / s) ** 2 - torch.log(s) - 0.5 * math.log(TWO_PI)

        return init + (lp * pair).sum(1)

    def phase_log_prior(self, phi, dotphi, w, kappa_p):
        pair = (w[:, 1:] > 0) & (w[:, :-1] > 0)
        predicted = phi[:, :-1] + dotphi[:, :-1]

        return (vonmises_log_density(phi[:, 1:], predicted, kappa_p) * pair).sum(1)

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0, raw=None):
        post, memory = self.encoder(h, mask)
        phase, tempo = post["phase"], post["tempo"]

        phase_mu_offset, phase_kappa = phase["mu_offset"], phase["kappa"]
        tempo_mu, tempo_sigma = tempo["mu"], tempo["sigma"]

        w = torch.ones_like(phase_kappa) if mask is None else mask
        pair_w = ((w[:, 1:] > 0) & (w[:, :-1] > 0)).to(w.dtype)

        entropy_norm = 0.5 * math.log(2.0 * math.pi * math.e)
        h_tempo = ((entropy_norm + torch.log(tempo_sigma)) * w).sum(1)

        kappa_p, walk_sigma, mu0, kappa0 = self.prior_params(memory, mask)
        target = None
        if self.recon_kind == "class":
            assert raw is not None, "the three-way emission needs the batch's class targets"
            target = raw["cls"].to(h.device)

        theta = None
        if self.phase_init == "anchor":
            mean_ramp = torch.cumsum(tempo_mu, dim=1) - tempo_mu[:, :1]
            theta, _ = self.encoder._anchor(mean_ramp.detach(),
                                            post["rotation"]["weight"])

        recon = 0.0
        logp_tempo = 0.0
        logp_phi1 = 0.0
        kl_phase = 0.0
        phi = None

        for _ in range(samples):
            phi_1 = (torch.rand_like(tempo_mu[:, 0]) * TWO_PI if theta is None
                     else theta)
            dotphi = tempo_mu * torch.exp(tempo_sigma * torch.randn_like(tempo_sigma))
            eps = sample_vonmises(phase_kappa[:, 1:])

            steps = torch.cumsum(
                (dotphi[:, :-1] + phase_mu_offset[:, 1:] + eps) * pair_w, dim=1)
            phi = phi_1[:, None] + torch.cat([torch.zeros_like(steps[:, :1]), steps], dim=1)

            logp_tempo = logp_tempo + self.tempo_log_prior(dotphi, w, walk_sigma)
            logp_phi1 = logp_phi1 + self.initial_phase_log_prior(phi_1, mu0, kappa0)
            kl_phase = kl_phase + self.kl_phase_step(
                phase_mu_offset[:, 1:], phase_kappa[:, 1:], pair_w, kappa_p[:, 1:])

            if target is None:
                recon = recon + self.recon_term(self.emission_logits(phi, mask),
                                                y, w, pos_weight)
            else:
                recon = recon + self.recon_term(self.class_logits(phi), target, w,
                                                pos_weight)

        recon = recon / samples
        logp_tempo = logp_tempo / samples
        logp_phi1 = logp_phi1 / samples

        kl_phase = kl_phase / samples
        kl_tempo = h_tempo + logp_tempo
        kl = kl_phase - kl_tempo - logp_phi1

        return {"elbo": recon - kl, "recon": recon, "kl": kl,
                "phi": phi, "kappa": phase_kappa,
                "tempo_prior": logp_tempo, "tempo_entropy": h_tempo,
                "kl_phase": kl_phase, "phi1_prior": logp_phi1,
                "kappa_p": kappa_p.mean(1), "kappa0": kappa0}

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
                 disp_weight: float = 0.0, count_weight: float = 0.0, **kw):
        super().__init__(input_dim, **kw)
        self.b_ratio = float(b_ratio)
        self.kappa_place = float(kappa_place)
        self.phase_half = int(phase_half)
        self.interval_kind = interval_kind
        self.disp_weight = float(disp_weight)
        self.count_weight = float(count_weight)

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
            if self.interval_kind == "gauss_time":
                ll = gauss_time_loglik(phi, ann_f, ann_valid, self.b_ratio,
                                       fps=float(raw["fps"][0]),
                                       phase_half=self.phase_half)
                if self.count_weight > 0.0:
                    ll = ll + self.count_weight * count_loglik(phi, ann_valid, mask)
                em = {"loglik": ll, "resultant": torch.zeros_like(ll)}
            else:
                em = interval_loglik(phi, ann_f, ann_valid, self.kappa_place,
                                     self.b_ratio, self.phase_half,
                                     self.interval_kind, phi_place, self.disp_weight,
                                     self.placement.coord, self.count_weight, mask)
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
    early bias.
    """
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
