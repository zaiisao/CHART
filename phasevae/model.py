"""The phase-tracking VAE: latent bar phase per frame, audio-blind emission.

Generative model, with the beat grid GIVEN (so tempo is observed, not latent):

    phi_1 ~ prior_init                                  (Uniform[0, 2pi) by default)
    phi_t = phi_{t-1} + delta_t + eps_t  (mod 2pi),     eps_t ~ vM(0, kappa_t)
    kappa_t = softplus(f_psi(h_t))
    logit p(y_i = 1 | phi) = a + b cos(phi_{t_i}),      b > 0

delta_t = 2 pi / (m * IBI_t * fps) is the deterministic bar-phase advance implied by the
given grid. The emission reads the LATENT ONLY -- if it read h, the decoder would fit y
directly and the latent would go unused.

Inference q(phi_t | phi_{t-1}, h, y) = vM(phi_{t-1} + delta_t + delta^q_t, lambda_t):
a RESIDUAL against the prior mean, so the KL is a statement about what the encoder adds.

One structural consequence, load-bearing and stated up front. Because the increments
are independent of phi given (h, y), the whole trajectory is a cumulative sum and needs
no recurrence over t -- sampling and the KL are fully vectorised. The same fact means
that with the default Uniform phi_1 the generative model is EXACTLY invariant to a global
rotation phi -> phi + c, so p(y | h) is invariant to cyclic shifts of the downbeat
pattern and the deployment offset is unidentifiable at exactly chance. ``anchor_init``
adds the minimal repair: phi_1 ~ vM(mu_0(h), kappa_0(h)), which is the only term that can
break that symmetry. Both arms are built and both are reported.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .vonmises import kl_vonmises, sample_vonmises

TWO_PI = 2.0 * math.pi
INIT_KAPPA = 2000.0   # per-frame phase noise sd ~ 1/sqrt(kappa) = 0.022 rad at 50 fps.
                      # The earlier 20 was catastrophic: sd 0.22 rad/frame against a
                      # bar advance of ~0.06 rad/frame, i.e. a prior whose phase is
                      # already uniform two seconds in. q is initialised to the same
                      # value so the KL starts near 0 instead of near 10^4.


def inverse_softplus(value: float) -> float:
    """Pre-activation giving softplus(x) = value; linear above 30 where expm1 overflows."""
    return value if value > 30.0 else math.log(math.expm1(value))


def score_offsets(logits, m: int):
    """[B, m] log p(y_r | .) marginalised over a sample bank of phase trajectories.

    log p(y_r | phi) = sum_i [ y_ri * logit_i - softplus(logit_i) ], and y_r is the
    indicator of beats r, r+m, r+2m, ... so the first sum is one matrix product against
    an [n, m] offset matrix. Doing all m hypotheses and all samples at once instead of
    in nested Python loops is what makes evaluation minutes rather than hours.

    Args:
        logits: [samples, B, n] emission logits at the beats.
        m: beats per bar; the offset hypotheses are 0..m-1.

    Returns:
        [B, m] log-probabilities, up to the constant that is common to all offsets.
    """
    n = logits.shape[-1]
    offsets = (torch.arange(n, device=logits.device)[:, None] % m
               == torch.arange(m, device=logits.device)[None, :]).to(logits.dtype)
    per_sample = (logits @ offsets
                  - nn.functional.softplus(logits).sum(-1, keepdim=True))
    return torch.logsumexp(per_sample, dim=0) - math.log(logits.shape[0])


class KappaNet(nn.Module):
    """kappa_t = softplus(f_psi(h_t)) -- the audio's only entry into the transition."""

    def __init__(self, input_dim: int, hidden: int = 128, init_kappa: float = INIT_KAPPA):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, 1)
        # NOT zero-initialised: a zero output layer makes d(loss)/d(body) exactly zero at
        # step 0, which is how this project shipped a dead subnetwork three times.
        nn.init.normal_(self.out.weight, std=0.05)
        nn.init.constant_(self.out.bias, inverse_softplus(init_kappa))

    def forward(self, h):
        """[B, T, D] features -> [B, T] concentrations."""
        return nn.functional.softplus(self.out(self.body(h)).squeeze(-1)) + 1e-3


class Encoder(nn.Module):
    """q's residual mean and concentration from (h, y), read bidirectionally.

    y enters as two per-frame channels (beat / downbeat markers at the beat frames) --
    training only; nothing here runs at deployment.
    """

    def __init__(self, input_dim: int, hidden: int = 96,
                 init_lambda: float = INIT_KAPPA):
        super().__init__()
        self.init_lambda = inverse_softplus(init_lambda)
        self.proj = nn.Linear(input_dim + 2, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 2)
        # small but NOT zero (a zero row would cut d(loss)/d(body) at step 0). The
        # residual head has to start near 0 for a different reason: at kappa ~ 2000 a
        # random 1 rad residual costs ~3000 nats per frame against a ~20 nat
        # reconstruction, and the optimiser spends the whole run undoing it.
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, y_channels):
        """([B, T, D], [B, T, 2]) -> (residual mean [B, T], concentration [B, T])."""
        x = torch.tanh(self.proj(torch.cat([h, y_channels], dim=-1)))
        out = self.out(self.rnn(x)[0])
        residual = math.pi * torch.tanh(out[..., 0])
        lam = nn.functional.softplus(out[..., 1] + self.init_lambda) + 1e-3
        return residual, lam


class PriorInit(nn.Module):
    """phi_1's prior: Uniform, or vM(mu_0(h), kappa_0(h)) when anchored.

    Uniform is the brief's model. The anchored arm exists because Uniform + additive
    noise makes the generative model exactly rotation-invariant, hence the deployment
    offset exactly unidentifiable; this is the minimal term that can break it and it
    still reads h only.
    """

    def __init__(self, input_dim: int, hidden: int = 96, anchored: bool = False):
        super().__init__()
        self.anchored = anchored
        if not anchored:
            return   # register NO parameters: an unused submodule would show up in the
                     # gradient audit as ten dead tensors and mask a real dead subnetwork
        self.rnn = nn.GRU(input_dim, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(4 * hidden, 3)
        nn.init.normal_(self.out.weight, std=0.05)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, mask):
        """([B, T, D], [B, T]) -> (mu_0 [B], kappa_0 [B]); kappa_0 = 0 when not anchored."""
        if not self.anchored:
            zeros = h.new_zeros(h.shape[0])
            return zeros, zeros
        # BOTH ends, not a mean pool: phi_1's prior mean is a statement about frame 1,
        # so the crop's first frame must be in the summary. Reading position 0 ALONE
        # leaves the forward direction's recurrent weights with exactly zero gradient
        # (at t = 0 a forward GRU has seen nothing) -- the gradient audit caught that.
        sequence = self.rnn(h)[0]
        last = (mask.sum(1).long() - 1).clamp_min(0)
        out = self.out(torch.cat(
            [sequence[:, 0],
             sequence[torch.arange(len(last), device=h.device), last]], dim=-1))
        mu = torch.atan2(out[:, 0], out[:, 1])
        return mu, nn.functional.softplus(out[:, 2] + 1.0) + 1e-3


class PhaseVAE(nn.Module):
    """The full model: transition, audio-blind emission, amortised inference.

    Args:
        input_dim: frontend feature width.
        anchor_init: give phi_1 an h-dependent von Mises prior (see PriorInit).
    """

    def __init__(self, input_dim: int, hidden: int = 128, anchor_init: bool = False):
        super().__init__()
        self.kappa_net = KappaNet(input_dim, hidden)
        self.encoder = Encoder(input_dim)
        self.prior_init = PriorInit(input_dim, anchored=anchor_init)
        self.emission_a = nn.Parameter(torch.tensor(-1.0))
        self.emission_b_raw = nn.Parameter(torch.tensor(1.0))

    @property
    def emission_b(self):
        """Concentration b > 0, by softplus: the emission peaks at phi = 0, never inverted."""
        return nn.functional.softplus(self.emission_b_raw)

    def emission_logits(self, phi):
        """The audio-blind downbeat logit a + b cos(phi). Smooth: no wrap indicator."""
        return self.emission_a + self.emission_b * torch.cos(phi)

    def rollout(self, delta, residual, lam, mu_0, phi_1_residual, phi_1_kappa):
        """Sample phi_{1:T} by cumulative sum -- increments are independent of phi.

        Args:
            delta: [B, T] deterministic advance; delta[:, 0] is unused (no previous frame).
            residual: [B, T] q's residual mean.
            lam: [B, T] q's concentration.
            mu_0: [B] prior mean for phi_1.
            phi_1_residual: [B] q's residual on phi_1.
            phi_1_kappa: [B] q's concentration on phi_1.

        Returns:
            [B, T] phases (unwrapped; cos() makes the wrap immaterial).
        """
        increments = delta + residual + sample_vonmises(lam)
        first = mu_0 + phi_1_residual + sample_vonmises(phi_1_kappa)
        # indexed on the LAST axis throughout, so the same code serves [B, T] in
        # training and [samples, B, T] when the evaluator draws a whole sample bank
        increments = torch.cat([first[..., None], increments[..., 1:]], dim=-1)
        return torch.cumsum(increments, dim=-1)

    def forward(self, h, delta, y_channels, mask, beat_frames, y, samples: int = 1):
        """One ELBO evaluation on a padded batch.

        Args:
            h: [B, T, D] features.
            delta: [B, T] per-frame bar-phase advance.
            y_channels: [B, T, 2] the encoder's view of y (training only).
            mask: [B, T] 1 on real frames.
            beat_frames: [B, n] frame index of each beat.
            y: [B, n] per-beat downbeat indicator.
            samples: Monte Carlo samples for the reconstruction term.

        Returns:
            dict with elbo, recon, kl (all per crop, [B]) and the sampled phases.
        """
        kappa = self.kappa_net(h)
        residual, lam = self.encoder(h, y_channels)
        mu_0, kappa_0 = self.prior_init(h, mask)

        kl_steps = kl_vonmises(residual, lam, torch.zeros_like(residual), kappa)
        kl_first = kl_vonmises(residual[:, 0] + mu_0, lam[:, 0], mu_0, kappa_0)
        kl = (kl_steps[:, 1:] * mask[:, 1:]).sum(1) + kl_first

        recon, phi_last = 0.0, None
        for _ in range(samples):
            phi = self.rollout(delta, residual, lam, mu_0, residual[:, 0], lam[:, 0])
            phi_beats = torch.gather(phi, 1, beat_frames)
            logits = self.emission_logits(phi_beats)
            recon = recon - nn.functional.binary_cross_entropy_with_logits(
                logits, y.float(), reduction="none").sum(1)
            phi_last = phi
        recon = recon / samples
        return {"elbo": recon - kl, "recon": recon, "kl": kl, "phi": phi_last,
                "kappa": kappa, "lam": lam}

    @torch.no_grad()
    def deploy_offset_scores(self, h, delta, mask, beat_frames, m: int,
                             samples: int = 256):
        """[B, m] log p(y_r | h) for each candidate offset -- reads h ONLY, never y.

        The deployment decision rule: score every hypothesis "the first downbeat of this
        crop is beat r" under the generative model, marginalising phi by Monte Carlo from
        the PRIOR (no encoder). Returns log-probabilities up to a common constant.
        """
        assert not self.training, "deploy path must run in eval mode"
        B, n = beat_frames.shape
        kappa = self.kappa_net(h)
        mu_0, kappa_0 = self.prior_init(h, mask)
        expand = (samples,) + kappa.shape
        first = (mu_0 + sample_vonmises(kappa_0.expand(samples, B))
                 if self.prior_init.anchored
                 else TWO_PI * torch.rand(samples, B, device=h.device))
        noisy = delta + sample_vonmises(kappa.expand(expand))
        phi = torch.cumsum(torch.cat([first[..., None], noisy[..., 1:]], dim=2), dim=2)
        logits = self.emission_logits(
            torch.gather(phi, 2, beat_frames.expand(samples, B, n)))
        return score_offsets(logits, m), torch.sigmoid(logits).mean(0)

    @torch.no_grad()
    def posterior_offset_scores(self, h, delta, y_channels, mask, beat_frames, m: int,
                                samples: int = 32, prior_sample: bool = False):
        """[B, m] the same score with phi from the POSTERIOR -- diagnostic, reads y.

        Not a deployment number: q sees y, so this measures only whether the latent
        CARRIES the phase, not whether the model can find it from audio.
        ``prior_sample=True`` is the latent-use ablation: keep the trained emission but
        draw phi from the prior instead.
        """
        residual, lam = self.encoder(h, y_channels)
        kappa = self.kappa_net(h)
        mu_0, kappa_0 = self.prior_init(h, mask)
        if prior_sample:
            residual = torch.zeros_like(residual)
            lam = kappa
        B, n = beat_frames.shape
        expand = (samples,) + lam.shape
        phi = self.rollout(delta, residual, lam.expand(expand), mu_0,
                           residual[:, 0], lam[:, 0].expand(samples, B))
        logits = self.emission_logits(
            torch.gather(phi, 2, beat_frames.expand(samples, B, n)))
        return score_offsets(logits, m), torch.sigmoid(logits).mean(0)
