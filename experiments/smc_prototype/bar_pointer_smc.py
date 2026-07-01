"""FIVO/SMC bar-pointer model: FIXED (non-audio-conditioned) prior + trained emission + particle filter.

This is the "VAE-preserving fix" this project's prior diagnosis identified: a sequential VAE need not
be inferred with a feed-forward amortized encoder. FIVO (Maddison et al. 2017 / Naesseth et al. 2018)
trains the SAME generative model as the plain VBPM (model/bar_pointer_vae.py) -- meter/phase/tempo
latents with the same von Mises / Normal / Categorical transition family, reused verbatim from
model/latents.py -- but:

  * the PRIOR is fixed (global learned scalars kappa, sigma, meter-transition logits; NOT a function
    of the audio at all -- deliberately, per the diagnosis: audio must only inform the emission, never
    silently override the dynamics prior, or "inference" degenerates back into amortization).
  * the PROPOSAL used to propagate particles is the prior itself (a *bootstrap* particle filter): we
    do not learn a separate proposal distribution. This is the simplest correct FIVO variant and
    matches what model/svt_core.py's `sample_from_prior_pf` (the archived reference) and
    `experiments/deploy_gap/scripts/dbn_vae.py`'s `pf_deploy` both do. A learned proposal (AESMC-style)
    would reduce variance further but is not needed to test the make-or-break hypothesis (emission
    strength), so it is left as a documented simplification.
  * the EMISSION is the strong, separately-trained ActivationEmissionHead (emission.py) -- a learned
    per-frame (p_beat, p_downbeat) detector -- NOT the raw 512-dim features and NOT a fixed hand-rolled
    onset envelope (the earlier attempt's mistake). The particle filter's importance weight compares,
    for each particle's phase/meter hypothesis, a geometric "expected activation" bump against the
    emission head's actual (p_beat, p_downbeat) output for that frame, via a Gaussian observation model
    (the same bump-vs-observation pattern as dbn_vae.py's emit_mu, generalized to a variable meter and
    to TWO channels (beat, downbeat) instead of one).

FIVO training propagates K particles PER SEQUENCE, for a whole BATCH of sequences at once (shapes
[batch, num_particles] throughout) -- the Python loop is over TIME only, so a training step over
several sequences costs about the same wall-clock as one (batching amortizes the per-step kernel-
launch overhead, which dominates at these small tensor sizes). Each sequence resamples independently,
triggered by its OWN effective sample size (ESS) dropping below `ess_frac * K` -- implemented by a
per-sequence boolean mask blending "resampled" and "kept" particle sets via torch.where, since
different sequences in a batch resample at different times.

The training loss accumulates the incremental log-mean-particle-weight each frame (the FIVO bound),
and resampling uses systematic resampling (reusing the exact scheme validated in the archived
model/svt_core.py `_systematic_resample`). Gradients flow through the REPARAMETERIZED per-particle
transition samples (von Mises implicit-reparam + Normal rsample, both from model/latents.py) between
resampling events; the resampling indices themselves are treated as constants (standard FIVO/AESMC
practice, exactly as in Maddison et al. 2017 -- the resampling step is non-differentiable).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.latents import sample_von_mises, sample_normal

TWO_PI = 2.0 * math.pi


def systematic_resample_batched(weights: torch.Tensor) -> torch.Tensor:
    """Systematic resampling per row: normalized weights [batch, K] -> ancestor indices [batch, K].

    Same low-variance O(K) scheme as the archived model/svt_core.py `_systematic_resample`
    (one uniform offset per row, evenly spaced positions), vectorized over the batch dimension.
    """
    batch, num_particles = weights.shape
    device = weights.device
    positions = (
        torch.arange(num_particles, device=device, dtype=weights.dtype).unsqueeze(0)
        + torch.rand(batch, 1, device=device)
    ) / num_particles                                                   # [batch, K]
    cumsum = torch.cumsum(weights, dim=-1)
    cumsum = cumsum / cumsum[:, -1:].clamp(min=1e-12)
    cumsum[:, -1] = 1.0
    # searchsorted needs a sorted last dim per row, which cumsum is by construction.
    return torch.searchsorted(cumsum, positions).clamp(max=num_particles - 1)


def _gather_rows(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """tensor [batch, K] gathered along dim=-1 by index [batch, K] -> [batch, K]."""
    return torch.gather(tensor, -1, index)


class FixedBarPointerPrior(nn.Module):
    """The FIXED (non-audio-conditioned) generative prior over (meter, log-tempo, bar-phase).

    Every parameter here is a GLOBAL nn.Parameter -- a function of nothing but time-step structure,
    never of the audio features. This is the deliberate faithfulness choice: the prior expresses only
    "how bar-pointers behave in general" (tempo drifts slowly, phase advances smoothly, meter rarely
    changes); the audio's influence comes ENTIRELY through the emission's importance weights in the
    particle filter, never by leaking into the transition itself.

    Matches model/bar_pointer_vae.py's generative story:
        m_t   ~ Categorical(transition from m_{t-1})            (global transition matrix)
        s_t   ~ Normal(s_{t-1}, sigma)                           (log tempo random walk; global sigma)
        phi_t ~ vonMises(phi_{t-1} + exp(s_{t-1}), kappa)        (phase advance; global kappa)

    All sample/step methods operate on [batch, num_particles] tensors (batch = sequences in the step).
    """

    def __init__(self, num_meters: int, init_log_tempo_mean: float = math.log(2.0 * math.pi / 172.27)):
        # Default bar-phase advance corresponds to a ~2.0s bar (~120 BPM at 4/4, 172.27 frames/bar
        # at the cached features' 86.13 fps) -- a generic "typical tempo" prior mean, NOT fit to any
        # song (that would violate the fixed-prior, non-audio-conditioned design choice). This replaces
        # an earlier placeholder that (by an arithmetic slip: 2*pi/60 treating "60" as a frame count,
        # not a BPM) implied an implausible ~60-frame (0.7s) bar period; fixed after the zero-condition
        # leak-test diagnostic below showed a too-fast, too-confident prior can look "audio-locked" by
        # accident (a very short assumed period is more likely to alias onto true beat times by chance).
        super().__init__()
        self.num_meters = num_meters
        # log_tempo random-walk step std (learned, global, in log-radians/frame units).
        self.log_tempo_sigma_raw = nn.Parameter(torch.tensor(-3.0))     # softplus -> ~0.05 initially
        # von Mises phase concentration (learned, global): how tightly phase tracks phi_prev + exp(s).
        self.phase_kappa_raw = nn.Parameter(torch.tensor(2.0))          # softplus -> ~40 initially (tight)
        # meter self-transition strength: meter changes are rare, so init strongly diagonal.
        self.meter_transition_logits = nn.Parameter(torch.eye(num_meters) * 4.0)
        # Initial-frame prior: broad/uninformative (we don't know the phase or tempo before hearing anything).
        self.init_log_tempo_mean = nn.Parameter(torch.tensor(init_log_tempo_mean))
        self.init_log_tempo_std_raw = nn.Parameter(torch.tensor(-1.0))
        self.init_phase_kappa_raw = nn.Parameter(torch.tensor(-2.0))    # softplus -> ~0.1 (nearly uniform)

    @property
    def log_tempo_sigma(self) -> torch.Tensor:
        return F.softplus(self.log_tempo_sigma_raw) + 1e-3

    @property
    def phase_kappa(self) -> torch.Tensor:
        return F.softplus(self.phase_kappa_raw) + 0.01

    @property
    def init_log_tempo_std(self) -> torch.Tensor:
        return F.softplus(self.init_log_tempo_std_raw) + 1e-2

    @property
    def init_phase_kappa(self) -> torch.Tensor:
        return F.softplus(self.init_phase_kappa_raw) + 0.01

    def meter_transition_matrix(self) -> torch.Tensor:
        """[num_meters, num_meters] row-stochastic transition matrix (softmax over destination)."""
        return F.softmax(self.meter_transition_logits, dim=-1)

    def sample_initial(self, batch: int, num_particles: int, device: torch.device):
        """Sample [batch, K] particles' initial (meter, log_tempo, phase). meter is UNIFORM categorical."""
        shape = (batch, num_particles)
        meter = torch.randint(0, self.num_meters, shape, device=device)
        log_tempo = sample_normal(
            self.init_log_tempo_mean.expand(shape), self.init_log_tempo_std.expand(shape))
        phase = sample_von_mises(
            torch.zeros(shape, device=device), self.init_phase_kappa.expand(shape))
        return meter, log_tempo, phase

    def step(self, meter_prev: torch.Tensor, log_tempo_prev: torch.Tensor, phase_prev: torch.Tensor):
        """Propagate [batch, K] particles one frame under the fixed prior (bootstrap proposal = prior)."""
        shape = phase_prev.shape
        # meter: categorical transition, sampled (non-reparameterized; meter has no gradient path
        # through the discrete choice, matching the KL treatment already used for the fixed prior).
        transition_row = self.meter_transition_matrix()[meter_prev]              # [batch, K, num_meters]
        meter = torch.multinomial(
            transition_row.reshape(-1, self.num_meters), 1).squeeze(-1).reshape(shape)
        # log tempo: random walk
        log_tempo = sample_normal(log_tempo_prev, self.log_tempo_sigma.expand(shape))
        # phase: advance by exp(tempo), von Mises noise around that prediction
        predicted_phase = (phase_prev + torch.exp(log_tempo_prev.clamp(-10.0, 3.0))) % TWO_PI
        phase = sample_von_mises(predicted_phase, self.phase_kappa.expand(shape)) % TWO_PI
        return meter, log_tempo, phase


class GeometricBumpEmission(nn.Module):
    """Expected (beat, downbeat) activation as a function of phase -- compared to the trained emission
    head's actual output via a Gaussian observation model. This is the particle filter's likelihood.

    expected_beat(phi)     = bump centered at each of the `beats_per_bar` beat sub-phases
    expected_downbeat(phi) = bump centered at phase 0 (the downbeat)
    Learned amplitude/bias/sharpness (like model/divergences.py's GeometricEmission) so the bump's
    scale can match the emission head's actual dynamic range.
    """

    def __init__(self, beats_per_bar: int):
        super().__init__()
        self.beats_per_bar = beats_per_bar
        self.beat_kappa_raw = nn.Parameter(torch.tensor(2.0))
        self.downbeat_kappa_raw = nn.Parameter(torch.tensor(2.0))
        self.beat_amplitude_raw = nn.Parameter(torch.tensor(0.0))
        self.downbeat_amplitude_raw = nn.Parameter(torch.tensor(0.0))
        self.beat_floor_raw = nn.Parameter(torch.tensor(-2.0))
        self.downbeat_floor_raw = nn.Parameter(torch.tensor(-2.0))
        self.log_obs_sigma = nn.Parameter(torch.tensor(math.log(0.25)))

    def expected_activation(self, phase: torch.Tensor) -> torch.Tensor:
        """phase [...] -> expected [..., 2] (p_beat, p_downbeat) bump, values in (floor, floor+amp)."""
        beat_kappa = F.softplus(self.beat_kappa_raw) + 0.5
        downbeat_kappa = F.softplus(self.downbeat_kappa_raw) + 0.5
        beat_bump = torch.exp(beat_kappa * (torch.cos(self.beats_per_bar * phase) - 1.0))
        downbeat_bump = torch.exp(downbeat_kappa * (torch.cos(phase) - 1.0))
        beat_amplitude = F.softplus(self.beat_amplitude_raw)
        downbeat_amplitude = F.softplus(self.downbeat_amplitude_raw)
        beat_floor = torch.sigmoid(self.beat_floor_raw)
        downbeat_floor = torch.sigmoid(self.downbeat_floor_raw)
        expected_beat = beat_floor + beat_amplitude * beat_bump
        expected_downbeat = downbeat_floor + downbeat_amplitude * downbeat_bump
        return torch.stack([expected_beat, expected_downbeat], dim=-1)

    def log_likelihood(self, phase: torch.Tensor, observed_activation: torch.Tensor) -> torch.Tensor:
        """log N(observed_activation; expected_activation(phase), sigma^2).

        phase: [batch, K]. observed_activation: [batch, 2] (one observation per sequence, broadcast
        over that sequence's K particles). Returns [batch, K].
        """
        sigma = torch.exp(self.log_obs_sigma).clamp(min=1e-3)
        expected = self.expected_activation(phase)                     # [batch, K, 2]
        squared_error = (expected - observed_activation.unsqueeze(1)).pow(2).sum(dim=-1)   # [batch, K]
        return -squared_error / (2.0 * sigma * sigma) - math.log(TWO_PI) - 2.0 * torch.log(sigma)


@dataclass
class FIVOResult:
    bound: torch.Tensor            # [batch] FIVO lower-bound estimate per sequence (to MAXIMIZE)
    mean_ess_fraction: float       # diagnostic: average ESS/K over the sequence(s)
    num_resamples: float           # diagnostic: mean resamples per sequence
    phase_trace: torch.Tensor      # [batch, T] weighted-circular-mean phase (for inspection only)


class BarPointerFIVO(nn.Module):
    """Wires the fixed prior + geometric-bump emission into a trainable bootstrap-FIVO particle filter.

    All methods process a BATCH of sequences at once: shapes are [batch, num_particles] internally,
    [batch, T, 2] for the observed activations. The time loop (length T) is the only Python loop --
    the batch and particle dimensions are both vectorized, which is what makes this fast enough to
    train (kernel-launch overhead, not FLOPs, dominates at these tensor sizes).
    """

    def __init__(self, num_meters: int, beats_per_bar: int):
        super().__init__()
        self.prior = FixedBarPointerPrior(num_meters)
        self.emission = GeometricBumpEmission(beats_per_bar)
        self.beats_per_bar = beats_per_bar

    def fivo_bound(self, observed_activations: torch.Tensor, num_particles: int,
                   ess_frac: float = 0.5) -> FIVOResult:
        """Run the bootstrap particle filter over a BATCH of sequences [batch, T, 2] (independent
        particle sets per sequence, sharing only the Python time-loop) and return the FIVO bound.

        FIVO bound = sum_t log( mean_k w_tilde_{t,k} ) per sequence, where w_tilde is the incremental
        (post-previous-resample) importance weight -- the biased-but-consistent lower bound of
        Maddison et al. 2017 / Naesseth et al. 2018. Because the proposal IS the prior (bootstrap),
        the incremental weight is exactly the emission likelihood: w_tilde_t = p(o_t | z_t).
        """
        batch, num_frames, _ = observed_activations.shape
        device = observed_activations.device

        meter, log_tempo, phase = self.prior.sample_initial(batch, num_particles, device)
        log_likelihood = self.emission.log_likelihood(phase, observed_activations[:, 0])   # [batch, K]
        log_mean_weight = torch.logsumexp(log_likelihood, dim=-1) - math.log(num_particles)   # [batch]
        bound = log_mean_weight.clone()
        log_weight = log_likelihood - log_mean_weight.detach().unsqueeze(-1)

        phase_trace = [phase.detach()]
        ess_fraction_sum, ess_fraction_count = 0.0, 0
        resample_count = torch.zeros(batch, device=device)

        for t in range(1, num_frames):
            weight = torch.softmax(log_weight, dim=-1)                       # [batch, K]
            ess = 1.0 / (weight * weight).sum(dim=-1).clamp(min=1e-12)       # [batch]
            ess_fraction_sum += float((ess / num_particles).sum().item())
            ess_fraction_count += batch
            needs_resample = ess < ess_frac * num_particles                  # [batch] bool

            if needs_resample.any():
                ancestor_index = systematic_resample_batched(weight.detach())    # [batch, K]
                resampled_meter = _gather_rows(meter, ancestor_index)
                resampled_log_tempo = _gather_rows(log_tempo, ancestor_index)
                resampled_phase = _gather_rows(phase, ancestor_index)
                mask = needs_resample.unsqueeze(-1)
                meter = torch.where(mask, resampled_meter, meter)
                log_tempo = torch.where(mask, resampled_log_tempo, log_tempo)
                phase = torch.where(mask, resampled_phase, phase)
                log_weight = torch.where(mask, torch.zeros_like(log_weight), log_weight)
                resample_count = resample_count + needs_resample.float()

            meter, log_tempo, phase = self.prior.step(meter, log_tempo, phase)
            log_likelihood = self.emission.log_likelihood(phase, observed_activations[:, t])   # [batch, K]

            combined_log_weight = log_weight + log_likelihood
            log_mean_weight = torch.logsumexp(combined_log_weight, dim=-1) - math.log(num_particles)
            bound = bound + log_mean_weight
            log_weight = combined_log_weight - log_mean_weight.detach().unsqueeze(-1)

            phase_trace.append(phase.detach())

        weight = torch.softmax(log_weight, dim=-1)
        ess_fraction_sum += float(
            (1.0 / (weight * weight).sum(dim=-1).clamp(min=1e-12) / num_particles).sum().item())
        ess_fraction_count += batch

        phase_trace_tensor = torch.stack(phase_trace, dim=1)    # [batch, T, K]
        weighted_cos = (weight.unsqueeze(1) * torch.cos(phase_trace_tensor)).sum(-1)
        weighted_sin = (weight.unsqueeze(1) * torch.sin(phase_trace_tensor)).sum(-1)
        mean_phase = torch.atan2(weighted_sin, weighted_cos) % TWO_PI       # [batch, T]

        return FIVOResult(
            bound=bound, mean_ess_fraction=ess_fraction_sum / ess_fraction_count,
            num_resamples=float(resample_count.mean().item()), phase_trace=mean_phase,
        )

    @torch.no_grad()
    def deploy_smc(self, observed_activations: torch.Tensor, num_particles: int,
                   ess_frac: float = 0.5, readout: str = "map") -> torch.Tensor:
        """Inference-time particle filter for ONE sequence (no labels, no teacher forcing): same
        dynamics/emission as training, run under no_grad. Returns the phase trajectory [T] used for
        the geometric read-out.

        readout="map"          -> genealogical trajectory of the single highest-final-weight particle
                                   (matches the archived model/svt_core.py `sample_from_prior_pf`).
        readout="weighted_mean"-> per-frame circular weighted mean of the filtering posterior.
        """
        observed_activations = observed_activations.unsqueeze(0)     # [1, T, 2] -- batch=1
        num_frames = observed_activations.shape[1]
        device = observed_activations.device
        batch = 1

        meter, log_tempo, phase = self.prior.sample_initial(batch, num_particles, device)
        log_likelihood = self.emission.log_likelihood(phase, observed_activations[:, 0])
        log_weight = log_likelihood - torch.logsumexp(log_likelihood, dim=-1, keepdim=True)

        phase_history = torch.zeros(num_frames, num_particles, device=device)
        phase_history[0] = phase[0]
        weight_history = torch.zeros(num_frames, num_particles, device=device)
        weight_history[0] = torch.softmax(log_weight, dim=-1)[0]
        ancestor_history = torch.arange(num_particles, device=device).unsqueeze(0).repeat(num_frames, 1)

        for t in range(1, num_frames):
            weight = torch.softmax(log_weight, dim=-1)
            ess = 1.0 / (weight * weight).sum(dim=-1).clamp(min=1e-12)
            if ess.item() < ess_frac * num_particles and t < num_frames - 1:
                ancestor_index = systematic_resample_batched(weight)         # [1, K]
                meter = _gather_rows(meter, ancestor_index)
                log_tempo = _gather_rows(log_tempo, ancestor_index)
                phase = _gather_rows(phase, ancestor_index)
                ancestor_history[:t] = ancestor_history[:t].gather(1, ancestor_index[0].unsqueeze(0).expand(t, -1))
                log_weight = torch.zeros_like(log_weight)

            meter, log_tempo, phase = self.prior.step(meter, log_tempo, phase)
            log_likelihood = self.emission.log_likelihood(phase, observed_activations[:, t])
            log_weight = log_weight + log_likelihood
            log_weight = log_weight - torch.logsumexp(log_weight, dim=-1, keepdim=True)

            phase_history[t] = phase[0]
            weight_history[t] = torch.softmax(log_weight, dim=-1)[0]
            ancestor_history[t] = torch.arange(num_particles, device=device)

        if readout == "map":
            final_weight = weight_history[-1]
            best = torch.argmax(final_weight)
            trajectory_index = ancestor_history[:, best]        # [T] -- genealogy of the MAP particle
            phase_trajectory = phase_history.gather(1, trajectory_index.unsqueeze(1)).squeeze(1)
            return phase_trajectory
        # weighted circular mean per frame (filtering posterior, not a single genealogy)
        weighted_cos = (weight_history * torch.cos(phase_history)).sum(-1)
        weighted_sin = (weight_history * torch.sin(phase_history)).sum(-1)
        return torch.atan2(weighted_sin, weighted_cos) % TWO_PI
