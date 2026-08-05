"""The bar-pointer CVAE of the tutorial's section 7, with a periodic latent (section 9.9).

Three structural commitments, all taken from the tutorial rather than invented here:

**Point 1 (7.3.1) -- the generative model does not involve x.** The bar-pointer state
causally determines the downbeats; the audio was never a generative parent of them. So the
prior is p(z), FIXED and physical, with no learnable parameters and no audio input, and the
emission is p_theta(b | z) reading the latent alone. An emission that also read the audio
would let it fit the target directly and leave the latent unused.

**Point 2 (7.3.2) -- the encoder depends only on x.** The optimal recognition network would
be q(z | x, b), but b is exactly what we predict, so it is unavailable at deployment. Using
q_phi(z | x) from the outset makes the trained encoder the DEPLOYED encoder, and there is
no training-inference gap to remedy. This is the whole reason the model is shaped this way.

**Consequence: only theta and phi are learnable.** The CVAE's third parameter set psi is
absent -- there is no conditional prior network.

    p(z):        phi_1 ~ Uniform[0, 2pi),  phi_t = phi_{t-1} + delta + eps,
                 eps ~ vM(0, KAPPA_PHYSICAL)          -- fixed, no parameters, no audio
    q_phi(z|x):  phi_t ~ vM(mu_t(x), kappa_t(x))      -- per frame, factorised (9.9.2)
    p_theta(b|z): logit = a + b cos(phi_t),  b > 0    -- two scalars, latent only

    ELBO (7.7) = E_q[log p_theta(b | z)] - KL(q_phi(z | x) || p(z))

At inference (8.1.2) the encoder mean IS the answer: zhat = mu_phi(x), then the
deterministic rule g reads downbeats off the phase trajectory. The prior and the emission
are training scaffolding (8.1.4).

Note what this formulation does NOT need. An earlier build gave the prior an
audio-conditioned initial phase because a generative rollout from a uniform phi_1 is
exactly invariant to a global rotation, making the phase unidentifiable. That invariance
is a property of SAMPLING THE PRIOR at inference. Here inference reads the encoder, which
sees the audio at every frame, so the symmetry is broken by construction and no anchor
term is required.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .vonmises import log_i0, mean_resultant, sample_vonmises

TWO_PI = 2.0 * math.pi

# The physical prior's concentration: how much per-frame phase drift the dynamics allow.
# 1/sqrt(kappa) is the per-frame sd, so 2000 gives 0.022 rad at 50 fps against a bar
# advance of roughly 0.06 rad. A much softer prior (kappa ~ 20, sd 0.22 rad) is a random
# walk whose phase is uniform within two seconds and regularises nothing.
KAPPA_PHYSICAL = 2000.0
MAX_KAPPA = 1.0e5     # smooth ceiling on the encoder's concentration; an unbounded
                      # softplus inside a KL term can and did run away. Bounded via
                      # MAX * tanh(x / MAX), which is the identity for x << MAX and never
                      # passes exactly zero gradient -- unlike a hard clamp.


def bounded_kappa(raw):
    """Smoothly bound a concentration to (0, MAX_KAPPA); see MAX_KAPPA."""
    return MAX_KAPPA * torch.tanh(raw / MAX_KAPPA)


def inverse_softplus(value: float) -> float:
    """Pre-activation giving softplus(x) = value; linear above 30 where expm1 overflows."""
    return value if value > 30.0 else math.log(math.expm1(value))


def vonmises_entropy(kappa):
    """H(vM(mu, kappa)) = log(2 pi I0(kappa)) - kappa A(kappa). Independent of mu."""
    return math.log(TWO_PI) + log_i0(kappa) - kappa * mean_resultant(kappa)


class Encoder(nn.Module):
    """q_phi(phi_t | x) = vM(mu_t(x), kappa_t(x)), per frame, reading AUDIO ONLY.

    Tutorial 7.3.2 and 9.9.2. It never sees the target, which is what makes it the object
    deployed at inference rather than training scaffolding. The mean is emitted as a
    (cos, sin) pair and read back with atan2, so it is continuous across the wrap -- a
    scalar head would have to jump by 2 pi somewhere on the circle.
    """

    def __init__(self, input_dim: int, hidden: int = 128, layers: int = 2,
                 drift_bound: float = 0.0):
        super().__init__()
        self.drift_bound = drift_bound
        self.proj = nn.Linear(input_dim, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True)
        # channels: 0,1 = (cos, sin) of the crop's phase offset, read at frame 0;
        # 2 = concentration; 3 = per-frame drift, only when the bound is active
        self.out = nn.Linear(2 * hidden, 4 if drift_bound > 0 else 3)
        # small but NOT zero: a zero output layer makes d(loss)/d(body) exactly zero at
        # step 0, which is how this project shipped a dead subnetwork three times.
        nn.init.normal_(self.out.weight, std=1e-2)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("kappa_bias", torch.tensor(inverse_softplus(KAPPA_PHYSICAL)))

    def forward(self, h, delta=None):
        """[B, T, D] -> (mu [B, T], kappa [B, T]).

        With ``drift_bound > 0`` the phase is not emitted per frame. It is built as

            mu = offset(h) + cumsum(delta + eps * tanh(g_t(h)))

        so the per-frame advance cannot leave ``delta +- eps``. This is the classical
        bar pointer's hard lattice restriction, moved into the parameterisation.

        Why it exists: emitting mu per frame lets the encoder produce a SPIKE TRAIN --
        measured at 0.715 rad/frame advance against a true 0.065, dipping to phase 0 on
        downbeats (concentration 0.969) and doing anything in between (error 1.44 rad
        against 1.57 for chance). That satisfies a reconstruction which is nonzero on 3%
        of frames and says nothing about the other 97%. The von Mises prior only makes it
        expensive, and a 30x-weighted reconstruction outbids the penalty. Bounding the
        drift makes it unrepresentable instead. Free-q measurement: F 0.075 -> 0.911.
        """
        out = self.out(self.rnn(torch.tanh(self.proj(h)))[0])
        kappa = bounded_kappa(
            nn.functional.softplus(out[..., 2] + self.kappa_bias) + 1e-3)
        if self.drift_bound <= 0.0:
            return torch.atan2(out[..., 0], out[..., 1]), kappa
        offset = torch.atan2(out[:, :1, 0], out[:, :1, 1])       # ONE anchor per crop
        drift = self.drift_bound * torch.tanh(out[..., 3])       # its own channel
        advance = (delta if delta is not None else 0.0) + drift
        mu = offset + torch.cumsum(advance, dim=1) - advance[:, :1]
        return mu, kappa


class EmissionTransformer(nn.Module):
    """The tutorial's section 9.6 emission: a Transformer over the LATENT sequence.

    Faithful replacement for the two-scalar `a + b cos(phi)` that stood here. That was my
    invention, not the spec, and it is the term carrying all of the task signal: the KL is
    provably blind to absolute phase, so the emission is the only thing that can locate a
    bar. Compressing it to two parameters -- one of which the optimiser then flattens --
    handicapped exactly the component that had to do the work.

    Section 9.6, translated:
        z~_k = W_in z_k + PE_k ;  g = TransformerLayers(z~) ;  pi_k = sigma(W_out g_k)

    Two deliberate choices the spec leaves open:

    * **Phase enters as (cos, sin), never as a raw angle.** phi is circular; a scalar input
      has a discontinuity at the wrap that the network would have to learn around, and the
      wrap is exactly where the downbeat is.
    * **Positional encoding is a shortcut risk.** With PE and self-attention the emission
      could emit a periodic pattern from POSITION alone and ignore the latent entirely --
      the decoder shortcut Point 1 exists to prevent. `use_positional` is therefore a flag,
      default OFF, and `phase_ablation_gap` measures how much the output actually depends
      on phi. An emission that does not move when phi is frozen has taken the shortcut.
    """

    def __init__(self, d_model: int = 64, layers: int = 2, heads: int = 4,
                 use_positional: bool = False, max_len: int = 4096):
        super().__init__()
        self.proj = nn.Linear(2, d_model)            # (cos phi, sin phi) -- latent ONLY
        self.use_positional = use_positional
        if use_positional:
            self.register_buffer("pe", self._sinusoidal(max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model,
                                           dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d_model, 1)
        nn.init.normal_(self.out.weight, std=1e-2)
        nn.init.constant_(self.out.bias, -3.4)       # ~= the 3.2% base rate, so training
                                                     # starts calibrated instead of spending
                                                     # its first epochs finding the prior

    @staticmethod
    def _sinusoidal(length, dim):
        """Standard sinusoidal positional encoding [length, dim]."""
        pos = torch.arange(length, dtype=torch.float32)[:, None]
        rate = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                         * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * rate)
        pe[:, 1::2] = torch.cos(pos * rate)
        return pe

    def forward(self, phi, mask=None):
        """[B, T] phase -> [B, T] downbeat logits. Reads the LATENT only, never h."""
        x = self.proj(torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1))
        if self.use_positional:
            x = x + self.pe[:x.shape[1]]
        pad = None if mask is None else (mask <= 0)
        return self.out(self.blocks(x, src_key_padding_mask=pad)).squeeze(-1)


class BarPhaseVAE(nn.Module):
    """Encoder + physical prior + latent-only emission. Learnable: theta and phi only."""

    def __init__(self, input_dim: int, hidden: int = 128, emission: str = "cosine",
                 emission_layers: int = 2, emission_dim: int = 64,
                 emission_positional: bool = False, drift_bound: float = 0.0):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden, drift_bound=drift_bound)
        self.emission_kind = emission
        # ONLY the arm in use gets parameters. Registering both left the unused pair with
        # no gradient, which the audit correctly refused -- and an audit that has to be
        # weakened to accommodate dead parameters stops being an audit.
        self.emission_net = None
        if emission == "transformer":
            self.emission_net = EmissionTransformer(emission_dim, emission_layers,
                                                    use_positional=emission_positional)
        else:
            # the two-scalar cosine: the BASELINE arm, not the default story. It is my
            # invention rather than the spec's, and every number before 2026-08-04 was
            # measured with it.
            self.emission_a = nn.Parameter(torch.tensor(-3.0))
            self.emission_b_raw = nn.Parameter(torch.tensor(1.0))
        # Scheduled LOWER BOUND on the emission amplitude, set by the training loop
        # (0 = free). Rationale: broad-emission/imprecise-phase is a self-consistent
        # equilibrium -- while q's phase is off, the local gradient on b points DOWN
        # (measured: cosine b 1.35 -> 0.91; triangle settled at 2.1 with p_peak 0.14).
        # A rising floor forces the likelihood to sharpen anyway, so precision becomes
        # worth paying for. The mirror of beta annealing: that schedules how hard the
        # PRIOR binds, this schedules how much the DATA localises.
        #
        # A BUFFER, not a plain attribute: the floor is part of the trained likelihood,
        # and as an attribute it silently reset to 0 on every state_dict reload -- a
        # sharpened checkpoint re-evaluated flatter than the free arm. Found
        # independently by the audit and the blind test suite on the same day.
        self.register_buffer("emission_b_floor", torch.tensor(0.0))

    @property
    def emission_b(self):
        """Amplitude, positive by softplus, never below the scheduled floor."""
        return self.emission_b_floor + nn.functional.softplus(self.emission_b_raw)

    def emission_logits(self, phi, mask=None):
        """Downbeat logits from the LATENT alone (Point 1) -- never from h."""
        if self.emission_net is not None:
            return self.emission_net(phi, mask)
        if self.emission_kind == "triangle":
            # Tent in phase: 1 at phi = 0, -1 at phi = pi, LINEAR between. Same two
            # parameters and the same range as the cosine, different geometry where it
            # matters: d(logit)/d(phi) has CONSTANT magnitude 2b/pi everywhere, where
            # the cosine's b sin(phi) vanishes both at alignment (no precision) and at
            # half-bar error (no pull home). The pointwise Bernoulli likelihood is
            # metrically blind across frames; the emission's shape in phi is the only
            # carrier of "closer is better", so its tails must not flatten.
            wrapped = torch.atan2(torch.sin(phi), torch.cos(phi))   # (-pi, pi]
            return self.emission_a + self.emission_b * (1.0 - 2.0 * wrapped.abs() / math.pi)
        return self.emission_a + self.emission_b * torch.cos(phi)

    @torch.no_grad()
    def phase_ablation_gap(self, phi, mask=None):
        """How much the emission actually depends on phi. Near 0 means it has cheated.

        Compares the logits at the real phase against the logits at a FROZEN phase. A
        transformer with positional encoding can emit a periodic pattern from position
        alone and ignore the latent -- the decoder shortcut Point 1 forbids -- and that
        failure is invisible in the loss.
        """
        live = self.emission_logits(phi, mask)
        frozen = self.emission_logits(torch.zeros_like(phi), mask)
        weight = torch.ones_like(live) if mask is None else mask
        return float(((live - frozen).abs() * weight).sum() / weight.sum())

    def kl_to_physical_prior(self, mu, kappa, delta, mask):
        """KL( q_phi(phi_{1:T} | x) || p(phi_{1:T}) ), closed form -- never sampled.

        q is factorised over frames and p is a Markov chain, so the cross term is an
        expectation of cos(phi_t - phi_{t-1} - delta) under two INDEPENDENT von Mises
        variables. For phi ~ vM(mu, kappa), E[e^{i phi}] = A(kappa) e^{i mu}, so that
        expectation is A(kappa_t) A(kappa_{t-1}) cos(mu_t - mu_{t-1} - delta): the
        concentrations enter as a product of mean resultant lengths. Everything below is
        that identity plus the von Mises entropy.
        """
        entropy = vonmises_entropy(kappa)                       # -E_q[log q]
        log_p_first = -math.log(TWO_PI)                         # phi_1 ~ Uniform
        resultant = mean_resultant(kappa)
        drift = mu[:, 1:] - mu[:, :-1] - delta[:, 1:]
        cross = (KAPPA_PHYSICAL * resultant[:, 1:] * resultant[:, :-1] * torch.cos(drift)
                 - math.log(TWO_PI) - log_i0(torch.as_tensor(KAPPA_PHYSICAL,
                                                             device=mu.device)))
        neg_entropy = -(entropy * mask).sum(1)
        log_p = log_p_first * mask[:, 0] + (cross * mask[:, 1:]).sum(1)
        return neg_entropy - log_p

    def forward(self, h, delta, mask, y, samples: int = 1, pos_weight: float = 1.0):
        """One ELBO evaluation on a padded batch (tutorial 7.7).

        Args:
            h: [B, T, D] audio features -- the ONLY input the encoder sees.
            delta: [B, T] the per-frame bar-phase advance, one constant per crop.
            mask: [B, T] 1 on real frames.
            y: [B, T] per-frame downbeat target. Enters the RECONSTRUCTION only; the
                encoder never receives it.
            samples: Monte Carlo samples for the reconstruction term.
            pos_weight: multiplier on the downbeat-labelled frames. The target is ~3.2%
                positive, so an unweighted per-frame Bernoulli lets a phase-IGNORING
                constant predictor capture most of the achievable likelihood by being
                right about the 96.8% of empty frames. Measured: the whole value of
                knowing the phase perfectly rather than ignoring it is 0.051 nats/frame
                against a KL operating at 1.13. At pos_weight=30 that rises to 0.754.
                NOTE this makes the objective a weighted surrogate, NOT a bound on
                log p(y|h) -- the same caveat as beta != 1, and it must be reported.

        Returns:
            dict with elbo, recon, kl (per crop, [B]) and the encoder's mu and kappa.
        """
        mu, kappa = self.encoder(h, delta)
        kl = self.kl_to_physical_prior(mu, kappa, delta, mask)

        weight = torch.where(y > 0, torch.as_tensor(pos_weight, device=y.device,
                                                    dtype=torch.float32),
                             torch.ones((), device=y.device, dtype=torch.float32)) * mask
        recon = 0.0
        for _ in range(samples):
            phi = mu + sample_vonmises(kappa)
            per_frame = nn.functional.binary_cross_entropy_with_logits(
                self.emission_logits(phi), y.float(), reduction="none")
            recon = recon - (per_frame * weight).sum(1)
        recon = recon / samples
        return {"elbo": recon - kl, "recon": recon, "kl": kl, "mu": mu, "kappa": kappa}

    @torch.no_grad()
    def infer_phase(self, h, delta=None):
        """Deployment (8.1.2): zhat = mu_phi(x). Reads audio only; returns [B, T].

        ``delta`` is the GIVEN bar rate, not an annotation of phase -- the model is handed
        the bar period and has to find the bar's position, which is the whole task.
        """
        assert not self.training, "deployment path must run in eval mode"
        return self.encoder(h, delta)[0]

    @torch.no_grad()
    def emission_probs(self, h):
        """Alternative D (8.3.4): the emission evaluated at the encoder mean."""
        return torch.sigmoid(self.emission_logits(self.infer_phase(h)))


def downbeat_frames(mu, mask=None):
    """Rule g (8.1.2): a downbeat is where the phase crosses ZERO. Deterministic.

    The bar starts at phi = 0 -- that is where the emission a + b cos(phi) peaks and where
    crops.true_phase puts 0. So the read-out must detect the phi = 0 crossing.

    The subtlety that made this wrong for most of a day: the encoder emits mu through
    atan2, i.e. in (-pi, pi], whose DISCONTINUITY sits at phi = pi -- half a bar from the
    downbeat. Differencing mu directly therefore finds the half-bar points, and scores
    F = 0.000 against the truth while looking entirely reasonable. Mapping to [0, 2pi)
    first puts the discontinuity on phi = 0, where the downbeat actually is; that scores
    F = 0.995 on the oracle trajectory.

    There is exactly one implementation of this on purpose. The bug survived because a
    corrected copy was written into one caller while run.py went on importing the original.
    """
    zero_to_two_pi = torch.remainder(mu, 2.0 * math.pi)
    crossing = torch.diff(zero_to_two_pi, dim=-1) < -math.pi
    if mask is not None:
        crossing = crossing & (mask[:, 1:] > 0)
    return crossing
