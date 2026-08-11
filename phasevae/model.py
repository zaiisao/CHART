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

from .vonmises import log_i0, mean_resultant, sample_vonmises, second_resultant


TWO_PI = 2.0 * math.pi

# The physical prior's concentration: how much per-frame phase drift the dynamics allow.
# 1/sqrt(kappa) is the per-frame sd, so 2000 gives 0.022 rad at 50 fps against a bar
# advance of roughly 0.06 rad. A much softer prior (kappa ~ 20, sd 0.22 rad) is a random
# walk whose phase is uniform within two seconds and regularises nothing.
KAPPA_PHYSICAL = 383.0
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


def vonmises_log_density(z, mu, kappa):
    """Per-element log vM(z; mu, kappa): kappa cos(z - mu) - log(2 pi I0(kappa))."""
    return kappa * torch.cos(z - mu) - math.log(TWO_PI) - log_i0(kappa)


def vonmises_entropy(kappa):
    """H(vM(mu, kappa)) = log(2 pi I0(kappa)) - kappa A(kappa). Independent of mu."""
    return math.log(TWO_PI) + log_i0(kappa) - kappa * mean_resultant(kappa)


class Encoder(nn.Module):
    """q_phi(phi_t | x) = vM(mu_t(x), kappa_t(x)), per frame, reading AUDIO ONLY.

    Tutorial 7.3.2: the encoder depends only on x, which is what makes it the object
    DEPLOYED at inference rather than training scaffolding.

    The mean is emitted as a 2-vector and read back with atan2, so it stays continuous
    across the wrap: a scalar head would have to jump by 2 pi somewhere on the circle,
    and wherever that seam sits the network must learn an impossible cliff there. Two
    nearly identical windows whose true offsets are 0.01 and 6.27 rad would demand
    outputs 6.26 apart; a smooth head splits the difference at pi, the worst possible
    answer -- which is how rule g once scored F 0.000 on ground truth. Nothing forces
    the 2-vector onto the unit circle: atan2 is scale-invariant, so the radius is a
    gauge. It matters only for conditioning -- the gradient scales as 1/r -- which is
    why bias[1] starts at 1.0 rather than 0.
    """

    def __init__(self, input_dim: int, d_model: int = 128, heads: int = 4, layers: int = 2,
                 kappa_physical: float = KAPPA_PHYSICAL, pool_span: int = 150,
                 max_len: int = 4096):
        super().__init__()

        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model,
                                        dropout=0.0, activation="gelu",
                                        batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d_model, 4)
        self.pool_span = int(pool_span)

        nn.init.normal_(self.out.weight, std=1e-2)
        nn.init.zeros_(self.out.bias)

        with torch.no_grad():
            self.out.bias[1] = 1.0

        self.register_buffer("pe", EmissionTransformer._sinusoidal(max_len, d_model))
        self.register_buffer("log_kappa_bias",
                     torch.tensor(math.log(kappa_physical)), persistent=False)
        self.register_buffer("log_rate_bias",
                     torch.tensor(math.log(TWO_PI / (3.0 * 50.0))), persistent=False)

    def features(self, h, mask=None):
        """[B, T, D] -> [B, T, d_model]: the trunk shared by every head (tutorial 9.2).

        Split from ``heads`` so a variant module can attach extra heads to the same body
        without duplicating it (see variants/psi.py). ``mask`` is audio-length validity;
        it must reach attention, because self-attention is not causal -- without it every
        real frame attends to the zero-padded tail, and the offset head reads frame 0.
        """
        pad = None if mask is None else (mask <= 0)
        h = self.proj(h) + self.pe[:h.shape[1]]
        return self.blocks(h, src_key_padding_mask=pad)

    def heads(self, trunk):
        """Trunk -> (mu [B, T], kappa [B, T]): the parameters of q, not a sample.

        Channels 0,1 are the (sin, cos) of the phase offset -- note atan2's argument
        order, which the previous comment here had reversed; bias[1] = 1.0 therefore
        starts every network at angle 0, the physical prior's mode. Channel 2 is the
        concentration in LOG space (tutorial 9.4's convention for the spread head), so
        kappa = kappa_physical * exp(raw): raw 0 gives the prior's concentration and the
        whole useful range is within +-5. The additive form this replaced needed raw
        -1990 to reach kappa 10, i.e. kappa could not move.
        """
        out = self.out(trunk)
        kappa = bounded_kappa(torch.exp(out[..., 2] + self.log_kappa_bias) + 1e-3)

        log_rate = self._pool(out[..., 3] + self.log_rate_bias, self.pool_span)
        rate = torch.exp(log_rate.clamp(math.log(0.01), math.log(0.2)))
        offset = torch.atan2(out[:, 0, 0], out[:, 0, 1]).unsqueeze(1)
        mu = offset + (torch.cumsum(rate, dim=1) - rate[:, :1])

        return mu, kappa

    @staticmethod
    def _pool(x, span):
        """Mean over fixed blocks of `span` frames, broadcast back. Deletes the degrees of
        freedom rather than taxing them: within-span increment variance becomes exactly 0."""
        b, t = x.shape
        pad = (-t) % span
        xp = torch.nn.functional.pad(x, (0, pad))
        means = xp.reshape(b, -1, span).mean(-1, keepdim=True)
        return means.expand(-1, -1, span).reshape(b, -1)[:, :t]

    def forward(self, h, mask=None):
        """[B, T, D] -> (mu, kappa). Per-frame and free: see kl_to_physical_prior.

        Nothing here ties consecutive mu together. That is deliberate under §7 -- the
        tutorial prescribes no phase dynamics -- but it means the KL is the ONLY force
        making the trajectory coherent, and a free init has second differences of order
        1 rad. MEASURED: the KL opens at ~151k against a reconstruction of ~286, and the
        steepest descent is to flatten mu.
        """
        return self.heads(self.features(h, mask))


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

    def __init__(self, input_dim: int, d_model: int = 128, emission: str = "cosine",
                 emission_layers: int = 2, emission_dim: int = 64,
                 emission_positional: bool = False, kappa_physical: float = KAPPA_PHYSICAL):
        super().__init__()
        self.kappa_physical = float(kappa_physical)
        self.encoder = Encoder(input_dim, d_model, kappa_physical=kappa_physical)

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

    def kl_to_physical_prior(self, mu, kappa, mask):
        """KL( q_phi(phi_{1:T} | x) || p(phi_{1:T}) ), closed form -- never sampled.

        q is factorised over frames and p is a Markov chain, so the cross term is an
        expectation of cos(phi_t - phi_{t-1} - delta) under two INDEPENDENT von Mises
        variables. For phi ~ vM(mu, kappa), E[e^{i phi}] = A(kappa) e^{i mu}, so that
        expectation is A(kappa_t) A(kappa_{t-1}) cos(mu_t - mu_{t-1} - delta): the
        concentrations enter as a product of mean resultant lengths. Everything below is
        that identity plus the von Mises entropy.
        """
        kp = torch.as_tensor(self.kappa_physical, device=mu.device)
        a1, a2 = mean_resultant(kappa), second_resultant(kappa)

        inc = mu[:, 1:] - mu[:, :-1]
        accel = inc[:, 1:] - inc[:, :-1]                        # [B, T-2]

        # E_q[log p(phi_t | phi_{t-1}, phi_{t-2})] for t >= 3
        cross = (kp * a1[:, 2:] * a2[:, 1:-1] * a1[:, :-2] * torch.cos(accel)
                 - math.log(TWO_PI) - log_i0(kp))

        neg_entropy = -(vonmises_entropy(kappa) * mask).sum(1)
        log_p = (-math.log(TWO_PI) * (mask[:, 0] + mask[:, 1])
                 + (cross * mask[:, 2:]).sum(1))
        return neg_entropy - log_p

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0,
                targets=None, valid=None, sigma_frames: float = 3.5):
        """One ELBO evaluation on a padded batch (tutorial 7.7).

        The OBSERVATION is the annotated downbeat TIMES, not the per-frame indicator:
        see align_log_prob for why, and note that both encode the same annotation, so
        this is a change of sufficient statistic rather than extra supervision. The
        bound is intact -- align_log_prob is a normalised density over those times.

        Args:
            h: [B, T, D] audio features -- the ONLY input the encoder sees.
            mask: [B, T] 1 on real frames.
            y: [B, T] per-frame downbeat target. RETAINED for the emission read-out and
                for variants that still use a per-frame Bernoulli; the base objective no
                longer consumes it.
            samples: Monte Carlo samples for the reconstruction term.
            pos_weight: unused by this objective, kept so the hook signature is stable
                across variants. It exists because a per-frame Bernoulli on a ~3.2%
                positive target lets a phase-IGNORING constant predictor capture most of
                the achievable likelihood; an alignment likelihood has no negative class
                to drown in, so there is nothing to reweight -- and with it goes the
                "weighted surrogate, not a bound" caveat.
            targets: [B, K] annotated downbeat times as window-relative FRAME indices,
                from collate_excerpts. valid: [B, K] their mask.
            sigma_frames: timing tolerance in frames; 3.5 at 50 fps = the F tolerance.

        Returns:
            dict with elbo, recon, kl (per crop, [B]) and the encoder's mu and kappa.
        """
        assert targets is not None and valid is not None, \
            "the alignment objective needs collate_excerpts' targets/valid"

        mu, kappa = self.encoder(h, mask)
        kl = self.kl_to_physical_prior(mu, kappa, mask)

        recon = 0.0
        for _ in range(samples):
            phi = mu + sample_vonmises(kappa)
            # velocity=mu: q is per-frame factorised, so the SAMPLE has no usable
            # derivative and it appears here as a divisor. See align_log_prob.
            recon = recon + align_log_prob(phi, targets, valid, sigma_frames,
                                           velocity=mu)

        recon = recon / samples

        return {"elbo": recon - kl, "recon": recon, "kl": kl, "mu": mu, "kappa": kappa}

    @property
    def deployed_net(self):
        """The inference network read at test time; controls assert ITS target-blindness.

        The base model deploys the encoder (tutorial 8.1.2). A variant that deploys a
        different object (e.g. a conditional prior) overrides this property, and the
        controls follow it without knowing the variant exists.
        """
        return self.encoder

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        """Deployment (8.1.2): zhat = mu_phi(x). Reads audio only; returns [B, T].

        ``delta`` is the GIVEN bar rate, not an annotation of phase -- the model is handed
        the bar period and has to find the bar's position, which is the whole task.
        ``mask`` is audio-length validity (which frames are real vs window padding),
        ignored by this family; a variant whose deployed statistic POOLS over frames
        (anchor_k's slot vote) must exclude pad frames or untrained pad responses
        outvote the decision margin on every short-song window (33% of every gtzan
        item) -- found by the anchor_k pre-launch review.
        """
        assert not self.training, "deployment path must run in eval mode"
        return self.encoder(h, mask)[0]

    @torch.no_grad()
    def emission_probs(self, h, mask=None):
        """Alternative D (8.3.4): the emission evaluated at the deployed mean path.

        delta must be passed whenever the trajectory family uses it (drift_bound > 0):
        omitting it silently built the path without its base advance -- the audit found
        every bounded arm's emission-D column had been scored on that broken path.
        """
        return torch.sigmoid(self.emission_logits(self.infer_phase(h, mask), mask))


def align_log_prob(mu, targets, valid, sigma_frames: float = 3.5, velocity=None):
    """log p(D | phi): the annotated downbeat TIMES as the observation. [B, T], [B, K] -> [B]

    The per-frame Bernoulli asks "is this frame a downbeat" 2250 times, of which ~97% are
    negatives whose pulls cancel in pairs; the alignment information survives only when
    EVERY downbeat lands on a peak at once, which needs the period right to ~0.2%. Measured
    on one ballroom song: the true rate wins by 97 nats, but in a well 0.2% wide sitting in
    a plain that is flat to +-4 nats over a 5x range. There is no gradient to the answer.

    This asks a different question of the SAME annotation: the k-th bar line occurs where
    the phase has turned exactly k times. Two consequences.

      * CONVEX. For mu = offset + rate*t the cost is sum_k (offset + rate*t_k - 2 pi k)^2,
        least squares in (offset, rate): one global minimum, usable gradient everywhere.
      * OCTAVE-BREAKING. k is an integer COUNT, so half rate gives mu(t_11) = 11 pi against
        22 pi -- an error of eleven turns that GROWS with k. Every circular formulation is
        blind to this (each octave puts the downbeats at phi = 0 mod 2 pi and scores the
        same: measured 133.6 / 133.8 / 134.0 nats at 1x / 2x / 3x).

    Normalised, so recon - beta*KL stays a bound: this is a Gaussian density over the
    OBSERVED TIMES. The phase residual is divided by the local phase velocity, which is the
    change of variables from phase to time -- that division is what keeps the rate honest
    and is the same Jacobian a point-process form carries as log(phidot).

    Args:
        mu: [B, T] the UNWRAPPED phase. Wrapping destroys the index and hands the octave
            degeneracy straight back.
        targets: [B, K] annotated downbeat times as (possibly fractional) FRAME indices,
            window-relative -- (downbeat_times - t0) * fps. Take these from
            ``downbeat_times``, NEVER from nonzero(y > 0.5): y is the +-target_tol_frames
            WIDENED indicator, so that gives 3 (or 7) events per downbeat and inflates any
            derived rate by the same factor.
        valid: [B, K] 1 for real entries, 0 for padding.
        sigma_frames: timing tolerance in FRAMES. 3.5 at 50 fps = 70 ms = the F tolerance,
            which is the point: this is the same quantity the metric uses, on a continuous
            scale, with no effect on class balance. It replaces target_tol_frames.

    Returns:
        [B] log-likelihood per crop.
    """
    B, T = mu.shape
    idx = targets.clamp(0.0, T - 2.0)
    lo = idx.floor()
    frac = idx - lo
    lo = lo.long()

    mu_lo = torch.gather(mu, 1, lo)
    mu_hi = torch.gather(mu, 1, lo + 1)
    mu_at = mu_lo + frac * (mu_hi - mu_lo)          # phase at t_k, sub-frame

    # The VELOCITY comes from `velocity`, not from mu, whenever mu is a SAMPLE. q is
    # factorised over frames, so a sampled trajectory has no meaningful derivative: at
    # kappa 383 the per-frame noise is 0.051 rad and a first difference of two draws has
    # sd 0.072, against a real advance of ~0.05. Taken from the sample this divisor is
    # noise twice the size of its own signal, and it can cross zero -- the clamp would
    # then hand back a 1e6 residual. Pass the encoder's mean path.
    src = mu if velocity is None else velocity
    v_lo = torch.gather(src, 1, lo)
    v_hi = torch.gather(src, 1, lo + 1)
    rate_at = (v_hi - v_lo).clamp(min=1e-6)         # local phase velocity, rad/frame

    # k counts downbeats within the window, and BOTH SIDES ARE CENTRED so the term is
    # invariant to a constant phase shift. Which absolute bar a window starts on is
    # arbitrary, so pinning mu(t_0) = 0 imposes a constraint the data never asked for --
    # and the offset head cannot meet it: it emits atan2, bounded to (-pi, pi], while
    # t_0 lands anywhere within the first bar, so rate * t_0 is roughly uniform on
    # [0, 2 pi) and 47.1% of songs need an intercept outside the representable range.
    # (The encoder can fake one by bending the first pooling block's rate, which is why
    # this is a needless constraint rather than a hard wall.) Centring deletes the
    # intercept from the problem; placing the bar line stays the anchor's job.
    k = torch.cumsum(valid, dim=1) - 1.0
    n = valid.sum(1, keepdim=True).clamp(min=1.0)

    raw = mu_at - TWO_PI * k
    c = (raw * valid).sum(1, keepdim=True) / n          # the window's mean offset
    # Split it: the INTEGER part of c is a free gauge (which absolute bar the window
    # starts on is arbitrary), the SUB-BAR part is not (it is where the bar line sits).
    # Centring alone deletes both and the anchor loses its only signal -- measured, the
    # rate converged faster and in-tol fell to 0%. So subtract c to free the shape, then
    # add back its wrapped remainder, which is 0 exactly when c is a whole number of
    # turns and grows to +-pi otherwise.
    phase_err = (raw - c) + torch.atan2(torch.sin(c), torch.cos(c))
    dt = phase_err / rate_at                        # phase residual -> TIME residual

    log_density = (-0.5 * (dt / sigma_frames) ** 2
                   - math.log(sigma_frames) - 0.5 * math.log(TWO_PI))
    return (log_density * valid).sum(1)


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
