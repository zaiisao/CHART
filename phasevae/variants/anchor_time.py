"""The anchor as a DISCRETE TIME grid, marginalised exactly: 80 ms bins over the window.

The bar pointer needs two things from the audio -- a rate and an anchor -- and measurement
says they cannot bootstrap each other. With a wrong anchor the reconstruction is flat
(-0.150 +- 0.005 across every rate tested, coherence <= 0.085) so the rate gradient carries
no signal; with a wrong rate the anchor is placed on a grid that spins at the wrong speed.
The base recipe's continuous offset head loses that race by construction: the anchor's
landscape is PERIODIC, with one near-identical maximum per bar (measured: 16.5 per 45 s
window, median, all within 8-16 ms of each other), and a single continuous point cannot
cross between them by gradient descent. Its measured outcome is phase_err 1.581 against a
chance of 1.571, i.e. exactly nothing, with advance pinned at the 0.2 clamp.

So the anchor is ENUMERATED here, not descended. k indexes a candidate ANCHOR TIME on a
fixed stride of whole frames, q(k|x) is categorical over the candidates, and k is
marginalised EXACTLY -- C closed-form emission evaluations, no Gumbel, no REINFORCE. The
gradient is dL/dlogit_j = q_j (R_j - sum_k q_k R_k): every candidate is priced every step,
so anchor discovery is by construction rather than by search.

WHY TIME AND NOT PHASE. The candidate is a time, on a grid whose spacing is fixed in
seconds, which makes the resolution INDEPENDENT of tempo and of input length: the half-bin
error is 40 ms for every item in the corpus. The phase-slot alternative (2 pi k / 64) is
proportional to the bar instead, so it is sharp at fast tempi and coarse at slow ones --
7 ms at a 1 s bar against 62.8 ms at the longest. Both stay inside the +-70 ms tolerance,
but only one of them does so uniformly. Nothing in this module refers to a bar length, a
number of bars, or the 45 s window; only the stride is a constant.

THE STRIDE IS AN INTEGER NUMBER OF FRAMES, and that is load-bearing. A candidate is then a
frame index, so its anchor value is READ OFF cum rather than interpolated into it, and the
whole construction stays exact. 14.3 bins/s (the tolerance divided out directly) would be
3.5 frames; 4 frames = 80 ms is the nearest integer and costs about a fifth of a point of F
against the ideal (hit rate 95% vs 96.5%, computed against the measured 27 ms anchor error).

Bin width is set by the MODEL's error budget, not by the tolerance alone. Quantization can
consume at most stride/2, and the tolerance is SHARED with the model's own anchor error, so
the widest ADMISSIBLE stride (140 ms, where quantization alone never misses) costs 14.4
points of hit rate against no quantization at all. 80 ms costs 3.5.

WHAT THIS MODULE DOES NOT CLAIM. The evidence head below is a frame-wise linear on the
frontend's features, which is the same functional form as the frontend's OWN downbeat head
(beat_this's task heads are frame-wise linears on exactly these features). It is therefore a
re-derived downbeat activation, and if the performance comes from it then the performance is
the frontend's. What this mechanism adds over peak-picking that activation is that its
OUTPUT SPACE is a periodic grid -- mu^k = cum - c_k -- so it cannot emit an isolated
downbeat and cannot skip one. Whether that constraint is worth anything is an empirical
question answered by one control this project has not yet run: the frontend alone, its own
head and its own peak-picker, on the same excerpts, compared on CMLt/AMLt rather than F.
The head is randomly initialised on purpose. It could be warm-started from the frontend's
downbeat linear, which would remove the cold start, but 82% of this project's last headline
number turned out to be exactly that activation, so a random init is what makes a positive
result attributable to this model. Keep the warm start as a diagnostic arm only.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..model import (BarPhaseVAE, Encoder, bounded_kappa, sample_vonmises)


# 4 frames at 50 fps = 80 ms; see the module docstring on why this is an integer.
STRIDE_FRAMES = 4
HARMONICS = 8
K_HEAD_HIDDEN = 32


def candidate_anchors(cum, mask, stride: int):
    """(c [B, C], ok [B, C]) -- candidate anchor VALUES, read off the trajectory.

    Candidate k is the frame index k * stride, and its anchor value is cum at that frame,
    so mu^k = cum - c_k is exactly 0 there. Because the stride is a whole number of frames
    the value is read, never interpolated, and the construction is exact.

    ``ok`` is the per-item candidate mask, and it is required for CORRECTNESS rather than
    tidiness: 62.6% of the corpus is shorter than the padded tensor (median valid fraction
    0.660, minimum 0.268 -- gtzan and ballroom are 30 s excerpts), and a candidate anchored
    in the padded tail reads a cum that does not exist. Without the mask q can put its mass
    there. Measured candidate counts on the annotated corpus: C_max 562 at 45 s, C_i median
    371, minimum 150.

    A fully-masked window keeps candidate 0 anyway. That is the backstop item run.py
    guarantees must cost 0 rather than produce nan, and with no valid candidate every logit
    would be -inf, so the softmax -- and through it the whole batch's loss -- goes nan.
    """
    idx = torch.arange(0, cum.shape[1], stride, device=cum.device)
    ok = mask[:, idx] > 0
    ok[:, 0] |= ~ok.any(-1)
    return cum[:, idx], ok


class AnchorEncoder(Encoder):
    """q(phi_t | x) with the OFFSET HEAD REMOVED: the anchor is enumerated, not emitted.

    Three output channels instead of four, because the base encoder's channels 0 and 1 --
    the (sin, cos) pair read back with atan2 -- exist solely to emit the phase offset, and
    that quantity is now the marginalised k. Leaving them would leave dead parameters, which
    this codebase refuses on the grounds that an audit weakened to accommodate them stops
    being an audit.

        0  log kappa       (as base: kappa = kappa_physical * exp(raw), raw 0 = the prior)
        1  log rate        (as base: pooled over pool_span, then clamped)
        2  residual        a BOUNDED sub-bin refinement, see AnchorTimeVAE.residual

    The atan2 construction is not missed. It exists because a phase offset is CIRCULAR and a
    scalar head would need an impossible 2 pi cliff somewhere on the circle -- the seam that
    once made rule g score F 0.000 on ground truth. Channel 2 is a small bounded residual,
    which is not circular, so a plain scalar is the right shape for it.

    ``heads`` returns the CUMULATIVE ROTATION rather than a mean path: cum is monotone with
    cum[0] = 0, and every candidate's mean path is cum minus a constant. That is what makes
    the trajectory KL shared across candidates (see AnchorTimeVAE.forward).
    """

    def __init__(self, input_dim: int, *args, **kw):
        super().__init__(input_dim, *args, **kw)
        self.out = nn.Linear(self.out.in_features, 3)
        nn.init.normal_(self.out.weight, std=1e-2)
        nn.init.zeros_(self.out.bias)          # raw 0 on every channel = kappa_physical,
                                               # the prior's rate, and residual 0

    def heads(self, trunk):
        """Trunk -> (cum [B, T], kappa [B, T], residual_raw [B, T]). Not a sample."""
        out = self.out(trunk)
        kappa = bounded_kappa(torch.exp(out[..., 0] + self.log_kappa_bias) + 1e-3)

        log_rate = self._pool(out[..., 1] + self.log_rate_bias, self.pool_span)
        rate = torch.exp(log_rate.clamp(math.log(0.01), math.log(0.2)))
        cum = torch.cumsum(rate, dim=1) - rate[:, :1]         # monotone, cum[:, 0] = 0

        return cum, kappa, out[..., 2]


class AnchorTimeVAE(BarPhaseVAE):
    """BarPhaseVAE whose anchor is a categorical over candidate anchor TIMES."""

    def __init__(self, input_dim: int, stride: int = STRIDE_FRAMES,
                 harmonics: int = HARMONICS, d_model: int = 128, **kw):
        super().__init__(input_dim, d_model=d_model, **kw)
        self.encoder = AnchorEncoder(input_dim, d_model,
                                     kappa_physical=self.kappa_physical)
        self.stride = int(stride)
        self.harmonics = int(harmonics)

        # OUR evidence head, over the frontend's features. Deliberately not the frontend's
        # own activation channels: see the module docstring on attribution, and note the
        # frontend's task heads are frame-wise linears on these same features, so this form
        # can represent them exactly -- only the initialisation differs.
        self.evidence = nn.Linear(input_dim, 1)
        self.k_head = nn.Sequential(nn.Linear(2 * harmonics, K_HEAD_HIDDEN),
                                    nn.GELU(),
                                    nn.Linear(K_HEAD_HIDDEN, 1))
        # The output layer keeps torch's DEFAULT init. An earlier std=1e-3 override, chosen
        # to start every candidate equally scored, combined with descriptors of ~1e-2 to
        # produce logits ~1e-5 and a permanently uniform q -- the head could not escape its
        # own initialisation. Near-uniform is not worth buying at the price of a dead head;
        # the pricing gradient does not need help starting from a flat posterior.
        nn.init.zeros_(self.k_head[-1].bias)

    # ------------------------------------------------------------------ the evidence

    def candidate_features(self, h, cum, c, mask):
        """[B, C, 2M] descriptors for every candidate, EXACTLY, in O(M T + M C).

        The m-th Fourier harmonic of candidate k's phase-folded activation histogram is
        available in closed form from a single pass over frames, because mu^k differs from
        cum by the constant c_k:

            sum_t a_t exp(i m (cum_t - c_k)) = exp(-i m c_k) * sum_t a_t exp(i m cum_t)

        So M sums over T give every candidate's descriptor by rotation. This matters beyond
        speed. A MATCHED FILTER is the m = 1 term with fixed weights, and since
        sum_t a_t cos(cum_t - c_k) is exactly sinusoidal in c_k, its argmax IS the circular
        mean of the activations -- i.e. a matched-filter categorical is the closed-form
        read-out, discretised, which is the component measured to carry 82% of this
        project's last headline with zero training. Letting a head read M harmonics is
        strictly more expressive, and that is where the +0.050 the learned head had over
        the closed form has to come from.

        The alternative -- soft-binning phase into a histogram per candidate -- needs a
        [B, C, T, P] tensor, 1.3 GB at P = 16 against 288 KB here, and hard binning would
        cut the rate's gradient path through the bin index. Harmonics keep it open.
        """
        a = torch.sigmoid(self.evidence(h).squeeze(-1)) * mask                 # [B, T]
        m = torch.arange(1, self.harmonics + 1, device=cum.device,
                         dtype=cum.dtype)                                      # [M]

        # Normalise by the EVIDENCE MASS, not the frame count, so S is the circular mean of
        # the evidence: |S| <= 1, near 0 when a is spread over phase and near 1 when it
        # concentrates. Dividing by T instead was MEASURED to break the mechanism outright:
        # a 45 s window spans ~15.8 cycles, so every case -- constant a included -- lands at
        # ~1e-2, features come out at 8e-3, and the head emits logits differing by 1.7e-5.
        # The softmax of that is uniform to six decimals, which is exactly the Hq = 1.000000
        # the first smoke run reported at every epoch. Under this normalisation the same
        # three cases separate 0.0044 (constant) / 0.0069 (random head) / 0.9744 (peaked at
        # the true wraps) -- a 221x range instead of none.
        ang = m[None, :, None] * cum[:, None, :]                               # [B, M, T]
        evidence_mass = a.sum(1).clamp(min=1e-6)[:, None]
        s_cos = (a[:, None, :] * torch.cos(ang)).sum(-1) / evidence_mass       # [B, M]
        s_sin = (a[:, None, :] * torch.sin(ang)).sum(-1) / evidence_mass       # [B, M]

        p = m[None, :, None] * c[:, None, :]                                   # [B, M, C]
        cos_p, sin_p = torch.cos(p), torch.sin(p)
        re = s_cos[..., None] * cos_p + s_sin[..., None] * sin_p                # [B, M, C]
        im = s_sin[..., None] * cos_p - s_cos[..., None] * sin_p
        return torch.cat([re, im], dim=1).permute(0, 2, 1)                     # [B, C, 2M]

    def candidate_logits(self, h, cum, c, ok, mask):
        """[B, C] unnormalised log q(k | x); invalid candidates are -inf, not merely small.

        The descriptors are STANDARDISED across the candidate axis before the head, over the
        valid candidates only. This is what makes the mechanism able to start. Only relative
        scores matter to a softmax, but the absolute scale of the descriptors is set by how
        peaked the evidence already is -- and at init it is not peaked at all: a near-constant
        a has structurally ZERO phase content over a window spanning ~15.8 cycles, so the
        across-candidate sd is 0.0049 and the head emits logits differing by 0.0026. MEASURED
        consequence: Hq = 1.000000 at every epoch of the first smoke run, with agree 0% --
        the posterior could not escape its own initialisation, and the evidence head's only
        gradient path ran through those same flat descriptors. Standardising turns the same
        input into a logit spread of 0.5697 and Hq 0.9987, i.e. a live posterior whose initial
        direction is arbitrary but correctable by the pricing gradient. Normalising the
        harmonics by evidence mass (see candidate_features) was necessary but NOT sufficient
        for this; it fixed the ceiling, this fixes the floor.
        """
        feat = self.candidate_features(h, cum, c, mask)
        w = ok.unsqueeze(-1).to(feat.dtype)
        n = w.sum(1, keepdim=True).clamp(min=1.0)
        mean = (feat * w).sum(1, keepdim=True) / n
        var = (((feat - mean) ** 2) * w).sum(1, keepdim=True) / n
        # eps INSIDE the sqrt, as BatchNorm/LayerNorm do. Clamping the denominator after
        # the root instead (min=1e-8) permits 1e8 amplification, and in the small-variance
        # regime this standardisation exists for it duly produced one -- logits so large
        # that log_softmax returned a one-hot posterior. MEASURED: Hq 0.0000 with agree
        # 100.0% at every step, the exact mirror of the Hq 1.0000 it was added to fix.
        # 1e-4 bounds amplification at 100x and decays to ~0 when the variance genuinely
        # vanishes, instead of exploding.
        feat = (feat - mean) / (var + 1e-4).sqrt()

        logits = self.k_head(feat).squeeze(-1)
        return logits.masked_fill(~ok, float("-inf"))

    def residual(self, residual_raw, cum, mask):
        """[B] a sub-bin refinement of the anchor, BOUNDED to half a bin of phase.

        The bound is what keeps the enumeration meaningful: an unbounded residual can shift
        the trajectory by whole bins and undo the categorical, putting the anchor back on
        the flat landscape it was introduced to escape. Bounded, it recovers the 40 ms
        quantization error instead -- worth about 3 points of the achievable hit rate.

        Half a bin in PHASE is the per-frame advance times half the stride, so the bound is
        data-dependent (a slow bar has wider bins in phase) and needs no bar length.
        """
        inc = (cum[:, 1:] - cum[:, :-1]) * mask[:, 1:]
        rate = inc.sum(1) / mask[:, 1:].sum(1).clamp(min=1.0)                  # [B]
        half_bin = rate * self.stride / 2.0
        scalar = (residual_raw * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return half_bin * torch.tanh(scalar)

    # ------------------------------------------------------------------ the objective

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0):
        """One ELBO evaluation with k marginalised exactly. Same signature as the base.

        The trajectory KL is computed ONCE, on cum. Every candidate's mean path differs
        from cum by a CONSTANT, and the KL depends on mu only through its increments, so it
        is identical for every k and factors out of the marginalisation entirely (measured:
        max |dKL| = 0.000e+00 under a candidate shift, 1.1e-13 at pi). Only the anchor's own
        term -- log C_i - H[q] against a Uniform(C_i) prior -- is new.

        Returns the base keys plus logq / R / n_i for telemetry. NOTE ``mu`` is the SELECTED
        candidate's mean path, not cum: run.py feeds out["mu"] to trajectory_health, and
        reporting advance or phase_err for a trajectory that is never deployed would make
        both numbers meaningless.
        """
        cum, kappa, residual_raw = self.encoder(h, mask)
        kl_traj = self.kl_to_physical_prior(cum, kappa, mask)

        c, ok = candidate_anchors(cum, mask, self.stride)                      # [B, C]
        logq = torch.log_softmax(self.candidate_logits(h, cum, c, ok, mask), dim=-1)
        q = logq.exp()

        base = cum + self.residual(residual_raw, cum, mask)[:, None]           # [B, T]
        mu_k = base[:, None, :] - c[..., None]                                 # [B, C, T]

        weight = torch.where(y > 0, torch.as_tensor(pos_weight, device=y.device,
                                                    dtype=torch.float32),
                             torch.ones((), device=y.device, dtype=torch.float32)) * mask
        target = y.float()[:, None, :].expand_as(mu_k)

        # The von Mises noise is drawn ONCE and shared across candidates: the candidates
        # differ by a deterministic constant, so common random numbers both halve the cost
        # and remove sampling noise from the DIFFERENCES between R_k -- which is the only
        # thing the pricing gradient reads.
        reward = 0.0
        for _ in range(samples):
            phi = mu_k + sample_vonmises(kappa)[:, None, :]
            per_frame = nn.functional.binary_cross_entropy_with_logits(
                self.emission_logits(phi), target, reduction="none")
            reward = reward - (per_frame * weight[:, None, :]).sum(-1)         # [B, C]
        reward = reward / samples

        recon = (q * reward.nan_to_num(0.0)).sum(-1)
        n_i = ok.sum(-1).clamp(min=1)
        # log C_i, not log C_max: it is constant in the parameters, so it has no gradient,
        # but the reported ELBO has to be comparable across items and the corpus mixes 30 s
        # excerpts with full-length songs.
        # NEVER let -inf reach the multiply, in the forward OR the backward. An earlier
        # torch.where(q > 0, q * logq, 0) got the forward right and the BACKWARD wrong:
        # autograd differentiates the unselected branch too, d(q logq)/dq = logq = -inf,
        # and where's backward multiplies that by 0 -> nan, which propagates into the
        # logits and the whole trunk. MEASURED: loss finite at 3.2621 while 34 parameter
        # gradients were nan, at step 0. It needs a MIXED-LENGTH batch to appear at all --
        # with every item the same length ~ok is empty and no -inf exists, which is why
        # --limit-per-fold 16 never showed it and the full 1661-song set failed instantly.
        # Zeroing logq where q is zero is exact (x log x -> 0) and keeps both passes finite.
        neg_entropy = (q * logq.masked_fill(~ok, 0.0)).sum(-1)
        kl = kl_traj + torch.log(n_i.to(cum.dtype)) + neg_entropy

        best = reward.masked_fill(~ok, float("-inf")).argmax(-1)
        rows = torch.arange(len(best), device=cum.device)

        return {"elbo": recon - kl, "recon": recon, "kl": kl,
                "mu": mu_k[rows, best], "kappa": kappa,
                "logq": logq, "R": reward, "n_i": n_i, "ok": ok}

    # ------------------------------------------------------------------ deployment

    @torch.no_grad()
    def infer_phase(self, h, mask=None):
        """Deployment: the mean path of the candidate q likes best. Audio only.

        argmax over q, NEVER over R -- R is the reconstruction, which reads y. And the
        argmax must respect the candidate mask: a pooled deployed statistic that ignores
        pad frames gets outvoted by untrained pad responses on every short-song window,
        which is 33% of every gtzan item (found by the anchor_k pre-launch review).
        Masking happens inside candidate_logits, so the -inf entries can never win.

        Only the selected path is materialised here; the [B, C, T] tensor is a training-time
        cost, not an inference one.
        """
        assert not self.training, "deployment path must run in eval mode"
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)

        cum, _kappa, residual_raw = self.encoder(h, mask)
        c, ok = candidate_anchors(cum, mask, self.stride)
        best = self.candidate_logits(h, cum, c, ok, mask).argmax(-1)
        rows = torch.arange(len(best), device=cum.device)

        base = cum + self.residual(residual_raw, cum, mask)[:, None]
        return base - c[rows, best][:, None]


# ----------------------------------------------------------------- run.py hooks

# Config keys this variant adds on top of the mainline schema, with their defaults.
DEFAULTS = dict(
    anchor_stride_frames=STRIDE_FRAMES,   # 80 ms at 50 fps; INTEGER frames keeps candidates
                                          # exact. See the module docstring for why the
                                          # width is set by the model's error budget rather
                                          # than by the tolerance alone.
    anchor_harmonics=HARMONICS,           # M = 1 with fixed weights would be the closed-form
                                          # circular mean; M > 1 with a learned head is the
                                          # part that is not already the frontend's.
)


def build_model(cfg, input_dim: int) -> AnchorTimeVAE:
    """The time-anchored model. REQUIRES an elementwise emission -- see the assert."""
    assert cfg.emission in ("triangle", "cosine"), (
        f"anchor_time needs an elementwise emission, got {cfg.emission!r}: the "
        f"reconstruction is evaluated at every candidate, so a transformer emission would "
        f"run B x C sequences per step")
    return AnchorTimeVAE(input_dim, stride=cfg.anchor_stride_frames,
                         harmonics=cfg.anchor_harmonics, emission=cfg.emission,
                         emission_layers=cfg.emission_layers,
                         emission_positional=cfg.emission_positional,
                         kappa_physical=cfg.kappa_physical)


def optimizer(model, cfg):
    """(optimizer, params-to-clip). One Adam group; everything clipped, as base."""
    from . import base
    return base.optimizer(model, cfg)


def objective(out, beta: float, cfg):
    """The base ELBO unchanged: ``kl`` already carries both the trajectory and anchor terms."""
    from . import base
    return base.objective(out, beta, cfg)


def on_epoch(model, cfg, epoch: int) -> None:
    """Same emission-sharpness schedule as the base recipe."""
    from . import base
    base.on_epoch(model, cfg, epoch)


def epoch_note(model, probe) -> str:
    """Hq and agree -- the two numbers that read this mechanism.

    ``Hq`` is H[q] / log C_i, so 1.0 means the evidence is not concentrating at all and the
    marginalisation is averaging over noise. ``agree`` is whether argmax q matches argmax R,
    i.e. whether the head has learned to want what the reconstruction wants.

    The predicted dynamic has a testable ORDER: Hq must fall before advance leaves its
    clamp, because a concentrated anchor is what makes the rate gradient coherent. advance
    moving first would refute that story.

    ``probe`` is run.py's frozen probe batch -- {"h", "mask", "y"}, the same windows every
    epoch, so a change in Hq is a change in the MODEL and not in the input.
    """
    with torch.no_grad():
        out = model(probe["h"], probe["mask"], probe["y"])
    logq, ok = out["logq"], out["ok"]
    q = logq.exp()
    entropy = -(q * logq.masked_fill(~ok, 0.0)).sum(-1)      # see forward: no -inf, ever
    agree = (q.argmax(-1) == out["R"].masked_fill(~ok, float("-inf")).argmax(-1))

    # Average over SCORABLE items only. A fully-masked backstop window keeps exactly one
    # candidate (candidate_anchors' guard), so log C_i = log 1 = 0 and its normalised
    # entropy is 0/0: nan, or ~1e6 under a clamped denominator. Either value from one
    # backstop item destroys the batch mean, and Hq is the number this whole variant is
    # read by. If nothing is scorable, say so rather than printing a fabricated average.
    live = out["n_i"] > 1
    if not bool(live.any()):
        return "  Hq    n/a  agree   n/a"
    normalised = entropy[live] / torch.log(out["n_i"][live].to(entropy.dtype))
    return (f"  Hq {float(normalised.mean()):5.3f}  "
            f"agree {float(agree[live].float().mean()):5.1%}")
