"""The window's rotation becomes a latent, so no term in the ELBO needs a detach.

In ``interval`` the rotation is not a random variable at all: ``offset`` is a
deterministic estimator computed inside q from a DETACHED ramp, and the placement
factor then scores a SECOND detached copy of the ramp. Two detaches mean the quantity
being differentiated is not the gradient of any scalar, and the reason they were needed
is arithmetic: with phi_1 = mu0(t_1) + theta the placement factor's derivative splits as

    d(kappa cos phi_1)/d log k = -kappa sin(phi_1) * (dmu0(t_1)/dlog k + dtheta/dlog k)

and with the first annotation ~42 frames in, mu0(t_1) reaches 5.8 rad at k = 2.65, so a
rotation the model has no reason to get right at a wrong rate injects +-476 nats of
sign-flipping force into the rate channel against the interval ruler's flat +-110.

Here theta is an explicit latent with

    q(theta) = vM(theta; anchor(a, mu0), kappa_theta)      p(theta) = 1 / 2pi

so KL(q || p) = log 2pi - H[vM(kappa_theta)] in closed form, and the placement factor is
the only term that reads theta at all: the interval ratios are DIFFERENCES of phase and
the Jacobian reads dotphi, both of which a constant rotation leaves untouched. The
placement factor therefore owns the rotation by construction, and its expectation under
q is available exactly,

    E_q[kappa cos(mu0_1 + theta)] = kappa * A(kappa_theta) * cos(mu0_1 + anchor),

which is the same scalar the reparameterised sample estimates, with the sampling
variance (+-kappa |mu0_1| per draw in the rate channel) removed rather than averaged.
``place_expect`` chooses between the two estimators; both are the same ELBO.

That A(kappa_theta) is the whole mechanism. kappa_theta is read from the fold's own
resultant, so a rate whose evidence fold does not concentrate gets a nearly uniform
rotation posterior, the placement factor is damped toward its uniform value, and the
leak into the rate is damped with it -- while at a rate whose fold DOES concentrate the
placement factor comes back at full strength and sin(phi_1) is small anyway because the
anchor is then a good estimate. The resultant is also exactly the statistic rate_grid
scores its candidates with, so this term carries the search signal into the ELBO as a
gradient instead of an enumeration.

Labels enter the loss only. The encoder is unchanged and still reads audio alone.
"""
from __future__ import annotations

import math

import torch

from .base import common_kwargs, epoch_note, objective, optimizer  # noqa: F401
from .interval import DEFAULTS as INTERVAL_DEFAULTS
from .interval import on_epoch  # noqa: F401
from ..model import IntervalVAE
from ..observation import (annotation_frames, interp_phase, interval_loglik,
                           smooth_phase)
from ..constants import TWO_PI
from ..nets import Encoder, bounded_kappa, vonmises_entropy
from ..vonmises import log_i0, mean_resultant, sample_vonmises

DEFAULTS = dict(INTERVAL_DEFAULTS, sigma_ceil=0.01, kappa_theta="fold", kappa_theta_scale=1.0,
                place_expect=True, place_index="first")

FOLD_EPS = 1e-12
MASS_EPS = 1e-6
RES_EPS = 1e-6
KAPPA_EPS = 1e-4


def evidence_fold(mu0, a):
    """(rotation estimate, normalised resultant) of the evidence folded on the ramp.

    Every guard is ADDITIVE. ``sqrt(real^2 + imag^2)`` has an unbounded gradient at the
    origin and ``atan2(0, 0)`` is undefined there, and a fold that cancels exactly is
    reachable: it is what NaN'd rate_grid at epoch 8. A clamp would trade that for a
    dead zone of exactly zero gradient, which is the species of bug that froze the tempo
    channel for hours, so the norm gets an additive floor inside the root and only the
    real part of a fold with no mass at all is substituted.
    """
    real = (a * torch.cos(mu0)).sum(1)
    imag = (a * torch.sin(mu0)).sum(1)
    norm2 = real ** 2 + imag ** 2
    anchor = -torch.atan2(imag, torch.where(norm2 > FOLD_EPS, real,
                                            torch.ones_like(real)))
    return anchor, torch.sqrt(norm2 + FOLD_EPS) / (a.sum(1) + MASS_EPS)


def rotation_concentration(resultant, kind: str, scale: float, kappa_place: float):
    """kappa_theta from the fold: how concentrated q(theta) is allowed to be.

    ``fold`` is Banerjee's A^-1(R), the von Mises MLE of the concentration of the very
    directions being averaged -- q(theta) is then no sharper than the evidence that
    produced it, and E_q[cos] = A(kappa_theta) ~= R makes the placement reward literally
    kappa_place * R * cos(phi_1), i.e. rate_grid's search score with an alignment term.
    ``place`` is the cruder kappa_place * R, which keeps the placement factor near full
    strength for any fold that concentrates at all.

    The cap is kappa_place because the exact optimal q(theta) for this likelihood is
    vM(-mu0_1, kappa_place cos phi_1): nothing sharper is ever wanted, and paying for it
    is strictly negative ELBO. It is applied as cap * tanh(k / cap), the identity well
    below the cap, never a hard min with its dead half-line.
    """
    if kind == "place":
        return scale * kappa_place * resultant + KAPPA_EPS
    raw = resultant * (2.0 - resultant ** 2) / (1.0 - resultant ** 2 + RES_EPS)
    cap = scale * kappa_place
    return cap * torch.tanh(raw / cap) + KAPPA_EPS


def fold_centroid(mu0, a, anchor):
    """The unwrapped ramp phase the rotation estimate is actually anchored to [B].

    d(anchor)/d(log k) is exactly minus this: the fold's weight on frame t is
    a_t cos(mu0_t + anchor) once the wrap is aligned, so rotating the ramp moves the
    estimate by the aligned-weighted mean of mu0. It is the far end of the lever the
    placement factor pushes on.
    """
    weight = a * torch.cos(mu0 + anchor[:, None])
    return (weight * mu0).sum(1) / (weight.sum(1) + MASS_EPS)


def placement_select(phi_place, ann_f, ann_valid, mu0, a, anchor, kind: str):
    """[B, N] one-hot: which annotation carries the placement coordinate.

    The change of variables (t_1..t_N) -> (phi_j, log r_1..log r_{N-1}) is triangular
    with |det| = prod 1/(2 pi r_i) for EVERY j, so which annotation carries the vM is a
    free choice of coordinate, not a modelling assumption -- and it is the only lever on
    the residual leak, whose size is kappa A(kappa_theta) sin(phi_j) (mu0_j - centroid).
    ``fold`` spends it: the coordinate goes to the annotation nearest the fold's own
    centroid, which is where the lever arm is shortest.
    """
    first = ann_valid.cumsum(1).eq(1.0).to(phi_place.dtype) * ann_valid
    if kind != "fold":
        return first
    with torch.no_grad():
        at = interp_phase(mu0, ann_f)
        far = (at - fold_centroid(mu0, a, anchor)[:, None]).abs()
        pick = torch.where(ann_valid > 0, far, torch.full_like(far, float("inf")))
        one = torch.zeros_like(first)
        one.scatter_(1, pick.argmin(1, keepdim=True), 1.0)
    return one


def placement_expectation(phi_place, ann_f, ann_valid, kappa_place, a_theta,
                          phase_half: int = 0, select=None):
    """E_q(theta)[log vM(phi(t_j); 0, kappa_place)] [B].

    ``a_theta`` is A(kappa_theta) when the expectation is taken in closed form and 1
    when ``phi_place`` already carries a reparameterised draw of theta; the normaliser
    is constant in theta either way, so only the cosine is scaled.
    """
    kappa = torch.as_tensor(kappa_place, device=phi_place.device, dtype=phi_place.dtype)
    at = interp_phase(smooth_phase(phi_place, phase_half), ann_f)
    if select is None:
        select = ann_valid.cumsum(1).eq(1.0).to(phi_place.dtype) * ann_valid
    scaled = kappa * a_theta[:, None] * torch.cos(at)
    return ((scaled - math.log(TWO_PI) - log_i0(kappa)) * select).sum(1)


def rotation_loglik(phi, phi_place, a_theta, ann_f, ann_valid, kappa_place: float,
                    b_ratio: float, phase_half: int = 0, kind: str = "laplace",
                    select=None):
    """log p(annotation times | phi) [B] with the placement factor's theta integrated.

    The interval ratios and the Jacobian are computed on the sampled path exactly as in
    ``interval``; they are invariant to theta, which is why the rotation can be given
    its own factor without touching them.
    """
    em = interval_loglik(phi, ann_f, ann_valid, kappa_place, b_ratio, phase_half, kind)
    place = placement_expectation(phi_place, ann_f, ann_valid, kappa_place, a_theta,
                                  phase_half, select)
    return {"loglik": em["loglik"] - em["place"] + place, "place": place,
            "interval": em["interval"]}


class RotationEncoder(Encoder):
    """``Encoder.heads`` with the fold left un-detached and exported as q(theta)."""

    def heads(self, trunk, mask=None, h=None):
        channels = self.output_channels(trunk)
        kappa = bounded_kappa(
            torch.exp(channels["log_phi_kappa"] + self.log_phi_kappa_bias) + 1e-3)

        w = torch.ones(trunk.shape[:2], device=trunk.device, dtype=trunk.dtype) \
            if mask is None else mask
        log_dotphi, seg = self._bar_seg(channels["log_dotphi"], w)
        tempo_log_mu = log_dotphi
        log_dotphi, tempo_entropy, _tempo_sigma = self._sample_learned_sigma(
            channels["log_sigma_dotphi"], log_dotphi, seg, w)

        _dotphi, mu0 = self._ramp(log_dotphi)
        a = torch.sigmoid(channels["downbeat_logit"]) * w
        anchor, resultant = evidence_fold(mu0, a)

        return {"phase": {"mu": mu0 + anchor[:, None], "kappa": kappa},
            "tempo": {"log_mu": tempo_log_mu, "sigma": _tempo_sigma, "seg": seg,
                      "log_prior": self._tempo_log_prior(log_dotphi, seg, w),
                      "entropy": tempo_entropy},
            "anchor": anchor, "resultant": resultant,
            "evidence": a}


class RotationIntervalVAE(IntervalVAE):
    """IntervalVAE with the rotation as a latent: q(theta) vs a uniform p(theta)."""

    wants_raw = True

    def __init__(self, input_dim: int, d_model: int = 128, kappa_theta: str = "fold",
                 kappa_theta_scale: float = 1.0, place_expect: bool = True,
                 place_index: str = "first", sigma_ceil: float = 0.0, **kw):
        super().__init__(input_dim, d_model=d_model, **kw)
        self.encoder = RotationEncoder(input_dim, d_model,
                                       kappa_physical=self.walk.kappa_physical)
        self.kappa_theta_kind = kappa_theta
        self.kappa_theta_scale = float(kappa_theta_scale)
        self.place_expect = bool(place_expect)
        self.place_index = place_index

    def rotation_posterior(self, resultant):
        """(kappa_theta, A(kappa_theta), KL(q(theta) || uniform)) [B] each."""
        kappa_theta = rotation_concentration(resultant, self.kappa_theta_kind,
                                             self.kappa_theta_scale, self.kappa_place)
        return kappa_theta, mean_resultant(kappa_theta), \
            math.log(TWO_PI) - vonmises_entropy(kappa_theta)

    def forward(self, h, mask, y, samples: int = 1, pos_weight: float = 1.0, raw=None):
        assert raw is not None, "the interval emission needs the batch's downbeat_times"
        post, _ = self.encoder(h, mask)
        mu, kappa, aux = post["phase"]["mu"], post["phase"]["kappa"], post
        kappa_theta, a_theta, kl_theta = self.rotation_posterior(aux["resultant"])
        kl = (self.kl_jitter(mu, kappa, mask) + kl_theta
              - aux["tempo"]["log_prior"] - aux["tempo"]["entropy"])
        ann_f, ann_valid = annotation_frames(raw, mu.device)

        mu0 = mu - aux["anchor"][:, None]
        recon = 0.0
        for _ in range(samples):
            jitter = sample_vonmises(kappa)
            theta = sample_vonmises(kappa_theta)[:, None]
            phi = mu + theta + jitter
            phi_place = mu + jitter if self.place_expect else phi
            weight = a_theta if self.place_expect else torch.ones_like(a_theta)
            select = placement_select(phi_place, ann_f, ann_valid, mu0, aux["evidence"],
                                      aux["anchor"], self.place_index)
            recon = recon + rotation_loglik(phi, phi_place, weight, ann_f, ann_valid,
                                            self.kappa_place, self.b_ratio,
                                            self.phase_half, self.interval_kind,
                                            select)["loglik"]
        recon = recon / samples

        return {"elbo": recon - kl, "recon": recon, "kl": kl, "phi": mu, "kappa": kappa,
                "tempo_prior": aux["tempo"]["log_prior"],
                "tempo_entropy": aux["tempo"]["entropy"],
                "resultant": aux["resultant"], "kappa_theta": kappa_theta,
                "kl_theta": kl_theta}


def build_model(cfg, input_dim: int) -> RotationIntervalVAE:
    return RotationIntervalVAE(input_dim, b_ratio=cfg.b_ratio,
                               kappa_place=cfg.kappa_place, phase_half=cfg.phase_half,
                               interval_kind=cfg.interval_kind,
                               kappa_theta=cfg.kappa_theta,
                               kappa_theta_scale=cfg.kappa_theta_scale,
                               place_expect=cfg.place_expect,
                               place_index=cfg.place_index,
                               **common_kwargs(cfg))
