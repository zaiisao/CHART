"""The Stage-P implementation *under test*, behind one adapter -- and the oracle.

Same contract as Stage 0's ``tests/subject.py``: every property takes an implementation as
a PARAMETER, so the identical test body runs against

  * ``oracle()``            -- known-correct, so a failure means the TEST is wrong,
  * ``mutants_p.MUTANTS``   -- known-broken, so a pass means the TEST is too weak,
  * a real Stage-P package once one exists -- the actual verdict.

Model (``docs/SPEC_phase.md`` SS4)
---------------------------------
    theta : emission  p_theta(y_i | r_i)   EXACTLY two scalars {alpha, beta} (SS4.3)
    psi   : prior     p_psi(r | h)         sees h and the given beat grid  <- deployable
    phi   : encoder   q_phi(r | h, y)      one scalar {c}, training only   (SS4.5)

Why psi is a PER-BEAT potential
-------------------------------
SS4.4 requires that at P2 ``psi`` supply a per-beat audio potential ``psi(r_i | h_i)``
consumed by the recursion, and the oracle takes the same shape at P1: the crop-level logit
for offset ``r`` is the per-beat potential SUMMED along that offset's comb,

    logits_psi(h)[r] = sum_{i : i == r (mod m)} g_psi(h_i).

Two things fall out, and both are load-bearing for this suite.

*The SS4.6 ``eps = 0`` reduction becomes exact in psi as well as in the likelihood.* With a
deterministic advance the chain admits exactly ``m`` paths, and the path through pointer
state 0 at the comb of offset ``r`` accumulates precisely the sum above. P2 and P1 then
share one parameter vector, so "P2 reduces to P1" is a statement about the whole model
rather than about the emission alone.

*The read-out is shift-EQUIVARIANT by construction*, which is what SS8.4 demands. Note what
this forbids: a per-offset bias term ``b[r]``. Stage P's latent is uniform over ``0..m-1``
by construction (SS1 -- a crop may begin anywhere in the bar), so a learned per-offset bias
is provably modelling nothing, and it breaks equivariance. The oracle has no such term and
``psi_offset_bias`` is a mutant that must be killed.

Note that SS4.4 also asserts "there is no pooled crop-level summary anywhere on the
generative path". That is not achievable at P1, where ``logits_psi(h)`` is necessarily a
crop-level object; the reducer dies structurally only at P2. The comb-sum above is the
weakest crop-level summary that can represent phase at all -- it is a sum over positions,
not a pool over them.

What is NOT here: a P2 encoder
------------------------------
SS4.5 defines ``q_phi`` only for the static P1 latent, and no P2 encoder is defined
anywhere in ``SPEC_phase.md``. The P2 *loss* is therefore unwritable from the spec, and
inventing one would put a design decision inside the acceptance suite. P2 appears here only
as exact INFERENCE -- forward algorithm, forward-backward, Viterbi, and the ``eps = 0``
reduction -- all of which SS4.6/SS9 specify without reference to a variational family.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

import reference_p as R

# SS6.2/P6: keys a deployable call may never see. r_true is scoring-only, y is the
# annotation, and pointer is the bench's ground-truth trajectory.
ANNOTATION_KEYS = ("y", "r_true", "pointer")


def deployable_view(crop: dict) -> dict:
    """Return the crop with every annotation key REMOVED.

    C2/P6 say the deployable path reads ``h`` only (plus the given beat grid). A signature
    of ``predict(crop)`` does not enforce that when ``crop`` also carries ``y``, so the
    properties hand ``predict`` this stripped view instead. An implementation that reaches
    for annotations then raises KeyError rather than quietly scoring better -- structural
    enforcement, not a promise. The stashing leak is a separate check.

    Args:
        crop: A crop dict from ``bench_p``.

    Returns:
        A shallow copy without ``y``, ``r_true`` or ``pointer``.
    """
    return {k: v for k, v in crop.items() if k not in ANNOTATION_KEYS}


@dataclass
class StageP:
    """One Stage-P implementation. All log-probabilities natural log, all tensors float64."""

    m: int = R.STAGE_P_M
    capacity: str = "full"          # 'exact' | 'full'  -- a TESTING affordance (SS4.5)
    name: str = "oracle"
    # theta -- SS4.3 pins this to exactly two scalars
    alpha: torch.Tensor = field(default=None)
    beta: torch.Tensor = field(default=None)
    # psi -- per-beat potential weights (deployable), plus P2 slip
    w_beat: torch.Tensor = field(default=None)
    eps_logits: torch.Tensor = field(default=None)
    # phi -- SS4.5's single scalar
    c: torch.Tensor = field(default=None)
    # hooks the mutants replace: a mutant stays a small reversible edit, not a forked class
    hooks: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.alpha is None:
            self.alpha = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        if self.beta is None:
            self.beta = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)
        if self.w_beat is None:
            # NOT zero-initialised. SS4.7 records that a brand-new implementation produced
            # exactly-zero gradients on 2026-08-03 because zero-initialised output layers
            # make d(loss)/d(body) vanish at step 0. Small non-zero init, deterministically.
            self.w_beat = torch.tensor([0.3, -0.1], dtype=torch.float64, requires_grad=True)
        if self.eps_logits is None:
            # softmax([0, l_hold, l_skip]) -> (advance, hold, skip); init to small slip
            self.eps_logits = torch.tensor([-3.0, -3.5], dtype=torch.float64,
                                           requires_grad=True)
        if self.c is None:
            self.c = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

    # -- introspection ---------------------------------------------------------------
    def param_groups(self, stage: str = "P1") -> dict:
        """Return the SS4.7 theta/psi/phi split as ``{group: {name: tensor}}``.

        Only theta's parameterisation is fixed by the spec (SS4.3: exactly ``alpha`` and
        ``beta``); psi's and phi's internals are implementation choices, so properties must
        reach them by GROUP and by ``p.shape``, never by field name.

        Args:
            stage: ``"P1"`` or ``"P2"``. The slip parameters belong to psi only at P2 --
                at P1 nothing consumes them, and a gradient audit that demanded a gradient
                for an unused parameter would fail correct code.

        Returns:
            The nested parameter mapping.
        """
        psi = {"w_beat": self.w_beat}
        if stage == "P2":
            psi["eps_logits"] = self.eps_logits
        return {"theta": {"alpha": self.alpha, "beta": self.beta},
                "psi": psi,
                "phi": {"c": self.c}}

    def named_params(self, stage: str = "P1") -> dict:
        """Return every parameter of a stage, flat -- the gradient-audit surface.

        Args:
            stage: ``"P1"`` or ``"P2"``.

        Returns:
            A flat ``{name: tensor}`` mapping.
        """
        out = {}
        for group in self.param_groups(stage).values():
            out.update(group)
        return out

    def trainable_params(self, stage: str = "P1") -> list:
        """Return the optimiser set with each TIED parameter included exactly once.

        Args:
            stage: ``"P1"`` or ``"P2"``.

        Returns:
            A list of tensors.
        """
        seen, ps = [], []
        for p in self.named_params(stage).values():
            if p.requires_grad and id(p) not in seen:
                seen.append(id(p))
                ps.append(p)
        return ps

    def _hook(self, name, default):
        return self.hooks.get(name, default)

    # -- psi: the deployable path (h and the GIVEN beat grid; never y) ----------------
    def beat_features(self, obs) -> torch.Tensor:
        """Return ``[n, 2]`` activation features sampled at the given beat frames.

        The beat grid is given at Stage P (SS2), so sampling ``h`` at beat positions reads
        no annotation. Channel order follows the bench: downbeat channel, then beat channel.

        Args:
            obs: A crop dict; only ``h`` and ``beat_frames`` are touched.

        Returns:
            A float64 tensor of per-beat features.
        """
        fn = self._hook("beat_features", _beat_features)
        return fn(self, obs)

    def beat_potential(self, obs) -> torch.Tensor:
        """Return ``[n]`` the per-beat downbeat log-potential ``g_psi(h_i)`` (SS4.4).

        Args:
            obs: A crop dict.

        Returns:
            A float64 tensor.
        """
        fn = self._hook("beat_potential", _beat_potential)
        return fn(self, obs)

    def prior_logits(self, obs) -> torch.Tensor:
        """Return ``[m]`` unnormalised offset logits: the potential summed along each comb.

        Args:
            obs: A crop dict.

        Returns:
            A float64 tensor indexed by offset.
        """
        fn = self._hook("prior_logits", _prior_logits)
        return fn(self, obs)

    def prior_logp(self, obs) -> torch.Tensor:
        """Return ``[m]`` log p_psi(r | h).

        Args:
            obs: A crop dict.

        Returns:
            A normalised float64 log-probability tensor.
        """
        logits = self.prior_logits(obs)
        return logits - torch.logsumexp(logits, -1)

    def predict(self, obs) -> torch.Tensor:
        """Return the deployed predictor's ``[m]`` log-distribution over the offset.

        By construction this may not see ``y``; properties call it on
        :func:`deployable_view` so that reaching for one raises.

        Args:
            obs: A crop dict, normally stripped of annotations.

        Returns:
            A float64 log-probability tensor.
        """
        fn = self._hook("predict", lambda s, o: s.prior_logp(o))
        return fn(self, obs)

    def predict_offset(self, obs) -> int:
        """Return the deployed offset ``r_hat = argmax_r p_psi(r | h)`` (SS7, P1).

        Args:
            obs: A crop dict.

        Returns:
            The predicted offset as an int.
        """
        fn = self._hook("predict_offset", lambda s, o: int(s.predict(o).argmax()))
        return fn(self, obs)

    def predict_placement(self, obs, n: int) -> np.ndarray:
        """Return the ``[n]`` predicted downbeat placement -- SS7's deployable read-out.

        Args:
            obs: A crop dict.
            n: Number of beats in the crop.

        Returns:
            A uint8 array with ones where a downbeat is predicted.
        """
        fn = self._hook("predict_placement", _predict_placement)
        return fn(self, obs, n)

    # -- theta: the emission ----------------------------------------------------------
    def emission_logp_all(self, y) -> torch.Tensor:
        """Return ``[m]`` of log p_theta(y | r) -- r is the LATENT, not marginalised.

        Args:
            y: Per-beat downbeat indicators.

        Returns:
            A float64 tensor indexed by offset.
        """
        fn = self._hook("emission_logp_all", _emission_logp_all)
        return fn(self, y)

    def marginalised_emission(self, y) -> torch.Tensor:
        """Return Stage 0's scalar emission: ``log[(1/m) sum_r p_theta(y|r)]``.

        The SS4.6 reduction handle. Present so the reduction can be asserted against the
        shipped Stage-0 code without the property reaching into internals.

        Args:
            y: Per-beat downbeat indicators.

        Returns:
            A float64 scalar tensor.
        """
        return torch.logsumexp(self.emission_logp_all(y), -1) - math.log(self.m)

    # -- phi and P1 inference ---------------------------------------------------------
    def q_logp(self, obs, y) -> torch.Tensor:
        """Return ``[m]`` log q_phi(r | h, y) = softmax(prior logits + c * emission).

        Args:
            obs: A crop dict.
            y: Per-beat downbeat indicators.

        Returns:
            A normalised float64 log-probability tensor.
        """
        fn = self._hook("q_logp", _q_logp)
        return fn(self, obs, y)

    def exact_posterior(self, obs, y) -> torch.Tensor:
        """Return ``[m]`` log p(r | y, h) by Bayes.

        Args:
            obs: A crop dict.
            y: Per-beat downbeat indicators.

        Returns:
            A normalised float64 log-probability tensor.
        """
        fn = self._hook("exact_posterior", _exact_posterior)
        return fn(self, obs, y)

    def log_evidence(self, obs, y) -> torch.Tensor:
        """Return the scalar log p(y | h) by exact enumeration over the m offsets.

        Args:
            obs: A crop dict.
            y: Per-beat downbeat indicators.

        Returns:
            A float64 scalar tensor.
        """
        return torch.logsumexp(self.emission_logp_all(y) + self.prior_logp(obs), dim=-1)

    def elbo(self, obs, y) -> torch.Tensor:
        """Return the scalar SS4.6 ELBO: ``E_q[log p(y|r)] - KL(q || p_psi)``.

        Args:
            obs: A crop dict.
            y: Per-beat downbeat indicators.

        Returns:
            A float64 scalar tensor.
        """
        fn = self._hook("elbo", _elbo)
        return fn(self, obs, y)

    # -- P2: exact inference only (no encoder exists in the spec) ---------------------
    def slip(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(eps_hold, eps_skip)`` as a softmax over (advance, hold, skip).

        Parameterised so the three branches provably sum to one whatever the logits do --
        SS4.1 leaves the relation between ``eps`` and ``{eps_hold, eps_skip}`` unstated.

        Returns:
            A pair of float64 scalar tensors.
        """
        full = torch.cat([torch.zeros(1, dtype=torch.float64), self.eps_logits])
        p = torch.softmax(full, 0)
        return p[1], p[2]

    def log_transition(self) -> torch.Tensor:
        """Return the ``[m, m]`` log transition over pointer states.

        Returns:
            A float64 tensor, row-stochastic in probability space.
        """
        fn = self._hook("log_transition", _log_transition)
        return fn(self)

    def node_potentials(self, obs) -> torch.Tensor:
        """Return ``[n, m]`` per-beat audio potentials: ``g_psi(h_i)`` on state 0 (SS4.4).

        Args:
            obs: A crop dict.

        Returns:
            A float64 tensor.
        """
        fn = self._hook("node_potentials", _node_potentials)
        return fn(self, obs)

    def _chain_logz(self, node) -> torch.Tensor:
        """Return the scalar log partition of the pointer chain given node potentials."""
        n, m = node.shape
        logT = self.log_transition()
        a = -math.log(m) + node[0]
        for i in range(1, n):
            a = torch.logsumexp(a.unsqueeze(1) + logT, dim=0) + node[i]
        return torch.logsumexp(a, -1)

    def p2_log_evidence(self, obs, y) -> torch.Tensor:
        """Return the scalar P2 log p(y | h) = log Z(potentials + emission) - log Z(potentials).

        The prior chain carries audio potentials, so it is unnormalised over paths and its
        own partition function must be divided out. Omitting it is a real and silent bug --
        it is the mutant ``p2_forgets_prior_partition``.

        Args:
            obs: A crop dict.
            y: Per-beat downbeat indicators.

        Returns:
            A float64 scalar tensor.
        """
        fn = self._hook("p2_log_evidence", _p2_log_evidence)
        return fn(self, obs, y)

    def p2_emission(self, y) -> torch.Tensor:
        """Return ``[n, m]`` of log p_theta(y_i | s_i) over pointer states.

        Args:
            y: Per-beat downbeat indicators.

        Returns:
            A float64 tensor.
        """
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
        lsig = torch.nn.functional.logsigmoid
        on = y_t * lsig(self.alpha) + (1 - y_t) * lsig(-self.alpha)
        off = y_t * lsig(self.beta) + (1 - y_t) * lsig(-self.beta)
        E = off.unsqueeze(1).repeat(1, self.m)
        E[:, 0] = on
        return E

    # -- fitting (P1) -----------------------------------------------------------------
    def fit(self, crops, steps: int = 300, lr: float = 0.1, seed: int = 0) -> "StageP":
        """Maximise the mean P1 ELBO. Exact enumeration, so no sampling and no seed noise.

        Args:
            crops: Crop dicts carrying ``h``, ``beat_frames`` and ``y``.
            steps: Number of Adam steps.
            lr: Learning rate.
            seed: Present for interface compatibility; nothing here samples.

        Returns:
            ``self``, fitted in place.
        """
        torch.manual_seed(seed)
        ps = self.trainable_params("P1")
        if not ps:
            return self
        opt = torch.optim.Adam(ps, lr=lr)
        cache = [(deployable_view(c), c["y"]) for c in crops]
        for _ in range(steps):
            opt.zero_grad()
            loss = -torch.stack([self.elbo(o, y) for o, y in cache]).mean()
            loss.backward()
            opt.step()
        return self


# ------------------------------------------------------------------------------------
# the correct implementation of each hook
# ------------------------------------------------------------------------------------
def _beat_features(s: StageP, obs) -> torch.Tensor:
    h = torch.as_tensor(np.asarray(obs["h"], dtype=np.float64), dtype=torch.float64)
    bf = torch.as_tensor(np.asarray(obs["beat_frames"], dtype=np.int64))
    bf = bf.clamp(0, h.shape[0] - 1)
    sel = h[bf]                                   # [n, 2]
    return torch.stack([sel[:, 1], sel[:, 0]], dim=1)


def _beat_potential(s: StageP, obs) -> torch.Tensor:
    return _beat_features(s, obs) @ s.w_beat


def _prior_logits(s: StageP, obs) -> torch.Tensor:
    g = s.beat_potential(obs)
    n = g.shape[0]
    idx = torch.arange(n)
    return torch.stack([g[((idx - r) % s.m) == 0].sum() for r in range(s.m)])


def _emission_logp_all(s: StageP, y) -> torch.Tensor:
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
    n = len(y_t)
    lsig = torch.nn.functional.logsigmoid
    on = y_t * lsig(s.alpha) + (1 - y_t) * lsig(-s.alpha)
    off = y_t * lsig(s.beta) + (1 - y_t) * lsig(-s.beta)
    idx = torch.arange(n, dtype=torch.float64)
    out = []
    for r in range(s.m):
        mask = (((idx - r) % s.m) == 0).to(torch.float64)
        out.append((mask * on + (1 - mask) * off).sum())
    return torch.stack(out)


def _q_logp(s: StageP, obs, y) -> torch.Tensor:
    prior = s.prior_logp(obs)
    if s.capacity == "exact":
        logits = prior + s.emission_logp_all(y)
    elif s.capacity == "full":
        logits = prior + s.c * s.emission_logp_all(y)
    else:
        raise ValueError(f"unknown capacity {s.capacity!r}")
    return logits - torch.logsumexp(logits, -1)


def _exact_posterior(s: StageP, obs, y) -> torch.Tensor:
    lp = s.emission_logp_all(y) + s.prior_logp(obs)
    return lp - torch.logsumexp(lp, -1)


def _elbo(s: StageP, obs, y) -> torch.Tensor:
    q_logp = s.q_logp(obs, y)
    q = q_logp.exp()
    recon = (q * s.emission_logp_all(y)).sum()
    kl = (q * (q_logp - s.prior_logp(obs))).sum()     # KL(q || p_psi), the ELBO's direction
    return recon - kl


def _predict_placement(s: StageP, obs, n: int) -> np.ndarray:
    r = s.predict_offset(obs)
    return R.downbeats_from_offset(int(n), int(r), s.m)


def _log_transition(s: StageP) -> torch.Tensor:
    eh, es = s.slip()
    adv = 1.0 - eh - es
    T = torch.zeros((s.m, s.m), dtype=torch.float64)
    for a in range(s.m):
        T[a, (a + 1) % s.m] = T[a, (a + 1) % s.m] + adv
        T[a, a] = T[a, a] + eh
        T[a, (a + 2) % s.m] = T[a, (a + 2) % s.m] + es
    return torch.log(T)


def _node_potentials(s: StageP, obs) -> torch.Tensor:
    g = s.beat_potential(obs)
    node = torch.zeros((g.shape[0], s.m), dtype=torch.float64)
    node[:, 0] = g
    return node


def _p2_log_evidence(s: StageP, obs, y) -> torch.Tensor:
    node = s.node_potentials(obs)
    return s._chain_logz(node + s.p2_emission(y)) - s._chain_logz(node)


# ------------------------------------------------------------------------------------
# factories
# ------------------------------------------------------------------------------------
def oracle(m: int = R.STAGE_P_M, capacity: str = "full", **kw) -> StageP:
    """Return the known-correct subject. Every property MUST pass against this.

    Args:
        m: Beats per bar (SS2.1: declared, parameterised, never inlined).
        capacity: Encoder capacity; a testing affordance only.
        **kw: Forwarded to :class:`StageP`.

    Returns:
        A fresh oracle.
    """
    return StageP(m=int(m), capacity=capacity, name="oracle", **kw)


def trained_oracle(crops, m: int = R.STAGE_P_M, steps: int = 300, lr: float = 0.1) -> StageP:
    """Return an oracle fitted on the given crops.

    Args:
        crops: Training crops.
        m: Beats per bar.
        steps: Adam steps.
        lr: Learning rate.

    Returns:
        The fitted oracle.
    """
    return oracle(m=m).fit(crops, steps=steps, lr=lr)
