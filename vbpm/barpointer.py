"""Increment B: the bar-pointer HMM over a given beat grid — the EXACT-INFERENCE reference.

Stage 0 (``vbpm/stage0.py``) carries one latent ``m`` per crop and marginalises the bar
offset ``r`` away inside the emission. This module promotes ``r`` to a latent and gives it
a time index: ``z_i = (m_i, r_i)`` for every beat ``i`` of the crop, with ``r_i = 0`` meaning
"beat i is a downbeat". ``r`` is the beat-grid discretisation of the spec's bar phase
``phi`` (docs/SPEC.md section 11), and the wrap ``r: m-1 -> 0`` is the discrete image of
``phi`` crossing ``2*pi``.

Three consequences, all deliberate (see docs/PHASE_PLAN.md):

    * the REDUCER IS GONE. Stage 0 needs ``s(h)`` only because its latent has no time
      index and its prior therefore needs a fixed-size input. Here ``h`` enters the
      recursion once per beat, through the transition and through a per-beat state
      potential, so there is no crop-level quantity left to summarise.
    * INFERENCE IS EXACT. Forward-backward over 9 states is closed form, so q IS the
      posterior, the KL slack is zero and the ELBO equals ``log p(y|h)``. This module does
      NO variational work and that is its purpose: it is the ceiling against which a
      variational Stage 1 is measured. There is no phi parameter set here, only theta and
      psi.
    * THE EMISSION IS UNCHANGED. ``p_theta(y_i|z_i)`` is the same two scalars {alpha, beta}
      as Stage 0, latent-only. ``h`` conditions the dynamics and never the emission, so the
      spec's conditional independence ``y _||_ h | z`` (section 4.3) survives verbatim.
      Holding the emission fixed across increments is what makes the Stage-0/B/Stage-1 gap
      attributable to inference rather than to the model class.

Everything is log-space float64 and deterministic.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .stage0 import DEFAULT_VALUES

NEG_INF = -1e30      # additive "impossible": finite, so 0 * NEG_INF never produces a NaN


def states(values=DEFAULT_VALUES) -> list:
    """The legal ``(m, r)`` pairs, ordered — 9 of them for ``values = (2, 3, 4)``.

    ``m`` is a COUNT of beats per bar and ``r`` an index within the bar; the two are never
    interchanged (docs/SPEC.md C1). Position in this list is the state index, which is an
    index into THIS list and nothing else.

    Args:
        values: the meter vocabulary, as counts.

    Returns:
        A list of ``(m, r)`` tuples with ``0 <= r < m``.
    """
    return [(int(m), r) for m in values for r in range(int(m))]


def transition_layout(values=DEFAULT_VALUES):
    """Static index arrays describing the bar-gated transition. Precomputed once.

    The transition has exactly two regimes (docs/SPEC.md section 11, section 4.1):

        r < m - 1   the pointer advances and ``m`` is COPIED — a deterministic move
        r = m - 1   the bar wraps, ``r`` resets to 0, and ``m`` is REDRAWN

    Meter is therefore piecewise-constant and changes only at bar boundaries. That gate is
    not decoration: section 10.4 measures an ungated per-bar redraw at 24x worse held-out
    likelihood than a transition that persists.

    Args:
        values: the meter vocabulary, as counts.

    Returns:
        A tuple ``(advance_from, advance_to, wrap_from, wrap_meter, downbeat, meter_of)``.
        ``advance_from[j] -> advance_to[j]`` are the deterministic moves; ``wrap_from[j]``
        are the states that wrap and ``wrap_meter[j]`` their meter INDEX; ``downbeat`` is a
        bool array marking ``r == 0``; ``meter_of`` maps state index -> meter index.
    """
    S = states(values)
    index = {s: i for i, s in enumerate(S)}
    meter_index = {int(m): k for k, m in enumerate(values)}

    advance_from, advance_to, wrap_from, wrap_meter = [], [], [], []
    for i, (m, r) in enumerate(S):
        if r < m - 1:
            advance_from.append(i)
            advance_to.append(index[(m, r + 1)])
        else:
            wrap_from.append(i)
            wrap_meter.append(meter_index[m])

    downbeat = np.array([r == 0 for _, r in S], dtype=bool)
    meter_of = np.array([meter_index[m] for m, _ in S], dtype=np.int64)
    return (np.array(advance_from), np.array(advance_to),
            np.array(wrap_from), np.array(wrap_meter), downbeat, meter_of)


def _logsumexp_pairs(a, b):
    """Stable ``logaddexp`` over the last axis of a broadcast sum — [.., X, Y] -> [.., Y]."""
    return torch.logsumexp(a + b, dim=-2)


class Chain:
    """One crop's log-space potentials, and exact inference over them.

    Holds the three additive pieces of a linear-chain model over ``n`` beats and ``|S|``
    states — an initial vector, per-step transition matrices and per-step state potentials —
    and nothing else. Kept separate from :class:`BarPointer` so the algorithms below can be
    tested against brute-force enumeration without a model in the way.

    Attributes:
        init: ``[S]`` log initial weights.
        trans: ``[n-1, S, S]`` log transition weights, ``trans[i, j, j']``.
        state: ``[n, S]`` log state potentials.
    """

    def __init__(self, init, trans, state):
        self.init = init
        self.trans = trans
        self.state = state
        self.n, self.n_states = state.shape

    def forward_logz(self) -> torch.Tensor:
        """Scalar ``log Z``: the log sum of unnormalised weight over every state path."""
        alpha = self.init + self.state[0]
        for i in range(1, self.n):
            alpha = _logsumexp_pairs(alpha[:, None], self.trans[i - 1]) + self.state[i]
        return torch.logsumexp(alpha, dim=-1)

    def forward_backward(self):
        """Per-beat posterior marginals over states, exactly.

        Returns:
            A tuple ``(gamma, logz)``: ``gamma`` is ``[n, S]`` and each row sums to 1;
            ``logz`` is the scalar computed on the forward pass.
        """
        alphas = [self.init + self.state[0]]
        for i in range(1, self.n):
            alphas.append(_logsumexp_pairs(alphas[-1][:, None], self.trans[i - 1])
                          + self.state[i])
        logz = torch.logsumexp(alphas[-1], dim=-1)

        betas = [torch.zeros_like(alphas[-1])]
        for i in range(self.n - 1, 0, -1):
            nxt = betas[-1] + self.state[i]
            betas.append(torch.logsumexp(self.trans[i - 1] + nxt[None, :], dim=-1))
        betas.reverse()

        gamma_logp = torch.stack([a + b for a, b in zip(alphas, betas)]) - logz
        return gamma_logp.exp(), logz

    def viterbi(self):
        """The single highest-weight state path.

        Returns:
            A tuple ``(path, score)``: ``path`` is a ``[n]`` int64 tensor of state indices
            and ``score`` the scalar log weight of that path.
        """
        delta = self.init + self.state[0]
        back = []
        for i in range(1, self.n):
            scores = delta[:, None] + self.trans[i - 1]
            best, argbest = scores.max(dim=-2)
            back.append(argbest)
            delta = best + self.state[i]

        score, last = delta.max(dim=-1)
        path = [int(last)]
        for argbest in reversed(back):
            path.append(int(argbest[path[-1]]))
        path.reverse()
        return torch.tensor(path, dtype=torch.int64), score


class BeatEvidenceHead(torch.nn.Module):
    """VBPM's OWN per-beat evidence head over frozen frontend features.

    Maps one beat's pooled features to (a) a scalar downbeat-evidence potential and (b) a
    ``K``-vector of meter logits consulted only at bar crossings. It is per BEAT: this is
    the object that replaces Stage 0's reducer, and the reason there is nothing left for a
    reducer to do.

    docs/SPEC.md section 6.1 forbids reusing the frontend's own beat/downbeat activation
    channels as our evidence, which is why this head exists rather than a direct read of
    channel 1.
    """

    def __init__(self, in_dim: int, n_meters: int, hidden: int = 32, seed: int = 0):
        super().__init__()
        # forked RNG: the head is the only stochastic thing in this module and it is only
        # stochastic at INITIALISATION. Forking keeps two BarPointers built from the same
        # seed bit-identical without perturbing the caller's global RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self._build(in_dim, n_meters, hidden)

    def _build(self, in_dim: int, n_meters: int, hidden: int):
        self.body = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden, dtype=torch.float64), torch.nn.Tanh())
        self.downbeat = torch.nn.Linear(hidden, 1, dtype=torch.float64)
        self.meter = torch.nn.Linear(hidden, n_meters, dtype=torch.float64)
        # Output biases start at zero so the potentials start near zero and the chain
        # starts near its Stage-0 reduction. The output WEIGHTS must NOT: zeroing both
        # output layers makes d(loss)/d(body) exactly zero at step 0, so the body never
        # trains -- the section 10.2 failure mode, reproduced here on the first run and
        # caught by test_b3_every_parameter_receives_gradient. Small, not zero.
        for layer in (self.downbeat, self.meter):
            torch.nn.init.normal_(layer.weight, std=1e-2)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, beat_h):
        """Per-beat potentials from ``[n, D]`` beat-synchronous features.

        Args:
            beat_h: ``[n, D]`` float64 features, one row per beat.

        Returns:
            A tuple ``(downbeat_potential [n], meter_logits [n, K])``.
        """
        body = self.body(beat_h)
        return self.downbeat(body).squeeze(-1), self.meter(body)


def beat_sync(h, beats_s, t0: float, fps: float):
    """Pool frame features over each beat's own span — ``[T, D]`` -> ``[n, D]``.

    Beat ``i`` owns the frames in ``[beats_s[i], beats_s[i+1])``; the last beat owns one
    inter-beat span past itself. This is the only place frames become beats, and it is why
    ``h`` can enter the recursion once per beat rather than once per crop.

    Args:
        h: ``[T, D]`` frame features.
        beats_s: ``[n]`` beat times in seconds, absolute.
        t0: the time in seconds of frame 0 of ``h``.
        fps: frames per second (docs/SPEC.md C4: one owner, ``vbpm.data.FPS``).

    Returns:
        ``[n, D]`` float64 array, the mean feature over each beat's frames.
    """
    h = np.asarray(h, dtype=np.float64)
    beats_s = np.asarray(beats_s, dtype=np.float64)
    n, T = len(beats_s), len(h)
    if n > 1:
        edges = np.append(beats_s, beats_s[-1] + (beats_s[-1] - beats_s[-2]))
    else:
        edges = np.append(beats_s, beats_s[-1] + 1.0)

    out = np.zeros((n, h.shape[1]), dtype=np.float64)
    for i in range(n):
        lo = int(np.clip(math.floor((edges[i] - t0) * fps), 0, max(T - 1, 0)))
        hi = int(np.clip(math.ceil((edges[i + 1] - t0) * fps), lo + 1, T))
        out[i] = h[lo:hi].mean(axis=0) if hi > lo else h[lo]
    return out


class BarPointer:
    """The increment-B model: a bar-pointer chain over the given beat grid.

    Parameter sets, following docs/SPEC.md section 4.7 (there is no phi — inference is
    exact, so the encoder that phi would parameterise does not exist here):

        theta : emission p_theta(y_i|z_i) — the SAME two scalars {alpha, beta} as Stage 0,
                latent-only. h never touches it.
        psi   : the dynamics — ``init_m`` (a prior over meters), ``meter_transition``
                (a KxK sticky matrix, consulted only at bar crossings) and the per-beat
                :class:`BeatEvidenceHead`. This is the deployable path.

    ``audio`` = False disables the head entirely, which turns this into increment A (the
    static ``(m, r)`` model) and, with ``sticky_init`` large, into Stage 0's own emission.
    That reduction is the correctness test in ``tests_phase/``.
    """

    def __init__(self, values=DEFAULT_VALUES, in_dim: int = 2, audio: bool = True,
                 hidden: int = 32, sticky_init: float = 4.0, fps: float = 50.0,
                 seed: int = 0):
        self.values = tuple(int(v) for v in values)
        self.fps = float(fps)
        self.audio = bool(audio)
        K = len(self.values)
        layout = transition_layout(self.values)
        (self._adv_from, self._adv_to, self._wrap_from,
         self._wrap_meter, downbeat, meter_of) = layout
        self.n_states = len(states(self.values))
        self.downbeat_mask = torch.tensor(downbeat, dtype=torch.float64)
        self.meter_of = torch.tensor(meter_of, dtype=torch.int64)
        self.log_m = torch.tensor([math.log(m) for m in self.values], dtype=torch.float64)

        # theta — identical semantics and initialisation to Stage0
        self.alpha = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        self.beta = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)
        # psi — the dynamics. Sticky by construction: section 10.4 measures free redraw at
        # 24x worse held-out likelihood, so the diagonal starts high.
        self.init_m = torch.zeros(K, dtype=torch.float64, requires_grad=True)
        self.meter_transition = torch.tensor(
            sticky_init * np.eye(K), dtype=torch.float64, requires_grad=True)
        self.head = BeatEvidenceHead(in_dim, K, hidden, seed) if audio else None

    # -- meter is a COUNT, never an index (docs/SPEC.md C1) ---------------------------
    def to_idx(self, m: int) -> int:
        """Beats-per-bar count -> position in ``values``; raises on an illegal count."""
        m = int(m)
        if m not in self.values:
            raise ValueError(f"{m} is not a legal meter count in {self.values}")
        return self.values.index(m)

    def to_value(self, k: int) -> int:
        """Position in ``values`` -> beats-per-bar count, the inverse of ``to_idx``."""
        return int(self.values[int(k)])

    # -- the chain's three potentials ------------------------------------------------
    def initial_logits(self) -> torch.Tensor:
        """``[S]`` log initial weights: ``log p(m) - log m``, i.e. ``r`` uniform in the bar.

        A uniform ``r`` over the ``m`` legal offsets is EXACTLY Stage 0's uniform
        marginalisation of the bar offset (docs/SPEC.md section 4.3), rewritten as a
        latent's initial distribution instead of a sum inside the emission. That identity
        is the hinge that makes this module an extension of Stage 0 rather than a rival to
        it, and it is asserted as a test.
        """
        log_prior = torch.log_softmax(self.init_m, dim=-1)
        return (log_prior - self.log_m)[self.meter_of]

    def transition_logits(self, meter_logits, n_steps: int) -> torch.Tensor:
        """``[n_steps, S, S]`` log transition weights, bar-gated.

        Args:
            meter_logits: ``[n, K]`` per-beat meter logits from the head, or None.
            n_steps: the number of transitions, ``n - 1``.

        Returns:
            ``[n_steps, S, S]`` where entry ``(i, j, j')`` is the log weight of moving from
            state ``j`` at beat ``i`` to state ``j'`` at beat ``i+1``.
        """
        trans = torch.full((n_steps, self.n_states, self.n_states), NEG_INF,
                           dtype=torch.float64)

        # deterministic advance: m copied, r incremented, log-probability 0
        trans[:, self._adv_from, self._adv_to] = 0.0

        # bar crossing: r resets to 0 and m is redrawn from a locally normalised categorical
        gate = self.meter_transition[self._wrap_meter]                 # [W, K]
        if meter_logits is not None:
            gate = gate[None, :, :] + meter_logits[1:, None, :]        # [n-1, W, K]
        else:
            gate = gate[None, :, :].expand(n_steps, -1, -1)
        gate = torch.log_softmax(gate, dim=-1)                         # over the new meter

        # column of the r=0 state of each candidate meter, in state-index order
        zero_state = torch.tensor(
            [states(self.values).index((m, 0)) for m in self.values], dtype=torch.int64)
        rows = torch.tensor(self._wrap_from, dtype=torch.int64)
        trans[:, rows[:, None], zero_state[None, :]] = gate
        return trans

    def potentials(self, beat_h, n: int):
        """The chain's ``(init, trans, state)`` for one crop, with ``h`` read per beat.

        The state potential is ``S_psi(i, (m, r)) = e_psi(h_i) * 1[r == 0]`` — the audio's
        say in WHERE downbeats fall. With no head it is identically zero and the chain is a
        locally normalised HMM; with a head the chain is globally normalised instead, which
        is why :meth:`log_likelihood` is a difference of two partition functions.

        Args:
            beat_h: ``[n, D]`` beat-synchronous features, or None when ``audio`` is off.
            n: the number of beats.

        Returns:
            A tuple ``(init [S], trans [n-1, S, S], state [n, S])``.
        """
        meter_logits = None
        state = torch.zeros((n, self.n_states), dtype=torch.float64)
        if self.head is not None and beat_h is not None:
            evidence, meter_logits = self.head(beat_h)
            state = evidence[:, None] * self.downbeat_mask[None, :]
        return (self.initial_logits(),
                self.transition_logits(meter_logits, n - 1), state)

    def emission_logits(self, y) -> torch.Tensor:
        """``[n, S]`` log ``p_theta(y_i | z_i)`` — the Stage-0 emission, per beat.

        Beat ``i`` is a downbeat with probability ``sigmoid(alpha)`` when ``r_i == 0`` and
        ``sigmoid(beta)`` otherwise. Two scalars, latent-only, no ``h``.
        """
        y = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
        lsig = torch.nn.functional.logsigmoid
        on = y * lsig(self.alpha) + (1 - y) * lsig(-self.alpha)
        off = y * lsig(self.beta) + (1 - y) * lsig(-self.beta)
        mask = self.downbeat_mask[None, :]
        return on[:, None] * mask + off[:, None] * (1 - mask)

    # -- exact inference --------------------------------------------------------------
    def chains(self, beat_h, y, n: int):
        """The prior chain and the posterior chain for one crop.

        Returns:
            A tuple ``(prior_chain, joint_chain)``. The prior chain reads ``h`` only and is
            the deployable object (docs/SPEC.md C2); the joint chain adds the emission and
            exists only during training. ``joint_chain`` is None when ``y`` is None.
        """
        init, trans, state = self.potentials(beat_h, n)
        prior = Chain(init, trans, state)
        joint = None if y is None else Chain(init, trans, state + self.emission_logits(y))
        return prior, joint

    def log_likelihood(self, beat_h, y) -> torch.Tensor:
        """Scalar exact ``log p(y | h)`` — the training objective.

        Both partition functions come from the same forward algorithm, so this is a
        difference of two exact quantities and carries no sampling noise and no bound
        slack. Inference being exact, this is simultaneously the ELBO and the log evidence:
        the KL slack is zero and there is no variational gap to measure HERE. Measuring
        that gap is Stage 1's job, and this number is the ceiling it is measured against.
        """
        prior, joint = self.chains(beat_h, y, len(y))
        return joint.forward_logz() - prior.forward_logz()

    def decode(self, beat_h, n: int):
        """DEPLOYABLE read-out: Viterbi over ``(m, r)`` from ``h`` alone.

        Reads ``h`` and never ``y`` (docs/SPEC.md C2), which is what makes this the path
        that survives to deployment.

        Args:
            beat_h: ``[n, D]`` beat-synchronous features, or None.
            n: the number of beats.

        Returns:
            A tuple ``(m_hat, downbeat_index)``: ``m_hat`` is a beats-per-bar COUNT (the
            modal meter on the Viterbi path, converted through ``to_value`` exactly once)
            and ``downbeat_index`` a ``[?]`` int array of the beat positions the path calls
            downbeats.
        """
        prior, _ = self.chains(beat_h, None, n)
        with torch.no_grad():
            path, _ = prior.viterbi()
        meters = self.meter_of[path]
        modal = int(torch.bincount(meters, minlength=len(self.values)).argmax())
        downbeats = np.flatnonzero(self.downbeat_mask[path].numpy() > 0.5)
        return self.to_value(modal), downbeats

    def marginals(self, beat_h, n: int) -> torch.Tensor:
        """``[n, S]`` deployable per-beat posterior marginals over states, from ``h`` alone."""
        prior, _ = self.chains(beat_h, None, n)
        gamma, _ = prior.forward_backward()
        return gamma

    # -- parameters and training ------------------------------------------------------
    def param_groups(self) -> dict:
        """The theta/psi split, as ``{group: {name: tensor}}``. There is no phi group.

        Every trainable tensor must receive gradient (docs/SPEC.md section 4.7, section
        10.2 — half a network once trained at exactly zero gradient here). This surface is
        what makes that a test rather than a post-mortem.
        """
        psi = {"init_m": self.init_m, "meter_transition": self.meter_transition}
        if self.head is not None:
            psi.update({f"head.{k}": v for k, v in self.head.named_parameters()})
        return {"theta": {"alpha": self.alpha, "beta": self.beta}, "psi": psi}

    def named_params(self) -> dict:
        """Every trainable tensor, flat — the gradient-audit surface."""
        out: dict = {}
        for group in self.param_groups().values():
            out.update(group)
        return out

    def trainable_params(self) -> list:
        """The optimiser set, with each TIED parameter included exactly once."""
        seen, params = set(), []
        for p in self.named_params().values():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        return params

    def fit(self, crops, steps: int = 300, lr: float = 0.05, verbose: bool = False):
        """Maximise the mean exact ``log p(y|h)`` over crops. Adam, full batch, float64.

        ``crops`` entries need ``beat_h`` (``[n, D]``, from :func:`beat_sync`) and ``y``.
        Deterministic: nothing here samples.
        """
        cache = [(None if c.get("beat_h") is None
                  else torch.as_tensor(np.asarray(c["beat_h"], dtype=np.float64)),
                  c["y"]) for c in crops]
        opt = torch.optim.Adam(self.trainable_params(), lr=lr)

        for step in range(steps):
            opt.zero_grad()
            loss = -torch.stack([self.log_likelihood(bh, y) / max(len(y), 1)
                                 for bh, y in cache]).mean()
            loss.backward()
            opt.step()
            if verbose and (step % 50 == 0 or step == steps - 1):
                print(f"  step {step:4d}  -logp/beat {float(loss):.5f}", flush=True)

        return self


def downbeat_f(pred_index, true_index) -> float:
    """Downbeat F on a GIVEN beat grid: exact set F over beat INDICES.

    Both sets are subsets of the same known grid, so no time tolerance is needed — and none
    should be used. A +-70 ms window on a fast grid can admit a NEIGHBOURING beat and
    silently forgive an off-by-one phase error, which is the one error mode a bar pointer
    exists to measure.

    Args:
        pred_index: predicted downbeat positions, as beat indices.
        true_index: annotated downbeat positions, as beat indices.

    Returns:
        F-measure in [0, 1]; 0.0 when either set is empty and the other is not, and 1.0
        when both are empty.
    """
    pred, true = set(int(i) for i in pred_index), set(int(i) for i in true_index)
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    hit = len(pred & true)
    precision, recall = hit / len(pred), hit / len(true)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
