"""Increment-B tests: the criteria pre-registered in docs/PHASE_PLAN.md section 4.3.

Run: /disk4/anaconda3/envs/vbpm/bin/python -m pytest tests_phase -q

These are NOT in ``tests/``. That suite is frozen and normative (docs/SPEC.md section 9)
and binds to ``Stage0``; it is re-run unchanged and must stay green.

The tests are grouped by the criterion they discharge:

    B-1  the chain EXTENDS Stage 0 -- reduced to Stage 0's configuration it reproduces
         ``Stage0.emission_logp_all`` to float64 precision
    B-2  internal exactness -- forward == backward, marginals normalised, Viterbi == the
         brute-force maximum
    B-3  phase recovery on the noise-free bench
    B-4  meter change recovery, which Stage 0 provably cannot do
    B-6  the deployable path never reads y
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from tests_phase import synth_phase

from vbpm.barpointer import BarPointer, Chain, beat_sync, downbeat_f, states
from vbpm.data import extract_crops_unaligned
from vbpm.stage0 import Stage0

EXACT = 1e-10
VALUES = (2, 3, 4)


def _freeze_meter(model, m):
    """Configure the chain as Stage 0: no audio, no switching, meter pinned to ``m``."""
    with torch.no_grad():
        model.init_m.fill_(-50.0)
        model.init_m[model.to_idx(m)] = 0.0
        model.meter_transition.fill_(-50.0)
        model.meter_transition.fill_diagonal_(50.0)


# ---------------------------------------------------------------------------------- B-1
@pytest.mark.parametrize("m", VALUES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_b1_reduces_to_stage0_emission(m, seed):
    """B-1: pinned to one meter with no audio, the chain IS Stage 0's emission.

    Stage 0 marginalises the bar offset uniformly inside the emission; the chain does it
    as a uniform initial distribution over ``r``. That these agree to 1e-10 is the formal
    statement that increment B EXTENDS Stage 0 rather than replacing it.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=17).astype(np.float64)

    stage0 = Stage0(VALUES)
    model = BarPointer(VALUES, audio=False)
    with torch.no_grad():
        model.alpha.copy_(stage0.alpha)
        model.beta.copy_(stage0.beta)
    _freeze_meter(model, m)

    chain_logp = float(model.log_likelihood(None, y))
    stage0_logp = float(stage0.emission_logp_all(y)[stage0.to_idx(m)])
    assert abs(chain_logp - stage0_logp) < EXACT, (chain_logp, stage0_logp)


def test_b1_uniform_meter_reduces_to_stage0_mixture():
    """B-1: with a uniform meter prior the chain equals Stage 0's mixture over meters."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=24).astype(np.float64)

    stage0 = Stage0(VALUES)
    model = BarPointer(VALUES, audio=False)
    with torch.no_grad():
        model.alpha.copy_(stage0.alpha)
        model.beta.copy_(stage0.beta)
        model.meter_transition.fill_(-50.0)
        model.meter_transition.fill_diagonal_(50.0)

    expected = float(torch.logsumexp(
        stage0.emission_logp_all(y) - np.log(len(VALUES)), dim=-1))
    assert abs(float(model.log_likelihood(None, y)) - expected) < EXACT


# ---------------------------------------------------------------------------------- B-2
def _random_chain(n=9, n_states=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return Chain(torch.randn(n_states, generator=g, dtype=torch.float64),
                 torch.randn(n - 1, n_states, n_states, generator=g, dtype=torch.float64),
                 torch.randn(n, n_states, generator=g, dtype=torch.float64))


def _brute_force_logz(chain):
    """Log-sum-exp over every state path, enumerated. Only tractable for tiny chains."""
    scores = []
    for path in itertools.product(range(chain.n_states), repeat=chain.n):
        acc = chain.init[path[0]] + chain.state[0, path[0]]
        for i in range(1, chain.n):
            acc = acc + chain.trans[i - 1, path[i - 1], path[i]] + chain.state[i, path[i]]
        scores.append(acc)
    return torch.logsumexp(torch.stack(scores), dim=0), max(float(s) for s in scores)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_b2_forward_matches_brute_force(seed):
    """B-2: the forward algorithm's logZ equals the enumerated one."""
    chain = _random_chain(n=5, n_states=3, seed=seed)
    logz, _ = _brute_force_logz(chain)
    assert abs(float(chain.forward_logz()) - float(logz)) < EXACT


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_b2_viterbi_matches_brute_force(seed):
    """B-2(c): the Viterbi score equals the maximum over enumerated paths."""
    chain = _random_chain(n=5, n_states=3, seed=seed)
    _, best = _brute_force_logz(chain)
    _, score = chain.viterbi()
    assert abs(float(score) - best) < EXACT


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_b2_forward_backward_consistent(seed):
    """B-2(a, b): backward logZ matches forward, and every marginal row sums to 1."""
    chain = _random_chain(seed=seed)
    gamma, logz = chain.forward_backward()
    assert abs(float(logz) - float(chain.forward_logz())) < EXACT
    assert torch.allclose(gamma.sum(-1), torch.ones(chain.n, dtype=torch.float64),
                          atol=EXACT)


def test_b2_marginals_respect_the_bar_gate():
    """B-2: the model's own chain never puts mass on an illegal (m, r) transition.

    A pointer that could jump r without wrapping would make phase unlearnable, so this
    pins the gate itself rather than the algorithm.
    """
    model = BarPointer(VALUES, audio=False)
    gamma = model.marginals(None, 20)
    layout = states(VALUES)
    for i in range(1, len(gamma)):
        for j, (m, r) in enumerate(layout):
            if gamma[i, j] > 1e-9 and r > 0:
                # mass at (m, r>0) requires mass at (m, r-1) one beat earlier
                assert gamma[i - 1, layout.index((m, r - 1))] > 1e-12


# ---------------------------------------------------------------------------------- B-3
def _score(model, songs):
    f, meter_hits, offset_hits = [], 0, 0
    for s in songs:
        beat_h = torch.as_tensor(s["beat_h"])
        m_hat, downbeats = model.decode(beat_h, len(s["y"]))
        f.append(downbeat_f(downbeats, s["downbeat_index"]))
        meter_hits += int(m_hat == s["m_true"])
        r0_hat = int((-int(downbeats[0])) % m_hat) if len(downbeats) else -1
        offset_hits += int(r0_hat == s["r0"])
    return float(np.mean(f)), meter_hits / len(songs), offset_hits / len(songs)


def test_b3_phase_recovery_on_clean_bench():
    """B-3: on the noise-free phase bench, F / meter / offset must all clear 0.95.

    The bench is exactly periodic, so this is a MACHINERY check (docs/SPEC.md section 6.4
    sets the same 0.95+ expectation for Stage 0), not a benchmark. It is also the gate on
    real data: a phase model that cannot recover known phase from clean input is broken.
    """
    train = synth_phase.make_dataset(12, seed=0)
    held = synth_phase.make_dataset(12, seed=1000)

    model = BarPointer(VALUES, in_dim=2, audio=True)
    model.fit(train, steps=300, lr=0.05)

    f, meter_acc, offset_acc = _score(model, held)
    assert f >= 0.95, f"downbeat F {f:.3f}"
    assert meter_acc >= 0.95, f"meter accuracy {meter_acc:.3f}"
    assert offset_acc >= 0.95, f"offset accuracy {offset_acc:.3f}"


def test_b3_beats_the_r0_baseline():
    """B-3: the un-aligned bench is not solvable by always predicting r = 0."""
    held = synth_phase.make_dataset(12, seed=1000)
    baseline = float(np.mean([
        downbeat_f(np.arange(0, len(s["y"]), s["m_true"]), s["downbeat_index"])
        for s in held]))
    assert baseline < 0.7, f"r=0 baseline scores {baseline:.3f}; the bench is bar-aligned"


def test_b3_every_parameter_receives_gradient():
    """Section 4.7 / 10.2: read gradients, do not assume them. Half a network once died."""
    songs = synth_phase.make_dataset(2, seed=3)
    model = BarPointer(VALUES, in_dim=2, audio=True)
    loss = -torch.stack([model.log_likelihood(torch.as_tensor(s["beat_h"]), s["y"])
                         for s in songs]).mean()
    loss.backward()

    dead = [name for name, p in model.named_params().items()
            if p.grad is None or bool((p.grad == 0).all())]
    assert not dead, f"parameters at exactly zero gradient: {dead}"


# ---------------------------------------------------------------------------------- B-4
def test_b4_recovers_a_mid_song_meter_change():
    """B-4: the Viterbi path finds a single mid-song meter change to within +-1 bar.

    Stage 0 cannot represent this at all (one m per crop, docs/SPEC.md section 4.1); it is
    increment B's clearest qualitative win, so it is asserted rather than assumed.
    """
    train = synth_phase.make_dataset(12, seed=0)
    model = BarPointer(VALUES, in_dim=2, audio=True)
    model.fit(train, steps=300, lr=0.05)

    songs = synth_phase.make_meter_change_dataset(20, seed=11)
    found = 0
    for s in songs:
        prior, _ = model.chains(torch.as_tensor(s["beat_h"]), None, len(s["y"]))
        with torch.no_grad():
            path, _ = prior.viterbi()
        meters = np.array([states(VALUES)[int(j)][0] for j in path])
        switches = np.flatnonzero(meters[1:] != meters[:-1])
        if len(switches):
            tolerance = s["m_true"] + s["m_after"]
            found += int(np.min(np.abs(switches + 1 - s["change_beat"])) <= tolerance)
    assert found / len(songs) >= 0.8, f"change recovered in {found}/{len(songs)}"


# ---------------------------------------------------------------------------------- B-6
def test_b6_deployable_path_never_reads_y():
    """B-6 (C2): ``decode`` takes no y, and deranging y leaves a fixed model's decode alone.

    A ``predict(h)`` signature does not prove annotation-freedom on its own, so this
    corrupts y and checks the DECODE is bit-identical.
    """
    songs = synth_phase.make_dataset(4, seed=5)
    model = BarPointer(VALUES, in_dim=2, audio=True)
    model.fit(songs, steps=50, lr=0.05)

    before = [model.decode(torch.as_tensor(s["beat_h"]), len(s["y"])) for s in songs]
    for s in songs:                      # destroy the labels; psi and theta are unchanged
        s["y"] = 1.0 - np.asarray(s["y"])
    after = [model.decode(torch.as_tensor(s["beat_h"]), len(s["y"])) for s in songs]

    for (m0, d0), (m1, d1) in zip(before, after):
        assert m0 == m1 and np.array_equal(d0, d1)


# ------------------------------------------------------------------ data + read-out ---
def test_unaligned_crops_actually_vary_the_offset():
    """B-0's mechanism, on synthetic annotations: the crop path must not pin r to 0.

    ``extract_crops`` yields r = 0 for 18902 of 18902 real crops; the un-aligned twin must
    not. (The real-data form of B-0 is a measurement, not a unit test -- see the report.)
    """
    rng = np.random.default_rng(0)
    offsets = set()
    for trial in range(24):
        beats = np.arange(200, dtype=np.float64) * 0.5
        downs = beats[::3]
        crops, _ = extract_crops_unaligned(beats, downs, rng=np.random.default_rng(trial))
        offsets.update(c["r0"] for c in crops)
    assert len(offsets) >= 2, f"un-aligned crops still pin the offset: {offsets}"
    assert rng is not None


def test_beat_sync_pools_each_beats_own_frames():
    """Frame features become beat features exactly once, at the beat's own span."""
    fps = 50.0
    h = np.arange(500, dtype=np.float64)[:, None]
    beats = np.array([0.0, 1.0, 2.0, 3.0])
    pooled = beat_sync(h, beats, t0=0.0, fps=fps)
    assert pooled.shape == (4, 1)
    assert pooled[0, 0] < pooled[1, 0] < pooled[2, 0] < pooled[3, 0]


def test_downbeat_f_is_exact_on_the_grid():
    """The metric is a set F over beat indices -- no time tolerance to forgive off-by-one."""
    assert downbeat_f([0, 4, 8], [0, 4, 8]) == 1.0
    assert downbeat_f([1, 5, 9], [0, 4, 8]) == 0.0
    assert downbeat_f([], []) == 1.0
    assert downbeat_f([0], []) == 0.0
    assert abs(downbeat_f([0, 4], [0, 4, 8]) - 0.8) < EXACT
