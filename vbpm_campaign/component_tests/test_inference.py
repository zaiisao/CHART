"""Characterization / oracle tests for the INFERENCE component of VBPM.

Components under test (IMPORTED, never modified):
    rungs/bar_pointer/inference.py    -- dense reference forward algorithm + Viterbi (log-space)
    rungs/bar_pointer/structured_dp.py-- StructuredBarPointerDP: the O(K+M V^2) engine
    rungs/bar_pointer/state_space.py  -- Krebs 2015 bar-pointer state space
    models/svt_core.py                -- SVTModel particle filter (sample_from_prior_pf,
                                          _systematic_resample)

Intended math (docs/ELBO_for_DBN.md): the bar-pointer states are collapsed into one composite
HMM, so forward / Viterbi inference is EXACT (finite + Markov + factorized). Every property below
is checked against an INDEPENDENT oracle -- brute-force enumeration over ALL state paths, a
closed-form analytic value, the autograd/marginal identity, or a resampling count invariant --
NEVER against the code's own output.

Run with:
    /home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python component_tests/test_inference.py
"""
import sys
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM")

import itertools
import math

import numpy as np
import torch

from rungs.bar_pointer import inference as inf
from rungs.bar_pointer.structured_dp import StructuredBarPointerDP
from rungs.bar_pointer.state_space import BarPointerStateSpace
from models.svt_core import SVTModel

DT = torch.float64
NEG_INF = -float("inf")

_RESULTS = []  # (property, oracle, measured, PASS/FAIL)


def _record(name, oracle, measured, ok):
    _RESULTS.append((name, oracle, measured, "PASS" if ok else "FAIL"))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"        oracle   = {oracle}")
    print(f"        measured = {measured}")


# ---------------------------------------------------------------------------
# Brute-force oracles: enumerate EVERY state path.
# ---------------------------------------------------------------------------

def brute_force_forward(log_pi, log_A, log_B):
    """log p(obs) = logsumexp over all state paths of the joint log-prob.

    log_A may be 2-D [n,n] (static) or 3-D [T-1,n,n] (time-varying). -inf allowed."""
    T, n = log_B.shape
    time_varying = log_A.dim() == 3
    path_scores = []
    for path in itertools.product(range(n), repeat=T):
        s = log_pi[path[0]].item() + log_B[0, path[0]].item()
        for t in range(1, T):
            a = log_A[t - 1] if time_varying else log_A
            s = s + a[path[t - 1], path[t]].item() + log_B[t, path[t]].item()
        path_scores.append(s)
    return torch.logsumexp(torch.tensor(path_scores, dtype=DT), dim=0)


def brute_force_marginals(log_pi, log_A, log_B):
    """Per-step posterior marginals gamma_t(k)=p(z_t=k|obs) by exact enumeration."""
    T, n = log_B.shape
    time_varying = log_A.dim() == 3
    paths = list(itertools.product(range(n), repeat=T))
    scores = []
    for path in paths:
        s = log_pi[path[0]].item() + log_B[0, path[0]].item()
        for t in range(1, T):
            a = log_A[t - 1] if time_varying else log_A
            s = s + a[path[t - 1], path[t]].item() + log_B[t, path[t]].item()
        scores.append(s)
    scores = torch.tensor(scores, dtype=DT)
    logZ = torch.logsumexp(scores, dim=0)
    post = torch.exp(scores - logZ)  # p(path|obs)
    gamma = torch.zeros(T, n, dtype=DT)
    for p_idx, path in enumerate(paths):
        for t in range(T):
            gamma[t, path[t]] += post[p_idx]
    return gamma


def brute_force_viterbi(log_pi, log_A, log_B):
    """Exact MAP path and its score by enumeration. Returns (path_tuple, score)."""
    T, n = log_B.shape
    time_varying = log_A.dim() == 3
    best_path, best_score = None, NEG_INF
    for path in itertools.product(range(n), repeat=T):
        s = log_pi[path[0]].item() + log_B[0, path[0]].item()
        for t in range(1, T):
            a = log_A[t - 1] if time_varying else log_A
            s = s + a[path[t - 1], path[t]].item() + log_B[t, path[t]].item()
        if s > best_score:
            best_score, best_path = s, path
    return best_path, best_score


def _rand_log_stochastic(n, gen, allow_zero=False):
    """A random ROW-stochastic transition, returned in log space (rows sum to 1)."""
    p = torch.rand(n, n, generator=gen, dtype=DT) + 0.05
    if allow_zero:  # knock out ~1/3 of entries to test -inf handling (keep diagonal)
        mask = torch.rand(n, n, generator=gen, dtype=DT) < 0.33
        mask.fill_diagonal_(False)
        p = p.masked_fill(mask, 0.0)
    p = p / p.sum(dim=1, keepdim=True)
    return torch.log(p)  # p==0 -> -inf


def _rand_log_emission(T, n, gen):
    return torch.log(torch.rand(T, n, generator=gen, dtype=DT) + 0.05)


# ===========================================================================
# 1. Dense forward algorithm  ==  brute-force enumeration
# ===========================================================================

def test_dense_forward_vs_bruteforce():
    gen = torch.Generator().manual_seed(1)
    n, T = 4, 5
    log_pi = torch.log(torch.softmax(torch.randn(n, generator=gen, dtype=DT), 0))
    log_A = _rand_log_stochastic(n, gen)
    log_B = _rand_log_emission(T, n, gen)

    code = inf.forward_log_likelihood(log_pi, log_A, log_B).item()
    oracle = brute_force_forward(log_pi, log_A, log_B).item()
    err = abs(code - oracle)
    _record("dense forward_log_likelihood == brute-force enumeration (n=4,T=5)",
            f"{oracle:.12f}", f"{code:.12f}  (|err|={err:.2e})", err < 1e-10)


def test_dense_forward_timevarying():
    gen = torch.Generator().manual_seed(2)
    n, T = 3, 5
    log_pi = torch.log(torch.softmax(torch.randn(n, generator=gen, dtype=DT), 0))
    log_A = torch.stack([_rand_log_stochastic(n, gen) for _ in range(T - 1)])  # [T-1,n,n]
    log_B = _rand_log_emission(T, n, gen)

    code = inf.forward_log_likelihood(log_pi, log_A, log_B).item()
    oracle = brute_force_forward(log_pi, log_A, log_B).item()
    err = abs(code - oracle)
    _record("dense forward, TIME-VARYING transition == brute force (n=3,T=5)",
            f"{oracle:.12f}", f"{code:.12f}  (|err|={err:.2e})", err < 1e-10)


def test_dense_forward_forbidden_transitions():
    """-inf (probability-0) transitions must be honoured exactly (no leakage)."""
    gen = torch.Generator().manual_seed(3)
    n, T = 4, 5
    log_pi = torch.log(torch.softmax(torch.randn(n, generator=gen, dtype=DT), 0))
    log_A = _rand_log_stochastic(n, gen, allow_zero=True)
    log_B = _rand_log_emission(T, n, gen)

    code = inf.forward_log_likelihood(log_pi, log_A, log_B).item()
    oracle = brute_force_forward(log_pi, log_A, log_B).item()
    err = abs(code - oracle)
    finite = math.isfinite(code)
    _record("dense forward with -inf (forbidden) transitions == brute force",
            f"{oracle:.12f}", f"{code:.12f}  (|err|={err:.2e}, finite={finite})",
            finite and err < 1e-10)


# ===========================================================================
# 2. Posterior marginals via the autograd identity  d logp / d logB = gamma
#    (a) sum to 1 at every t   (b) equal brute-force enumerated marginals
# ===========================================================================

def test_autograd_posterior_marginals():
    gen = torch.Generator().manual_seed(4)
    n, T = 4, 5
    log_pi = torch.log(torch.softmax(torch.randn(n, generator=gen, dtype=DT), 0))
    log_A = _rand_log_stochastic(n, gen)
    log_B = _rand_log_emission(T, n, gen).requires_grad_(True)

    ll = inf.forward_log_likelihood(log_pi, log_A, log_B)
    ll.backward()
    gamma_code = log_B.grad.detach()             # [T,n] posterior marginals
    gamma_oracle = brute_force_marginals(log_pi, log_A, log_B.detach())

    row_sums = gamma_code.sum(dim=1)
    sum_err = (row_sums - 1.0).abs().max().item()
    _record("posterior marginals (autograd d logp/d logB) SUM TO 1 at every t",
            "1.0 for all t", f"max|sum-1| = {sum_err:.2e}", sum_err < 1e-9)

    marg_err = (gamma_code - gamma_oracle).abs().max().item()
    _record("posterior marginals (autograd) == brute-force enumerated marginals",
            "exact gamma_t(k)", f"max|err| = {marg_err:.2e}", marg_err < 1e-9)


# ===========================================================================
# 3. Log-domain stability / shift invariance
#    Adding constant c to every emission entry:
#      - forward_log_likelihood increases by EXACTLY T*c  (T emissions in a path)
#      - the normalized posterior (marginals) is UNCHANGED
#      - the Viterbi path is UNCHANGED
# ===========================================================================

def test_shift_invariance():
    gen = torch.Generator().manual_seed(5)
    n, T = 4, 6
    log_pi = torch.log(torch.softmax(torch.randn(n, generator=gen, dtype=DT), 0))
    log_A = _rand_log_stochastic(n, gen)
    log_B = _rand_log_emission(T, n, gen)
    c = 500.0  # large enough to overflow a naive (non-log) implementation

    ll0 = inf.forward_log_likelihood(log_pi, log_A, log_B).item()
    ll1 = inf.forward_log_likelihood(log_pi, log_A, log_B + c).item()
    shift = ll1 - ll0
    _record("shift-invariance: +c to all emissions raises logZ by EXACTLY T*c",
            f"T*c = {T * c:.6f}", f"{shift:.6f}  (|err|={abs(shift - T * c):.2e})",
            abs(shift - T * c) < 1e-6)

    # normalized posterior unchanged (extract via autograd both times)
    lB0 = log_B.clone().requires_grad_(True)
    inf.forward_log_likelihood(log_pi, log_A, lB0).backward()
    lB1 = (log_B + c).clone().requires_grad_(True)
    inf.forward_log_likelihood(log_pi, log_A, lB1).backward()
    gamma_err = (lB0.grad - lB1.grad).abs().max().item()
    _record("shift-invariance: normalized posterior marginals UNCHANGED by +c",
            "0.0", f"max|gamma_shifted - gamma| = {gamma_err:.2e}", gamma_err < 1e-9)

    path0 = inf.viterbi(log_pi, log_A, log_B).tolist()
    path1 = inf.viterbi(log_pi, log_A, log_B + c).tolist()
    _record("shift-invariance: Viterbi MAP path UNCHANGED by +c",
            str(path0), str(path1), path0 == path1)


# ===========================================================================
# 4. Dense Viterbi  ==  brute-force MAP path
# ===========================================================================

def test_dense_viterbi_vs_bruteforce():
    for seed in (6, 7, 8):
        gen = torch.Generator().manual_seed(seed)
        n, T = 4, 5
        log_pi = torch.log(torch.softmax(torch.randn(n, generator=gen, dtype=DT), 0))
        log_A = _rand_log_stochastic(n, gen)
        log_B = _rand_log_emission(T, n, gen)

        path_code = inf.viterbi(log_pi, log_A, log_B).tolist()
        path_oracle, score_oracle = brute_force_viterbi(log_pi, log_A, log_B)
        # score of the code's path under the same factors
        s = log_pi[path_code[0]].item() + log_B[0, path_code[0]].item()
        for t in range(1, T):
            s += log_A[path_code[t - 1], path_code[t]].item() + log_B[t, path_code[t]].item()
        score_ok = abs(s - score_oracle) < 1e-10
        # paths may differ only on exact ties; compare by SCORE (the real oracle)
        _record(f"dense Viterbi score == brute-force MAP score (seed={seed})",
                f"{score_oracle:.12f} path={list(path_oracle)}",
                f"{s:.12f} path={path_code}", score_ok)


# ===========================================================================
# 5. Structured bar-pointer DP  ==  dense reference  ==  brute force
#    (chain of certificates on a TINY Krebs state space)
# ===========================================================================

def _tiny_state_space():
    # fps=4, [80,120] BPM -> intervals {2,3}, num_tempi=2, states_per_beat=5,
    # beats_per_bar=2 -> num_states=10 (brute-forceable at T=4: 10^4 paths).
    return BarPointerStateSpace(fps=4.0, min_bpm=80.0, max_bpm=120.0,
                                beats_per_bar=2, observation_lambda=16, num_tempi=None)


def _structured_setup(transition_lambda, T, seed):
    ss = _tiny_state_space()
    dp = StructuredBarPointerDP(ss, device="cpu", dtype=DT)
    n = ss.num_states
    gen = torch.Generator().manual_seed(seed)
    log_pi = torch.log(torch.softmax(torch.randn(n, generator=gen, dtype=DT), 0))
    log_tempo_A = dp.build_log_tempo_transition(transition_lambda)     # [V,V]
    log_B = _rand_log_emission(T, n, gen)                              # per-state emission
    dense_A = dp.dense_transition(log_tempo_A)                        # [n,n]
    return ss, dp, log_pi, log_tempo_A, log_B, dense_A


def test_structured_forward_vs_dense_and_bruteforce():
    ss, dp, log_pi, log_tempo_A, log_B, dense_A = _structured_setup(1.0, T=4, seed=9)

    structured = dp.forward_log_likelihood(log_pi, log_tempo_A, log_B).item()
    dense = inf.forward_log_likelihood(log_pi, dense_A, log_B).item()
    oracle = brute_force_forward(log_pi, dense_A, log_B).item()

    err_sd = abs(structured - dense)
    err_so = abs(structured - oracle)
    _record(f"structured DP forward == dense forward (num_states={ss.num_states})",
            f"{dense:.12f}", f"{structured:.12f}  (|err|={err_sd:.2e})", err_sd < 1e-9)
    _record("structured DP forward == brute-force enumeration",
            f"{oracle:.12f}", f"{structured:.12f}  (|err|={err_so:.2e})", err_so < 1e-9)


def test_structured_forward_forbidden_tempo_jumps():
    """High transition_lambda forces off-diagonal tempo jumps to -inf (forbidden).
    Structured DP must still equal the dense/brute-force answer exactly."""
    ss, dp, log_pi, log_tempo_A, log_B, dense_A = _structured_setup(100.0, T=4, seed=10)
    n_forbidden = int(torch.isinf(log_tempo_A).sum().item())

    structured = dp.forward_log_likelihood(log_pi, log_tempo_A, log_B).item()
    oracle = brute_force_forward(log_pi, dense_A, log_B).item()
    err = abs(structured - oracle)
    _record(f"structured forward with {n_forbidden} forbidden tempo jumps == brute force",
            f"{oracle:.12f}", f"{structured:.12f}  (|err|={err:.2e})",
            math.isfinite(structured) and err < 1e-9)


def test_structured_viterbi_vs_bruteforce():
    ss, dp, log_pi, log_tempo_A, log_B, dense_A = _structured_setup(1.0, T=4, seed=11)

    path, score = dp.viterbi(log_pi, log_tempo_A, log_B, return_log_score=True)
    path = path.tolist()
    oracle_path, oracle_score = brute_force_viterbi(log_pi, dense_A, log_B)
    # verify structured MAP score under the dense factors
    s = log_pi[path[0]].item() + log_B[0, path[0]].item()
    for t in range(1, len(path)):
        s += dense_A[path[t - 1], path[t]].item() + log_B[t, path[t]].item()
    score_matches_oracle = abs(oracle_score - s) < 1e-9
    score_matches_return = abs(score - s) < 1e-9
    _record("structured Viterbi path is a LEGAL max-scoring path (score==brute-force MAP)",
            f"{oracle_score:.12f}", f"{s:.12f}  (|err|={abs(oracle_score - s):.2e})",
            score_matches_oracle)
    _record("structured Viterbi returned log-score == recomputed path score",
            f"{s:.12f}", f"{score:.12f}", score_matches_return)


def test_structured_compact_emission_gather():
    """Compact [T,num_classes] emission + state_to_class must equal the expanded
    per-state emission (the differentiable on-the-fly gather)."""
    ss = _tiny_state_space()
    dp = StructuredBarPointerDP(ss, device="cpu", dtype=DT)
    n = ss.num_states
    gen = torch.Generator().manual_seed(12)
    log_pi = torch.log(torch.softmax(torch.randn(n, generator=gen, dtype=DT), 0))
    log_tempo_A = dp.build_log_tempo_transition(1.0)
    T, n_classes = 5, 3
    compact = _rand_log_emission(T, n_classes, gen)                   # [T,3]
    state_to_class = torch.from_numpy(ss.position_classes).long()     # [n]
    expanded = compact[:, state_to_class]                            # [T,n]

    ll_compact = dp.forward_log_likelihood(log_pi, log_tempo_A, compact,
                                           state_to_class=state_to_class).item()
    ll_expanded = dp.forward_log_likelihood(log_pi, log_tempo_A, expanded).item()
    err = abs(ll_compact - ll_expanded)
    _record("structured forward: compact-gather emission == expanded per-state emission",
            f"{ll_expanded:.12f}", f"{ll_compact:.12f}  (|err|={err:.2e})", err < 1e-12)


# ===========================================================================
# 6. Edge cases: T=1, single state
# ===========================================================================

def test_edge_T1():
    gen = torch.Generator().manual_seed(13)
    n = 5
    log_pi = torch.log(torch.softmax(torch.randn(n, generator=gen, dtype=DT), 0))
    log_A = _rand_log_stochastic(n, gen)
    log_B = _rand_log_emission(1, n, gen)                             # T=1

    code = inf.forward_log_likelihood(log_pi, log_A, log_B).item()
    oracle = torch.logsumexp(log_pi + log_B[0], dim=0).item()          # no transitions
    err = abs(code - oracle)
    _record("edge T=1: forward == logsumexp(log_pi + logB[0]) (no transition applied)",
            f"{oracle:.12f}", f"{code:.12f}  (|err|={err:.2e})", err < 1e-12)

    path = inf.viterbi(log_pi, log_A, log_B).tolist()
    oracle_state = int(torch.argmax(log_pi + log_B[0]))
    _record("edge T=1: Viterbi == argmax(log_pi + logB[0])",
            f"[{oracle_state}]", str(path), path == [oracle_state])


def test_edge_single_state():
    gen = torch.Generator().manual_seed(14)
    T = 6
    log_pi = torch.zeros(1, dtype=DT)                                 # log 1
    log_A = torch.zeros(1, 1, dtype=DT)                              # self-loop, log 1
    log_B = _rand_log_emission(T, 1, gen)

    code = inf.forward_log_likelihood(log_pi, log_A, log_B).item()
    oracle = log_B.sum().item()                                       # only one path
    err = abs(code - oracle)
    _record("edge single-state: forward == sum of emissions (unique path)",
            f"{oracle:.12f}", f"{code:.12f}  (|err|={err:.2e})", err < 1e-12)


# ===========================================================================
# 7. Particle filter (svt_core): systematic resampling exactness invariant
#    Systematic resampling GUARANTEES each particle i is copied either
#    floor(N*p_i) or ceil(N*p_i) times, for ANY uniform offset. Exact invariant.
# ===========================================================================

def test_systematic_resample_count_invariant():
    torch.manual_seed(0)
    N = 64
    worst_lo = 0    # violations below floor
    worst_hi = 0    # violations above ceil
    total_mismatch = 0
    for trial in range(200):
        raw = torch.rand(N, dtype=DT) + (0.01 if trial % 3 else 0.0)
        if trial % 5 == 0:  # zero out some weights (test that they are ~never selected)
            raw[torch.rand(N) < 0.4] = 0.0
        if raw.sum() == 0:
            continue
        w = raw / raw.sum()
        idx = SVTModel._systematic_resample(w)
        assert idx.shape[0] == N and idx.min() >= 0 and idx.max() < N
        counts = torch.bincount(idx, minlength=N).to(DT)
        lo = torch.floor(N * w)
        hi = torch.ceil(N * w)
        worst_lo = max(worst_lo, int((counts < lo - 1e-9).sum()))
        worst_hi = max(worst_hi, int((counts > hi + 1e-9).sum()))
        if counts.sum() != N:
            total_mismatch += 1
    ok = (worst_lo == 0 and worst_hi == 0 and total_mismatch == 0)
    _record("PF systematic resample: count_i in [floor(N p_i), ceil(N p_i)], sum==N",
            "0 violations, all partitions sum to N",
            f"below-floor={worst_lo}, above-ceil={worst_hi}, bad-partitions={total_mismatch}",
            ok)


# ===========================================================================
# 8. Particle filter: determinism under a fixed seed + valid beat probabilities
# ===========================================================================

def _tiny_pf_model():
    torch.manual_seed(7)
    model = SVTModel(hidden_dim=16, nhead=2, num_layers=1, num_meter_classes=3,
                     input_dim=2, audio_emission=True)
    model.eval()
    return model


def test_pf_determinism():
    model = _tiny_pf_model()
    acts = torch.randn(1, 12, 2)

    torch.manual_seed(999)
    out1 = model.sample_from_prior_pf(acts, n_particles=48, temperature=0.1)
    torch.manual_seed(999)
    out2 = model.sample_from_prior_pf(acts, n_particles=48, temperature=0.1)

    diffs = {k: (out1[k] - out2[k]).abs().max().item()
             for k in ("phase", "log_tempo", "beat_activation", "beat_logits")}
    max_diff = max(diffs.values())
    _record("PF sample_from_prior_pf is DETERMINISTic under a fixed seed",
            "0.0 (bit-identical re-run)", f"max|diff| over outputs = {max_diff:.2e}",
            max_diff == 0.0)


def test_pf_beat_activation_is_probability():
    model = _tiny_pf_model()
    acts = torch.randn(1, 16, 2)
    torch.manual_seed(123)
    out = model.sample_from_prior_pf(acts, n_particles=64, temperature=0.1)
    ba = out["beat_activation"]
    lo, hi = ba.min().item(), ba.max().item()
    _record("PF beat_activation is a valid weighted probability in [0,1]",
            "0 <= beat_activation <= 1", f"[min={lo:.4f}, max={hi:.4f}]",
            lo >= -1e-9 and hi <= 1.0 + 1e-9)


# ===========================================================================
# main
# ===========================================================================

def main():
    torch.set_grad_enabled(True)
    tests = [
        test_dense_forward_vs_bruteforce,
        test_dense_forward_timevarying,
        test_dense_forward_forbidden_transitions,
        test_autograd_posterior_marginals,
        test_shift_invariance,
        test_dense_viterbi_vs_bruteforce,
        test_structured_forward_vs_dense_and_bruteforce,
        test_structured_forward_forbidden_tempo_jumps,
        test_structured_viterbi_vs_bruteforce,
        test_structured_compact_emission_gather,
        test_edge_T1,
        test_edge_single_state,
        test_systematic_resample_count_invariant,
        test_pf_determinism,
        test_pf_beat_activation_is_probability,
    ]
    errors = []
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors.append((t.__name__, repr(e)))
            _record(f"{t.__name__} (RAISED)", "no exception", repr(e), False)
        print()

    n_pass = sum(1 for _, _, _, v in _RESULTS if v == "PASS")
    n_fail = len(_RESULTS) - n_pass
    print("=" * 70)
    print(f"SUMMARY: {n_pass}/{len(_RESULTS)} properties PASS, {n_fail} FAIL")
    for name, _, _, v in _RESULTS:
        if v == "FAIL":
            print(f"   FAIL: {name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
