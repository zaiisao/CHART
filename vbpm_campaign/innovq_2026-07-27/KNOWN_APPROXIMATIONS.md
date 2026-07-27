# KNOWN APPROXIMATIONS — deliberate exactness/speed trades (REVISIT BEFORE PUBLICATION)

## A1. Vectorized rollout (`rollout_vec.py`) — NOT bit-exact vs `innovq.rollout`
**Status:** accepted 2026-07-27 as a deliberate trade; user directive: "sacrifice of exactness
for speed ... we will have to revisit it later but right now it seems to be a smaller concern."

**What it is.** `innovq.rollout` unrolls a 1500-step Python loop (~2.6 s per 3x1500 rollout,
GPU at 7% — launch-bound). `rollout_vec` computes the same trajectory in parallel-in-time via
Picard iteration (guess trajectory -> all innovation heads at once -> rebuild by cumsum),
converging in 2 passes. Speedup ~1000x (2616 ms -> 2-3 ms).

**The discrepancy (measured, 3x1500 crops, trained ckpt innovq_pf_sm101_s0):**
| picard | max abs dphi (rad) | max abs dlt |
|---|---|---|
| 1 | 2.86e-03 | 7.84e-05 |
| 2 | 1.094e-03 | 1.86e-05 |
| 3 | 1.096e-03 | 1.86e-05 |
| 5 | 1.096e-03 | 1.86e-05 |
Converged by pass 2 (3 and 5 identical to 4 s.f.) -> the iteration IS contractive; the residual
is a genuine fixed-point difference, NOT incomplete convergence.

**Scale of the error.** 1.1e-03 rad = 0.02% of a bar ~= 0.3 ms of beat time, i.e. ~230x inside
the +-70 ms scoring window and ~15x smaller than the innovations themselves (s_phi = 0.05).

**Diagnosis so far (two hypotheses REFUTED):**
- NOT the meter carry: loop meter vs constant-meter0 differ by exactly 0.000e+00.
- NOT float32 accumulation: forcing float64 cumsum changed nothing (1.105e-03 -> 1.096e-03).
- Narrowed to: the `lt` (log-tempo) fixed point. max|dlt| = 1.86e-05; phase accumulates
  exp(lt) over T steps, so 1.86e-05 * 0.07 * 1500 ~= 2e-03 -- matches the observed phase residual.
  The Picard fixed point for lt settles slightly differently from the sequential solution
  (innov_head reads z_{t-1}, so lt feeds back into its own increments).
- Error profile: exactly 0 at t=0, 1.4e-06 at t=1, 5.8e-05 at t=100, 1.10e-03 at t=1499 --
  monotone growth with t, consistent with a small per-step bias compounding.

**Restrictions of the current implementation:** deterministic path only (`sample=False`),
`@torch.no_grad()`; meter collapsed to meter0 (valid ONLY because deterministic draws are
identical -- verified 0.000e+00, but this assumption breaks under `sample=True`).

**DECISION:** NOT wired into any training path as of 2026-07-27. Safe uses: eval probes,
diagnostics, sweeps where 1e-3 rad is immaterial. Unsafe until certified: anything whose
conclusion could turn on <1e-3 rad, and anything stochastic.

**TO REVISIT (in order):**
1. Finish the lt trace: is the Picard fixed point provably the sequential solution? If the
   innov head's dependence on z_{t-1} is weak (trained |mu| ~ 1e-4), quantify the induced
   fixed-point gap analytically rather than empirically.
2. Extend to `sample=True` (needs the crossing-carry cummax gather for meter, and shared
   noise draws to compare against the loop).
3. Gradient equivalence: BPTT through cumsum vs through the loop -- compare parameter grads,
   not just trajectories. THIS IS THE ONE THAT MATTERS for training use.
4. Only then wire into placement/handover, with the loop retained as the reference path
   behind a `--exact` flag for periodic re-verification.

**Why it is acceptable for now:** the current bottleneck is scientific throughput (each 30 s-crop
cell costs ~2 h on the loop; the same sweep is minutes vectorized), and the open question --
does long-horizon training keep the latent aligned -- is not decided at the 1e-3 rad level.
