# SPEC — Physical-prior anchoring (tutorial §6.8.11) for VBPM

Status: **DESIGN ONLY — nothing trained.** Two design-time *measurements* were made (labels-only
fit on the train fold, and a read-out of the existing checkpoint's prior kernel); they change the
design materially and are reported in §0. Code written so far:
`vbpm_anchor/fit_phys.py` (fits p_physical), `vbpm_anchor/diag_prior.py` (reads p_psi),
`vbpm_anchor/phys_params.json`. Nothing in `vbpm/`, `vbpm_fix/`, `vbpm_arms/`, `vbpm_final/` touched.

---

## 0. TWO MEASUREMENTS THAT DECIDE THE DESIGN

### 0.1 What the physical law actually is (train fold, labels only, 147 songs, 544 497 frames)

`fit_phys.py`, on the beat-linear oracle bar phase, inside the annotated span:

| quantity | value |
|---|---|
| phase residual around `phi_{t-1}+phidot_{t-1}` | median\|r\| 5.9e-4 rad, IQR 1.18e-3, **Cauchy MLE gamma = 5.55e-4** -> rho = 0.999445 |
| log bar-advance increment (slow level) | sd **1.25e-3**, half-IQR 1.7e-4 |
| log bar-advance level per meter | m=2: -2.487+/-0.228 (n=8); m=3: -2.543+/-0.339 (n=20); m=4: -2.794+/-0.239 (n=119); pooled range [-3.40, -2.06] |
| fraction of NEGATIVE true phase increments | **0.0000** |

Real music is an almost noiseless pointer: the per-frame phase innovation is ~1% of one frame's advance.

### 0.2 What the learned prior p_psi actually is (checkpoint `vbpm_arms/arm_i_ii_bern.pt`, 30 eval songs)

`diag_prior.py` reads the *same* checkpoint the decisive 2x2 (`vbpm_final/torch_pf.py`) used:

| prior head | learned value | physical value | ratio |
|---|---|---|---|
| phase concentration rho | **0.00999** (gamma = 4.62 rad) | 0.999445 (gamma = 5.55e-4) | gamma 8300x too large |
| sigma_level (per frame) | **0.777 nats** | 1.25e-3 | 620x |
| sigma_dev | 0.153 | ~0.01 | 15x |
| a_dev (AR(1) coef) | 0.013 | — | AR structure dead (white noise) |
| a_lv (level OU) | 0.838 | ~0.999 | stationary level sd = 1.42 nats = **4.1x tempo spread** |
| per-song SD of mean rho | 3.2e-4 on a mean of 1.0e-2 (**3 %**) | — | prior is essentially song-INDEPENDENT already |

**The measured frac_neg is explained exactly, analytically.** For a wrapped-Cauchy step of scale
gamma on top of a deterministic advance phidot, P(step < 0) = (1/pi) atan(gamma/phidot). At the
learned gamma = 4.62 and phidot = 0.0626 this is **0.4957** — the reported frac_neg is 0.496/0.503.
The learned phase transition is *numerically indistinguishable from a uniform draw on the circle*.

Two consequences that the rest of this spec is built around:

1. **The failure is NOT in the mean.** In `variant_b.elbo_b` / `e3_model.elbo_e3` the phase prior
   mean is *already* the physical law (`p_ph_mu = (phi_prev + exp(log_tempo_prev)) % 2pi`, line 164 /
   208). Anchoring the MEAN is a no-op. **Everything is in the dispersions** (rho, sigma_level,
   sigma_dev) and in the meter matrix. See §5(iii).
2. **The links cannot reach the physical values in 1200 steps.** rho = sigmoid(x): x(0.0100) = -4.60,
   x(0.999445) = +7.50 -> **12.1 units of pre-activation travel needed**; sigma = softplus(x)+1e-3:
   x(0.777) = +0.16, x(0.00125) = -8.29 -> **8.5 units**. AdamW at lr 3e-4 for 1200 steps moves a
   parameter by at most ~0.36 units (bias) / a few units (weights x tanh-bounded ctx). *A naive
   implementation of the anchoring term will therefore produce NO EFFECT at every lambda, and that
   result would be an artefact of the link function, not of the idea.* This forces §3.2.

---

## 1. THE OBJECTIVE

### 1.1 Notation (exactly the tensors in `vbpm_final/e3_model.py::elbo_e3`)

At step t the generative prior is the product of four factors, conditioned on the previous latent
state z_{t-1} = (phi_{t-1}, level_{t-1}, dev_{t-1}, m_{t-1}) and on `prior_ctx[:, t] = f_psi(x)_t`:

```
p_psi(phi_t   | z_{t-1}, x) = WC( mu = (phi_{t-1} + exp(lt_{t-1})) mod 2pi , rho_psi(ctx_t) )
p_psi(level_t | z_{t-1}, x) = StudentT( nu , mu = anchor + a_lv (level_{t-1}-anchor) , s_lv(ctx_t) )
p_psi(dev_t   | z_{t-1}, x) = N( mu = a(ctx_t) dev_{t-1} , s_dv(ctx_t) )
p_psi(m_t     | z_{t-1}, x) = Cat( pi_psi(m_{t-1}, phi_t, phi_{t-1}, ctx_t) ),  used only when cross_t=1
```

### 1.2 The physical prior p_physical (same conditioning, NO x)

```
p_phy(phi_t   | z_{t-1}) = WC( mu = (phi_{t-1} + exp(lt_{t-1})) mod 2pi , rho_phy )          # SAME mean
p_phy(level_t | z_{t-1}) = StudentT( nu , mu = mu_lt(m_{t-1}) + a_phy (level_{t-1}-mu_lt(m_{t-1})) , s_lv_phy )
p_phy(dev_t   | z_{t-1}) = N( 0 , s_dv_phy )                                                  # no fast deviation
p_phy(m_t     | m_{t-1}) = (1-p_switch) delta_{m_{t-1}} + p_switch * Cat(meter_prior)         # sticky
```

### 1.3 The regularised objective

```
L_reg-EB = ELBO(theta, phi_enc, psi)  -  lambda_prior * A(psi)

A(psi) = E_q [ sum_{t=1..T} (  KL( p_psi(phi_t | z_{t-1},x)   || p_phy(phi_t | z_{t-1}) )
                             + KL( p_psi(level_t | z_{t-1},x) || p_phy(level_t | z_{t-1}) )
                             + KL( p_psi(dev_t | z_{t-1},x)   || p_phy(dev_t | z_{t-1}) )
                             + cross_t * KL( p_psi(m_t | .)   || p_phy(m_t | m_{t-1}) ) ) ]
```

i.e. in code `loss = recon_b + recon_db + obs_w*recon_obs + beta*L_kl + lam*L_anchor` (all terms
SUMMED over T, averaged over B, exactly like the existing KLs, so lambda is comparable to beta).

The expectation over q is taken with the **single ELBO sample already drawn** (z_{t-1} from q) —
no extra sampling — and the conditioning state is **detached** (§4.3).

### 1.4 Closed forms (all already in `vbpm/distributions.py`)

* **phase** — `kl_wrapped_cauchy(p_ph_mu, rho_psi, p_ph_mu, rho_phy)`; means identical, so it
  collapses to the pure concentration divergence
  `log[ (1 - rho_psi*rho_phy)^2 / ((1-rho_psi^2)(1-rho_phy^2)) ]`.
  Magnitudes: rho_psi=0.010 -> **6.78 nats/frame**; 0.90 -> 3.87; 0.98 -> 2.26; ->0 as rho_psi->rho_phy.
* **level** — Student-t/Student-t with equal nu has no closed form. Use the **Gaussian
  location-scale surrogate** `kl_log_normal(mu_psi, s_lv, mu_phy, s_lv_phy)` (exact as nu->inf;
  it is a penalty on the prior's *parameters*, which is all we need). Optional exactness check:
  `kl_student_t_mc(dof, mu_psi, s_lv, mu_phy, s_lv_phy, z)` with **z drawn from p_psi by rsample**
  — never with the q-sample, which would not estimate KL(p_psi||p_phy).
* **dev** — both Gaussian: `kl_log_normal(a*dev_prev, s_dv, 0, s_dv_phy)` exactly.
* **meter** — `kl_categorical(log_pi_psi, log_pi_phy)`, gated by `cross_t` exactly as `kl_m` is.

---

## 2. WHERE p_physical's PARAMETERS COME FROM

**Fitted ONCE on the TRAIN fold; never on eval.** `vbpm_anchor/phys_params.json` already holds the
fit (§0.1). Two candidate targets, both train-derived; the choice is a hyperparameter selected on a
held-out *train* fold (§3.3):

| parameter | **P-FIT** (fitted physics, §0.1) | **P-DEPLOY** (the hand transition that scores 0.751) |
|---|---|---|
| rho_phy | 0.999445 (gamma 5.55e-4) | 0.9802 (gamma 0.02, matches `pf.py --sigma_phi 0.03`) |
| s_lv_phy | 1.25e-3 | 0.05 (`pf.py --sigma_lt 0.05`) |
| s_dv_phy | 1e-3 (head floor; jitter carried by the phase kernel) | 1e-3 |
| mu_lt(m), a_phy | per-meter means -2.487/-2.543/-2.794, a_phy = 0.999 | same, plus hard band clamp `pf.lt_band(m)` |
| p_switch, meter_prior | 0.005 per bar crossing (= `alt_meter_model` fitted 0.0052/bar), train meter histogram | same |

**Recommendation: P-FIT is the default.** Reason (quantitative): the frac_neg floor under a
wrapped-**Cauchy** kernel is (1/pi)atan(gamma/phidot) = **0.098 at gamma = 0.02** but **0.0028 at
gamma = 5.55e-4**. The hand PF reaches frac_neg 0.012 with sigma_phi = 0.03 only because its noise is
*Gaussian* (Phi(-0.0626/0.03) = 0.0185). Matching the hand transition's *scale* under the heavy-tailed
kernel would reproduce ~10 % backward steps and miss the mechanism criterion. P-DEPLOY is kept as the
second sweep arm precisely to test whether the PF needs the looser proposal for tracking.
`mu_lt(m)` and the per-meter band are indexed through the model's own meter one-hot as a soft mixture
`mu_lt = sum_j m_j * mu_lt(meter_j)`; class j maps to meter value `j + meter_offset`.

---

## 3. lambda_prior: PARAMETERISATION, SWEEP, SELECTION

### 3.1 Sweep range
Anchor magnitude at the current solution is 6.8 nats/frame (phase) versus an ELBO phase-KL of
~0.2-0.3 nats/frame (kl_phi 45-84 per 256-frame crop). So lambda = 1 already makes the anchor
dominant. Sweep **lambda in {0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, inf}** (9 cells; log-spaced).
`lambda = 0` is the unanchored control (must reproduce the 0.353/0.592 cells);
`lambda = inf` is implemented as "residual frozen at 0", i.e. p_psi == p_physical exactly, and is the
**decisive control** (see §5(i)): the experiment's real question is whether any finite lambda beats it.

### 3.2 MANDATORY companion: anchored-residual parameterisation
Because of §0.2(2) the anchor cannot be reached through the existing links. Replace the *links only*
(same distribution family, same architecture, no new inputs):

```
gamma_psi(ctx) = gamma_phy * exp( v_max * tanh( g_head(ctx) ) )      # phase; rho = exp(-gamma)
s_lv(ctx)      = s_lv_phy  * exp( v_max * tanh( l_head(ctx) ) )
s_dv(ctx)      = s_dv_phy  * exp( v_max * tanh( d_head(ctx) ) )
pi_psi         = softmax( log pi_phy + v_max * tanh( m_head(ctx) ) ) # meter, logit residual
```
with `v_max = 3` (+/- 20x around physics) and the **final layer of each head zero-initialised**, so
training *starts exactly at p_physical* and the ELBO must pay to move away. In this space the anchor
is a smooth shrinkage penalty on the residual (e.g. level: KL = -v + (e^{2v}-1)/2, which is
well-conditioned near v = 0), instead of a 1.9e5-nat cliff at the current initialisation, and
lambda -> inf is literally "freeze the residual". Report the fraction of frames with |tanh(.)| > 0.99
(saturation = v_max too small).
*This is a link/initialisation change, not a model-family change; it must be applied identically to the
lambda = 0 control so the sweep is internally comparable. The lambda = 0 control under the new link
must be checked against the archived 0.353/0.592 cells (§6.1).*

### 3.3 Selection without touching eval
Eval = fold 0 (79 songs) and is touched **once**, at the end. Train = folds 1-7 (147 songs;
counts 22/18/20/21/22/21/23). **Hold out fold 1 (22 songs) as the dev split**: train on folds 2-7
(125 songs), deploy + score on fold 1 with the identical PF + blind-control harness. Pick
`lambda*` by **margin over the density-matched blind control** on fold 1 (not raw beat_F).
Then retrain on all 147 with `lambda*` and report fold 0 once. Publish the whole fold-1 curve
(F, margin, frac_neg, n_ratio, ESS vs lambda) so the window — or its absence — is visible.

---

## 4. EXACTLY WHERE IT ENTERS

### 4.1 Files (all NEW, under `vbpm_anchor/`)
* `anchor_model.py` — `AnchoredVAE(E3VAE)`: overrides `prior_phase_conc`, `prior_level_scale`,
  `prior_dev_scale`, `meter_prior_logp` with the §3.2 residual links; holds the p_phy buffers.
* `anchor_elbo.py` — `elbo_anchor(...)`: **verbatim copy** of `e3_model.py::elbo_e3` plus the
  accumulator; returns `(loss, info)` with `info["anchor_*"]` per factor.
* `anchor_run.py` — driver (train + deploy + score), reusing `run_exp2.py`'s controls verbatim.
* `anchor_cells.py` — re-runs the reference cells in the same harness (§6.1).

### 4.2 The insertion, line by line (against `e3_model.py::elbo_e3`)
* after L179-180 (`p_dv_mu`, `p_dv_s` at t = 0) and after **L209-214** (`p_ph_rho`, `a`, `p_lv_mu`,
  `p_lv_s`, `p_dv_mu`, `p_dv_s`) and **L223** (`log_pi_p`): accumulate
  `L_anc = L_anc + kl_wrapped_cauchy(p_ph_mu_d, p_ph_rho, p_ph_mu_d, rho_phy)`
  `        + kl_log_normal(p_lv_mu_d, p_lv_s, p_lv_mu_phy_d, s_lv_phy)`
  `        + kl_log_normal(p_dv_mu, p_dv_s, zeros, s_dv_phy)`
  `        + cross * kl_categorical(log_pi_p, log_pi_phy_d)`  (shapes [B], same as `kl_p`).
* L245: `loss = (recon_b + recon_db + obs_w*recon_obs + beta*L_kl + lam*L_anc).mean()`.
* `info` gains `anchor_phase/level/dev/meter` (per frame) and `rho_psi_mean`, `s_lv_mean`.

### 4.3 Gradient hygiene (non-negotiable)
* Every q-sampled tensor entering the anchor (`phi_prev`, `log_tempo_prev`, `level_prev`,
  `dev_prev`, `meter_prev`, `anchor`, and the derived `p_ph_mu`, `p_lv_mu`) is **`.detach()`-ed** in
  the anchor term. The anchor is a regulariser on psi ONLY; it must not become a back-door
  regulariser on the encoder. (The mean terms then cancel identically to 0 for the phase factor —
  that is the correct, intended behaviour, see §5(iii).)
* Verify with the existing gradient audit (`e3_vae.py` L198-202 pattern): with `lam > 0, beta = 0`,
  `post_head`/`decoder` grad-norm from `L_anc` alone must be exactly 0, and
  `prior_phase_rho`/`prior_level_sigma` must be non-zero. Run this as a unit assertion before any sweep.

### 4.4 What must NOT change
ELBO terms and their weights; the beta warm-up (`min(1, step/600)`) and Gumbel temperature schedule;
the posterior/encoder architecture; the emission (whichever cell's emission, unchanged);
`beats_from_barphase` / `downbeats_from_barphase` / `f_measure` read-out; the blind controls
(copied verbatim from `run_exp2.py`); the fold split; steps/bs/frames/lr/K/alpha; seeds.
The single manipulated variable is `lambda_prior` (plus the §3.2 link, held fixed across the sweep).

---

## 5. FAILURE MODES AND THE GUARDS

**(i) lambda too large -> the learned prior is decorative.** Guards, all reported:
  a. `KL(p_psi||p_phy)` per frame at convergence (the anchor value itself), per factor.
  b. **Per-song adaptation**: SD across eval songs of the song-mean rho_psi, s_lv, and of the
     meter row entropy. Baseline for "no adaptation" is already measured: the *unanchored* prior has
     per-song SD of rho = 3.2e-4 on a mean of 1.0e-2 (3 %). If the anchored model is not clearly
     above that, "learned" is decorative.
  c. **Freeze ablation**: replace ctx-dependent prior params by their global (dataset-mean) values at
     deploy. If beat_F drops < 0.01, the audio-conditioning contributes nothing.
  d. **The lambda = inf cell** (residual frozen at 0 == p_physical exactly). *A finite-lambda win must
     exceed lambda = inf, not merely lambda = 0.* Report `F(lambda*) - F(inf)` with the same seeds.

**(ii) lambda too small -> no effect.** Detect by the anchor value itself: if
`KL(p_psi||p_phy)` at convergence is within 10 % of its lambda = 0 value, the cell is a null.
Also guard the §0.2(2) artefact: **before the sweep**, verify that `lambda = 3` actually moves
rho_psi to within 2x of rho_phy in 1200 steps. If it does not, the link (not the idea) is the blocker.

**(iii) mean-anchoring vs concentration-anchoring.** *Concentration is the one that matters.* The
phase prior mean is already exactly `phi_{t-1} + phidot_{t-1}` (`elbo_e3` L206-208), so the mean term
of the phase anchor is identically 0 and contributes no gradient; monotonicity is governed entirely by
gamma_psi/phidot. Same for the level: the mean is the OU recursion, and only `s_lv` (0.777 vs 1.25e-3)
makes the deploy tempo wander by 4.1x — which is what produces `n_ratio` 1.77-2.61 in the learned
cells. Anchoring the *means* would be a no-op; anchoring the *dispersions* (and `a_lv`, `a_dev`,
`pi_psi`) is the whole intervention.

**(iv) the conflict moves to the posterior.** A sharp prior makes `KL(q||p_psi)` expensive; the
optimizer may sharpen q (wanted) or kill the phase latent (the known collapse). Monitor per-frame
`kl_phase` and `R(phi_q, phi_true)` (already logged by `e3_vae.py` L207-208) at every lambda; a
collapse shows as R -> 0 with kl_phase -> 0. Report the R(lambda) curve next to the F(lambda) curve.

**(v) ESS / particle depletion at deploy.** A prior sharper than the hand transition narrows the
bootstrap proposal. Report ESS at every lambda; if ESS collapses while F drops, that is a *filtering*
failure, not a modelling one, and is the case P-DEPLOY (§2) exists to separate.

---

## 6. HARNESS AND THE MANDATORY EVALUATION RULE

### 6.1 Cells to run (all in ONE harness, all with the same read-out and controls)
| cell | transition | emission | note |
|---|---|---|---|
| R1 | learned, unanchored (lambda = 0) | supervised (swap at deploy) | must reproduce ~0.59 |
| R2 | hand (`pf.py` / `torch_pf.simple_pf`) | supervised | the 0.751 target |
| R3 | act-head peak-pick | — | 0.812 / 0.534 stretch bar |
| R4 | oracle true bar phase | — | 0.960 ceiling |
| **A** | **learned + anchored**, lambda sweep | supervised (swap at deploy) | PRIMARY: single change vs R1 |
| B | learned + anchored | frozen supervised inside the ELBO too (E3 config) | secondary arm |
| C | lambda = inf (== p_physical, learned residual frozen) | supervised | decorative-prior control |

Arm A is the criterion cell because it differs from the archived 0.592 cell by **one** term.
Arm B is run in parallel (it is the better model, but two changes).

### 6.2 Two harness asymmetries that must be fixed first (they currently bias the comparison)
1. `torch_pf.py` L134 sets `phase_path = phase_map` for the **learned** transition (no ancestral
   backtrace), while the **hand** transition gets a genuine ancestral path. Per-frame argmax inflates
   frac_neg and jitter. **Use `vbpm_final/e3_pf_learned.py::particle_filter_learned` (which has the
   backtrace) for every learned cell**, and report the `path` read-out for both sides.
2. The 2x2 checkpoint has `num_meters=4` with a degenerate class (meter value 1); `meter_acc = 0.00`
   for the learned cells. Keep the architecture identical for A vs R1 (so the comparison is clean),
   but report `meter_acc` for every cell; the anchored sticky meter transition is expected to repair it.

### 6.3 Reporting (mandatory for EVERY number)
Reuse `run_exp2.py::blind_grid_controls` **verbatim**. Every beat_F line reports:
`beat_F, downbeat_F, n_est/n_true (n_ratio), n_ratio_db, blind0 (same density), blind_best (12
offsets), MARGIN = beat_F - blind_best, db MARGIN, frac_neg, jitter/advance, ESS, obs_contrast,
meter_acc, n_songs`. An uncontrolled beat_F is not evidence. Success is judged on the **margin**.

### 6.4 Success criteria restated
* **PRIMARY**: arm A at lambda* matches R2 (hand transition, same emission) in beat_F **and** margin,
  with **frac_neg <= ~0.03** (analytic floor at P-FIT is 0.0028) and **n_ratio in [0.9, 1.3]**.
* **STRETCH**: beat R3 (0.812 beats / 0.534 downbeats) on margin-controlled F.
* **FAILURE**: if the best lambda cannot match R2, say so plainly, and report which of §5(i)-(v)
  the diagnostics point to.
* **Additional honesty clause**: even if PRIMARY passes, report `F(lambda*) - F(lambda = inf)`. If that
  is within noise, the correct conclusion is "anchoring stops variational training from being harmful
  by making the prior physical, and the *learned* part earns nothing" — not "anchoring works".

---

## 7. THE MONOTONICITY QUESTION

**Does a KL anchor enforce monotone phase advance? Partly — statistically, never structurally.**

Under the wrapped-Cauchy kernel the backward-step probability is exactly
`P(step < 0) = (1/pi) atan(gamma / phidot)`:

| gamma | rho | frac_neg floor |
|---|---|---|
| 4.62 (learned, measured) | 0.010 | **0.4957** (matches the observed 0.496/0.503) |
| 0.02 (P-DEPLOY) | 0.980 | 0.098 |
| 5.55e-4 (P-FIT) | 0.999445 | **0.0028** |

So anchoring to **P-FIT** does clear the mechanism criterion (0.0028 << 0.01 target), and a hard
constraint is **not required** to pass. Two caveats:
* the floor is not zero and never can be: a wrapped Cauchy has full circular support, so at any finite
  lambda a fraction of steps runs backwards, and the heavy tail also gives
  `P(|jump| > pi) = (2/pi)atan(gamma/pi) = 1.1e-4/frame` at P-FIT — about one phase reset per
  9 000 frames (3 min at 50 fps).
* it forces a trade-off the family cannot resolve: the PF wants proposal diversity (the hand PF
  deliberately uses sigma_phi = 0.03, 54x the fitted physics) but Cauchy diversity at that scale costs
  10 % backward steps.

**Therefore specify DEVIATION D2 as a pre-registered second arm (not a fallback excuse):**
positive-support increment parameterisation
```
phi_t = phi_{t-1} + Delta_t (mod 2pi),   Delta_t ~ LogNormal( log phidot_{t-1} , s_phi )
```
frac_neg is **identically 0** by construction, at any dispersion; it is reparameterizable
(`exp(mu + s*eps)`) and its KL to the physical LogNormal is the closed-form Gaussian KL in
log-increment space (`kl_log_normal`), so nothing about the ELBO machinery changes.

*Justification from the tutorial's own stance*: §6.8.11's physical prior is a claim about the **state
law**; "the pointer never runs backwards" is a **support** property, and a KL penalty between two
full-support densities can only *price* support violations, never forbid them — no finite lambda makes
frac_neg 0. Where the physics is a support constraint, the faithful implementation of "physical prior"
is the constrained parameterisation, and the KL anchor then governs only the *scale* of the (positive)
increment. D2 also *removes* reality-adjusted fix #3 (heavy circular tail, wrapped Cauchy), which was
justified by a within-pulse residual likelihood, so it must be reported as a deviation with its own
A/B, never silently merged into arm A.

---

## 8. PREDICTED FAILURE MODE (pre-registered)

1. **The mechanism will work.** frac_neg 0.50 -> ~0.003-0.03; n_ratio 1.77-2.61 -> ~1.0-1.3; ESS up.
   PRIMARY (match ~0.751) is likely met or nearly met, because the entire deficit is dispersion and
   the mean is already physical.
2. **But the learned prior will be decorative.** The unanchored prior already has only 3 % per-song
   variation in rho; the gain comes from restoring dispersion to physics, not from conditioning on x.
   Prediction: at lambda*, `KL(p_psi||p_phy) < ~0.05 nats/frame`, the freeze ablation costs < 0.01 F,
   and **F(lambda*) - F(lambda = inf) is within seed noise**. Expected verdict: *"variational training
   stops being harmful, and stops being load-bearing"* — consistent with the tutorial's own concession
   that x-only inference cannot close the aggregated-posterior gap by fitting psi.
3. **STRETCH will not be met.** The 2x2 says the binding constraint after the transition is fixed is
   the **emission** (learned obs_contrast 0.9998 vs supervised 5.78; peak-pick 0.812 uses the same
   activation more sharply). A better transition cannot cross 0.812 on beats; downbeats
   (0.534 bar) are the plausible win, since the hand cell already reaches db_F 0.52 vs blind 0.24.
4. **The most likely way to get a false NULL** is §0.2(2): implementing the anchor without §3.2 and
   observing no movement at any lambda. If the pre-flight check in §5(ii) is skipped, this experiment
   reports NO_EFFECT for a link-function reason. Guard it first.
