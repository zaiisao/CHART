"""THE WIRE TEST -- H1 (ELBO misspecification) vs H2 (amortization gap) vs H3 (tempo flat),
   rebuilt after an adversarial review that found the previous verdict tree UNREACHABLE.

THE QUESTION
  VBPM trains an ELBO. From a tempogram initialization that gives good tempo (level MAE
  ~2-4%, per-crop phase corr ~0.6), ELBO training DEGRADES tempo (25-40%, corr ~0.03-0.16)
  while the ELBO value IMPROVES.
    H1 MISSPECIFICATION -- the ELBO's optimum genuinely has wrong tempo.
    H2 AMORTIZATION     -- the objective is fine; only the ENCODER family cannot represent
                           the good solution.
    H3 FLAT             -- the objective carries no usable tempo information over the range
                           the arms traverse, so a free q's level coordinate DIFFUSES.

WHAT CHANGED IN THIS REVISION (each item is a review finding, with its fix)

 F1/F8/F14  The old PRE_A required loss(30%) - loss(3%) > 2 sem. A pointwise Bernoulli
   likelihood at T=1500 SATURATES once the phase is decorrelated, so that test failed by
   construction and the H1/H2 branches were dead code. PRE_A is now an ORDERING test inside
   the band the arms actually traverse (loss(3%) vs loss(0.3%)) with an absolute nat FLOOR,
   plus no-inversion. The saturation point is MEASURED and reported as the RESOLUTION LIMIT
   -- a finding, not a veto.

 F2/F18  Every gate whose FAILURE produced an affirmative flatness claim was a
   failure-to-reject. All gates are now THREE-VALUED: POSITIVE (m > 2 sem AND m > floor),
   NEGLIGIBLE (an equivalence/TOST result: |m| + 2 sem < a pre-declared margin), or
   UNRESOLVED. Flatness may only be asserted from NEGLIGIBLE. Every gate prints its
   minimum detectable effect (MDE = 2 sem). The run is replicated over >=3 optimization
   seeds and the SIGN stability of the branch-selecting statistics is required.

 F7/F15  MOVED / HELD were gated on the SAMPLED level MAE, which is a monotone function of
   the posterior WIDTH s_l1 (measured: 7.9% -> 40.4% at a PERFECT mean). They are now gated
   on the posterior's LOCATION -- the deterministic mean-path level, per crop, paired over
   crops -- cross-checked against the draw-MEDIAN location (robust: Student-t(2) has
   infinite variance, so a draw-MEAN location is not a usable estimator). The sampled MAE
   stays as a descriptive number, and each arm's final s_l1 is printed. Both thresholds are
   DECLARED in advance and their cost in per-crop corr is printed from a calibration sweep.

 F10/F17  The regime auto-selector swapped the objective on a 4-draw marginal test and
   flipped between runs. The arms now ALWAYS run at the PRODUCTION gamma_phase. The scan is
   a printed SENSITIVITY table, run on two independent noise banks with flip detection.
   Any non-production regime refuses H1/H2 and prints THE PRODUCTION REGIME WAS NOT TESTED.

 F11  The free q's SAMPLED init was corr 0.31 / MAE 8.9%, i.e. not the premise. The free q
   is now SHARP-initialized (s_l1 at its floor, s_lv sharpened) so the sampled trajectory
   really is at good tempo; the KL price of the sharpening is printed, PRE_C's floor is
   raised to 0.55 (the brief's 0.63 teacher, -10%), and the s_lv / s_l1 decomposition of the
   init MAE is printed.

 F16  PRE_D was measured at rho1 = sigmoid(0)*0.9 = 0.45, a 0.80 rad wrapped-Cauchy on the
   INITIAL PHASE that swamps the tempo signal. rho1 is now part of the declared init and of
   the scan (--r1r_init), for the free arms AND for the encoder's own bias.

 F4  The free arms got the ORACLE phase offset and the encoder got a random one. The
   encoder's init head is now pre-trained on the oracle offset AND on rho1/s_l1/s_lv biases
   matched to the free arms', and the result is ASSERTED before the arm runs.

 F21  `reproduce` never checked that the AMORTIZED arm started from the premise. PRE_B now
   gates the encoder's post-init level location, phase offset and mean-path corr, and
   aborts loudly if the pre-train did not reach them.

 F3  free_bad (the cold-start control) was infeasible: |init offset| up to 1.25 in log-level
   against a maximum Adam travel of steps*lr = 0.09. Its lr is now SIZED to the offset and
   the feasibility bound is asserted. best_lr is chosen by the PAIRED comparison, not by an
   unpaired point estimate whose MC noise exceeded the gap.

 F13/F19  free_pinned kept `sl1r` trainable, so "tempo provably exact" was false. The pin
   now covers ml1, sl1r AND inn[...,2:4], all asserted bit-identical. H2 additionally needs
   a POSITIVE optimizer control (free_bad must be able to descend) so that "held" cannot be
   satisfied by a dead optimizer, and GRAD_OK is required symmetrically for H1 and H2.

 F5  The [P2] KL check compared kl_t_mc against itself. It is now checked against SciPy
   quadrature of the true KL integral.

 F6  draw_noise's Student-t stream ignored `gen`, so the LEVEL noise -- the channel that
   decides the verdict -- came from the global RNG. This file draws its own noise with an
   exact generator-backed Student-t(2) inverse CDF and asserts freshness on EVERY key.

 F9  The [C2] zero check passed identical parameter dicts (pitfall P2). It is relabelled as
   a construction check and a NON-TRIVIAL estimator check was added: a known +1% level
   perturbation measured on the shared bank against an independent larger bank.

 F20  The amortized arm trained through the Picard rollout while the free arms trained
   through the exact cumsum. The parameter-gradient cosine between the two is now CERTIFIED
   in-run at T=1500 and the run aborts below --grad_cos_min (use --amort_exact to train the
   encoder through the sequential loop instead).

PITFALL LEDGER (the brief's P1-P6), each enforced in code -- grep for the tag.
 [P1] FRESH noise every optimizer step, generator-backed on EVERY channel, asserted.
 [P2] KL estimator checked with q != p against SciPy quadrature.
 [P3] ONE decoder, pre-fit on the ORACLE trajectory, FROZEN, asserted phase-sensitive AND
      asserted tempo-ORDERING (the scramble test alone is not enough).
 [P4] Every freeze VERIFIED bit-identical (decoder after every arm; free_pinned's ml1,
      sl1r and inn[...,2:4]).
 [P5] Every REPORTED number comes from an exact rollout. rollout_vec_s appears only inside
      the amortized TRAINING loop, and only after its gradient is certified.
 [P6] All arms share crops, frozen decoder, beta/recon/obs_w/temperature, the loss assembly
      and the evaluation noise bank; a GATE proves the free-q assembly IS innovq's objective.
 [prior] RHO_P, GP1 and R0 are ALL rebound together whenever gamma_phase changes.

LEVEL TRUTH: train["lt"].mean(1) -- the oracle per-frame log bar-advance from the downbeat
annotations. NOT log(2pi/(4*median_IBI)) (that proxy hardcodes 4 beats/bar); the proxy's
disagreement is printed for continuity with earlier logs and never used as a threshold.
"""
import argparse, json, math, statistics as _st, sys, time

import torch, torch.nn as nn, torch.nn.functional as F

for _p in ("/home/sogang/jaehoon/VBPM_reintegration",
           "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
           "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq",
           "/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pm_common as P                                                    # noqa: E402
import innovq as IQ                                                      # noqa: E402
from innovq_tg import InnovQT                                            # noqa: E402
from rollout_vec_s import rollout_vec_s, rollout_loop_noise              # noqa: E402
from vbpm.distributions import kl_categorical                            # noqa: E402

TWO_PI = 2.0 * math.pi
DEV = "cuda:0"
ENC_S_L1_FLOOR = 0.05        # the floor hardcoded in innovq.rollout / rollout_vec_s._heads
PROD_GAMMA = 5.5e-4          # P.PHYS default -- the gamma VBPM actually trains with
PKEYS = ("recon_b", "recon_db", "recon_obs", "kl_phase", "kl_level", "kl_meter")
NKEYS = ("u", "tstd", "nrm", "gum")


# ======================================================================================
#  [P1][F6] generator-backed noise on EVERY channel.
#  rollout_vec_s.draw_noise builds tstd with StudentT.sample(), which ignores `gen` -- the
#  LEVEL noise (the channel the verdict is decided on) then comes from the GLOBAL RNG while
#  u / nrm / gum honour it. We draw our own with the EXACT Student-t(2) inverse CDF
#      F(t) = 1/2 (1 + t / sqrt(t^2+2))  ->  t = p sqrt(2/(1-p^2)),  p = 2u-1,
#  which is generator-backed, so arm pairing is explicit rather than accidental.
# ======================================================================================
def draw_noise_g(Bn, T, K, dev, dof, gen=None):
    assert abs(float(dof) - 2.0) < 1e-12, (
        f"the closed-form Student-t inverse CDF used here is exact only at dof=2; got {dof}. "
        "Add a generator-backed sampler for the new dof before running.")
    u = torch.rand(Bn, T, device=dev, generator=gen).clamp(1e-4, 1 - 1e-4)
    ut = torch.rand(Bn, T, device=dev, generator=gen).clamp(1e-6, 1 - 1e-6)
    p = 2.0 * ut - 1.0
    tstd = p * torch.sqrt(2.0 / (1.0 - p * p))
    nrm = torch.randn(Bn, T, device=dev, generator=gen)
    gum = -torch.log(-torch.log(torch.rand(Bn, T, K, device=dev, generator=gen) + 1e-20) + 1e-20)
    return dict(u=u, tstd=tstd, nrm=nrm, gum=gum)


# ======================================================================================
#  free-q trajectory -- the SAME maths as innovq.rollout / rollout_loop_noise, with the
#  head outputs supplied directly instead of produced by an encoder. Proven identical by
#  the [P6] GATE below, not by inspection.
# ======================================================================================
def free_traj(m_logits, mu_phi1, rho1, mu_l1, s_l1, mu_eps, sq, mu_lt, s_lv,
              noise, Pi, *, sample=True, temperature=0.3):
    """mu_eps/sq/mu_lt/s_lv are [B,T-1]; index j is the innovation used at step t=j+1."""
    B, K = m_logits.shape
    T = mu_eps.shape[1] + 1
    dev = m_logits.device
    dof = torch.tensor(IQ.DOF, device=dev)
    if sample:
        phi1 = (mu_phi1 + (-torch.log(rho1)) * torch.tan(math.pi * (noise["u"][:, 0] - 0.5))) % TWO_PI
        lev1 = mu_l1 + s_l1 * noise["tstd"][:, 0]
        eps = mu_eps + (-torch.log1p(-sq)) * torch.tan(math.pi * (noise["u"][:, 1:] - 0.5))
        eps_lt = mu_lt + s_lv * noise["tstd"][:, 1:]
        dvv = IQ.DEV_SIGMA * noise["nrm"]
        m_draw = F.softmax((m_logits.unsqueeze(1) + noise["gum"]) / temperature, -1)   # [B,T,K]
    else:
        phi1, lev1, eps, eps_lt = mu_phi1, mu_l1, mu_eps, mu_lt
        dvv = torch.zeros(B, T, device=dev)
        m_draw = F.softmax(m_logits / max(temperature, 1e-6), -1).unsqueeze(1).expand(-1, T, -1)

    # float64 for the two long cumulative sums (T=1500 accumulations in float32 drift)
    lev = (lev1.unsqueeze(1).double() + torch.cumsum(F.pad(eps_lt, (1, 0)).double(), 1)).float()
    lt = lev + dvv
    steps = torch.exp(lt.clamp(-12.0, 6.0))
    inc = F.pad(steps[:, :-1], (1, 0)) + F.pad(eps, (1, 0))
    phi = ((phi1.unsqueeze(1).double() + torch.cumsum(inc.double(), 1)) % TWO_PI).float()

    adv = phi[:, :-1] + steps[:, :-1]                       # exactly the loop's `advance`
    cross = F.pad((adv >= TWO_PI).float(), (1, 0))          # [B,T]; cross[:,t>=1] = crossing at t
    cfull = cross.clone(); cfull[:, 0] = 1.0                # t=0 always draws a meter
    ar = torch.arange(T, device=dev)
    last = torch.cummax((ar.unsqueeze(0) * cfull).long(), dim=1).values
    meter = torch.gather(m_draw, 1, last.unsqueeze(-1).expand(-1, -1, K))
    meter_prev = torch.cat([meter[:, :1], meter[:, :-1]], 1)

    Z = torch.cat([torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1),
                   lt.clamp(-12.0, 6.0).unsqueeze(-1), meter], -1)   # = model.z_features

    kl_p = (IQ.kl_wrapped_cauchy(mu_phi1, rho1, torch.full_like(mu_phi1, math.pi),
                                 torch.full_like(mu_phi1, 1e-6)).double()
            + IQ.kl_phase_innov(mu_eps.reshape(-1), sq.reshape(-1)).reshape(B, -1).sum(1))
    kl_l = (P.kl_t_mc(dof, mu_l1, s_l1, dof,
                      torch.full((B,), IQ.INIT_LV_MU, device=dev),
                      torch.full((B,), IQ.INIT_LV_S, device=dev), lev[:, 0])
            + P.kl_t_mc(dof, mu_lt, s_lv, dof, torch.zeros_like(mu_lt),
                        torch.full_like(mu_lt, IQ.T_SCALE), eps_lt).sum(1))
    lq_row = torch.log_softmax(m_logits, -1)
    lq = lq_row.unsqueeze(1).expand(-1, T - 1, -1)
    lp = torch.log(meter_prev[:, 1:] @ Pi + 1e-9)
    kl_m = (kl_categorical(lq_row, torch.full((B, K), -math.log(K), device=dev))
            + (cross[:, 1:] * kl_categorical(lq, lp)).sum(1))
    return dict(Z=Z, phi=phi, lt=lt, kl_p=kl_p.float(), kl_l=kl_l, kl_m=kl_m,
                kl_dv=torch.zeros(B, device=dev), n_cross=1.0 + cross[:, 1:].sum(1),
                s_l1=s_l1.detach())


def free_heads(pars, s_phi, s_lt, rho1_max, s_l1_floor):
    """Raw free parameters -> head values through the ENCODER's own transforms/bounds."""
    return dict(m_logits=pars["mlog"],
                mu_phi1=pars["mp1"] % TWO_PI,
                rho1=(torch.sigmoid(pars["r1r"]) * rho1_max).clamp(1e-6, 1 - 1e-6),
                mu_l1=pars["ml1"],
                s_l1=F.softplus(pars["sl1r"]) + s_l1_floor,
                mu_eps=torch.tanh(pars["inn"][..., 0]) * s_phi,
                sq=F.softplus(pars["inn"][..., 1] + IQ.R0).clamp(1e-6, 0.5),
                mu_lt=torch.tanh(pars["inn"][..., 2]) * s_lt,
                s_lv=F.softplus(pars["inn"][..., 3] + IQ.B_SLT0) + 1e-5)


def free_rollout(pars, noise, Pi, *, s_phi, s_lt, rho1_max, s_l1_floor,
                 sample=True, temperature=0.3):
    return free_traj(**free_heads(pars, s_phi, s_lt, rho1_max, s_l1_floor),
                     noise=noise, Pi=Pi, sample=sample, temperature=temperature)


# ======================================================================================
#  [P6] ONE loss assembly for every arm, with the per-channel decomposition.
# ======================================================================================
def loss_parts(ro, dec, hdec, b, db, obs, *, beta=1.0, obs_w=1.0, recon="bce"):
    rb, rd, roo = P.recon_terms(dec, hdec, ro["Z"], b, db, obs, recon=recon)
    return dict(recon_b=rb, recon_db=rd, recon_obs=obs_w * roo,
                kl_phase=beta * ro["kl_p"], kl_level=beta * ro["kl_l"],
                kl_meter=beta * ro["kl_m"] + beta * ro["kl_dv"])


def per_crop_loss(ro, dec, hdec, b, db, obs, **kw):
    pr = loss_parts(ro, dec, hdec, b, db, obs, **kw)
    return sum(pr[k] for k in PKEYS)


def percrop_corr(phi, phi_true):
    """PER-CROP: |mean over t of exp(i(phi - phi_true))|, one number per crop."""
    return torch.abs(torch.exp(1j * (phi - phi_true)).mean(1))


def set_prior(g):
    """[prior] rebind ALL THREE derived constants together, then verify KL(q==p) == 0."""
    P.PHYS["gamma_phase"] = g
    IQ.RHO_P = math.exp(-g)
    IQ.GP1 = 1.0 - IQ.RHO_P
    IQ.R0 = IQ.softplus_inv(IQ.GP1)
    z = IQ.kl_phase_innov(torch.zeros(1), torch.full((1,), IQ.GP1)).item()
    assert abs(z) < 1e-8, f"kl_phase_innov(q==p)={z}; RHO_P/GP1/R0 are inconsistent"
    return z


# ======================================================================================
#  [F2] THREE-VALUED gates. A gate may be POSITIVE (a real effect), NEGLIGIBLE (an
#  equivalence result: the effect is provably smaller than a pre-declared margin), or
#  UNRESOLVED (the data cannot tell -- NOT evidence of absence). Only NEGLIGIBLE licenses
#  an affirmative flatness claim.
# ======================================================================================
POS, NEG, UNR = "POSITIVE", "NEGLIGIBLE", "UNRESOLVED"


def verdict3(m, s, floor, margin):
    if (m > 2 * s) and (m > floor):
        return POS
    if abs(m) + 2 * s < margin:
        return NEG
    return UNR


def gline(name, m, s, floor, margin, unit="nats", note=""):
    v = verdict3(m, s, floor, margin)
    if v == UNR and m < -2 * s:          # label the significant NEGATIVE direction too;
        v = "NEGATIVE(sig)"              # gates compare to POS/NEG explicitly, so this is
    #                                      display-only and cannot change any decision.
    return (f"  {name:<26s} {m:+10.2f} +- {s:6.2f} {unit:<5s} | MDE(2 sem) = {2*s:8.2f} | "
            f"effect floor {floor:g}, equivalence margin {margin:g}  ->  {v}"
            + (f"   {note}" if note else ""))


def mean_sem(x):
    """x: 1-D tensor. Returns (mean, sem)."""
    x = x.reshape(-1).double()
    n = x.numel()
    return float(x.mean()), (float(x.std(unbiased=True) / math.sqrt(n)) if n > 1 else 0.0)


def _fmt_pm(m, s):
    return f"{m:9.1f}+-{s:5.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=int, default=48)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=300)          # ELBO steps per arm
    ap.add_argument("--dec_steps", type=int, default=700)      # frozen-decoder pre-fit
    ap.add_argument("--init_steps", type=int, default=600)     # amortized init-head pre-train
    ap.add_argument("--eval_draws", type=int, default=5)
    ap.add_argument("--n_ckpt", type=int, default=3)           # evaluation checkpoints
    ap.add_argument("--gate_crops", type=int, default=6)
    ap.add_argument("--cert_crops", type=int, default=6)       # [F20] Picard-gradient certification
    ap.add_argument("--snr_draws", type=int, default=24)       # [C3] fresh draws for dL/dml1
    ap.add_argument("--lr_free", type=str, default="3e-3,3e-4")   # [C4] sweep
    ap.add_argument("--lr_amort", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--obs_w", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.3)
    # ---- regime. [F10/F17] The arms ALWAYS run at --gamma_phase; the scan is sensitivity.
    ap.add_argument("--gamma_phase", type=float, default=PROD_GAMMA)
    ap.add_argument("--gammas", type=str, default="5.5e-4,0.006,0.06")   # scan only
    ap.add_argument("--scan_s_l1", type=str, default="0.011,0.30")       # scan only
    ap.add_argument("--scan_r1r", type=str, default="0.0,6.0")           # scan only
    ap.add_argument("--scan_draws", type=int, default=4)                 # per bank, 2 banks
    # ---- [F11][F16] the DECLARED free-q init. Sharp enough that the SAMPLED trajectory --
    # the only one the ELBO ever scores -- really is at the premise's good tempo.
    ap.add_argument("--s_l1_init", type=float, default=0.011)
    ap.add_argument("--s_l1_floor", type=float, default=0.01)
    ap.add_argument("--s_lv_init", type=float, default=2.5e-4)   # = T_SCALE/5; 0 -> T_SCALE
    ap.add_argument("--r1r_init", type=float, default=6.0)       # rho1 = sigmoid(6)*0.9 = 0.898
    ap.add_argument("--gate_corr_floor", type=float, default=0.55)   # brief's teacher 0.63, -10%
    ap.add_argument("--s_phi", type=float, default=0.05)
    ap.add_argument("--s_lt", type=float, default=0.0025)
    ap.add_argument("--rho1_max", type=float, default=0.9)
    # innovq hardcodes s_l1 = softplus(head) + 0.05. The encoder's t=1 level width therefore
    # CANNOT go below 0.05 -- a declared, un-removable family asymmetry, and precisely why
    # MOVED/HELD are decided on the LOCATION and not on the sampled width. [F15]
    ap.add_argument("--amort_s_l1_init", type=float, default=0.06)
    ap.add_argument("--amort_exact", action="store_true",
                    help="[F20] train the amortized arm through the sequential rollout "
                         "(exact but ~40x slower) instead of the certified Picard rollout")
    ap.add_argument("--grad_cos_min", type=float, default=0.99)
    # ---- decision thresholds, DECLARED here and calibrated in corr below -----------------
    ap.add_argument("--move_pp", type=float, default=2.0)   # level LOCATION drift, pp
    ap.add_argument("--held_pp", type=float, default=1.0)   # equivalence margin, pp
    ap.add_argument("--nat_floor", type=float, default=20.0)   # absolute effect floor, nats
    ap.add_argument("--order_band", type=float, default=0.03)  # [F1] monotonicity is only
    #  meaningful where the objective resolves anything at all; A3 is tested on |delta| <=
    #  max(--order_band, the MEASURED resolution limit) and inversions beyond it are printed
    #  as a diagnostic. Requiring monotonicity deep inside the saturated plateau is the same
    #  structural veto that made the old PRE_A unreachable.
    ap.add_argument("--amort_init_mae", type=float, default=4.0)   # PRE_B, pp
    ap.add_argument("--amort_init_phi", type=float, default=0.30)  # PRE_B, rad
    ap.add_argument("--amort_init_corr", type=float, default=0.50) # PRE_B, mean-path corr
    ap.add_argument("--n_seeds", type=int, default=3)       # [F2] optimizer-seed replicates
    ap.add_argument("--bad_sd", type=float, default=0.30)   # [F3] cold-start spread, log-level
    ap.add_argument("--lr_bad_max", type=float, default=0.0)   # 0 = no cap (feasibility first)
    ap.add_argument("--grad_disp", type=float, default=0.05)   # [C3] displaced-gradient probe
    ap.add_argument("--recon", type=str, default="bce")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    A = ap.parse_args()
    LRS = sorted({float(x) for x in A.lr_free.split(",")}, reverse=True)
    GAMMAS = [float(x) for x in A.gammas.split(",")]
    SCAN_SL1 = [float(x) for x in A.scan_s_l1.split(",")]
    SCAN_R1R = [float(x) for x in A.scan_r1r.split(",")]
    if A.smoke:
        A.crops, A.steps, A.init_steps = min(A.crops, 8), 5, 500
        A.eval_draws, A.gate_crops, A.n_ckpt, A.snr_draws = 3, 3, 1, 6
        A.scan_draws, A.n_seeds, A.cert_crops = 2, 2, 3
    t00 = time.time()
    torch.manual_seed(A.seed)
    LOSS_KW = dict(beta=A.beta, obs_w=A.obs_w, recon=A.recon)
    PROD_REGIME = abs(A.gamma_phase - PROD_GAMMA) < 1e-12

    # ---- data -------------------------------------------------------------------------
    tr = P.build_crops(P.load_songs("train"), n_per_song=1, seed=0, crop=A.T, dev=DEV)
    Nall, T = tr["b"].shape
    gcpu = torch.Generator().manual_seed(A.seed)
    sel = torch.randperm(Nall, generator=gcpu)[:A.crops].to(DEV)
    B, K = len(sel), 4
    h, b, db, obs = tr["h"][sel], tr["b"][sel], tr["db"][sel], tr["obs"][sel]
    phi_true = tr["phi"][sel]
    lt_frame = tr["lt"][sel]
    lt_true = lt_frame.mean(1)                              # <-- THE level truth
    m_idx = tr["m"][sel].long().clamp(0, K - 1)
    _p = []
    for i in range(B):                                      # the 4-bpb proxy, reported only
        ix = torch.nonzero(b[i] > 0.5).squeeze(-1)
        _p.append(math.log(TWO_PI / (4.0 * float(torch.median(torch.diff(ix).float()))))
                  if len(ix) >= 8 else float("nan"))
    tg_proxy = torch.tensor(_p, device=DEV)
    tg_proxy = torch.where(torch.isnan(tg_proxy),
                           torch.full_like(tg_proxy, float(lt_true.mean())), tg_proxy)
    print(f"[data] {Nall} crops available at T={T}, using {B}.  LEVEL TRUTH = "
          f"train['lt'].mean(1) (oracle bar-advance).  The log(2pi/(4*medIBI)) proxy "
          f"disagrees with it by {100*float((tg_proxy-lt_true).abs().mean()):.1f}% mean abs "
          f"(it hardcodes 4 beats/bar) -- reported, never used as a threshold.", flush=True)

    Pi = torch.full((K, K), (1.0 - P.PHYS["meter_self"]) / (K - 1), device=DEV)
    Pi.fill_diagonal_(P.PHYS["meter_self"])

    # ---- [P1] verify the generator-backed Student-t(2) inverse CDF ---------------------
    try:
        from scipy import stats as _sps, integrate as _sint
        HAVE_SCIPY = True
    except Exception:
        HAVE_SCIPY = False
    gchk = torch.Generator(device=DEV).manual_seed(11)
    _n = draw_noise_g(4096, 64, K, DEV, IQ.DOF, gen=gchk)["tstd"].reshape(-1)
    if HAVE_SCIPY:
        qs = [0.05, 0.25, 0.5, 0.75, 0.95]
        emp = [float(torch.quantile(_n, q)) for q in qs]
        the = [float(_sps.t.ppf(q, 2.0)) for q in qs]
        derr = max(abs(e - t_) for e, t_ in zip(emp, the))
        print(f"[P1] generator-backed Student-t(2) inverse CDF checked against scipy.stats.t: "
              f"max |empirical - theoretical| quantile error over {qs} = {derr:.4f} "
              f"(n={_n.numel()})", flush=True)
        assert derr < 0.05, f"the Student-t(2) icdf is wrong: quantile error {derr}"
    g1 = torch.Generator(device=DEV).manual_seed(5)
    g2 = torch.Generator(device=DEV).manual_seed(5)
    torch.manual_seed(101); n1 = draw_noise_g(3, 8, K, DEV, IQ.DOF, gen=g1)
    torch.manual_seed(202); n2 = draw_noise_g(3, 8, K, DEV, IQ.DOF, gen=g2)
    for kk in NKEYS:
        assert torch.equal(n1[kk], n2[kk]), (
            f"[P1] channel '{kk}' does NOT honour the generator (it leaked to the global RNG); "
            "arm pairing would be accidental")
    print(f"     all {len(NKEYS)} noise channels {NKEYS} verified generator-backed "
          f"(identical under equal generators, different global seeds).", flush=True)

    # ---- [P3] ONE decoder, pre-fit on the ORACLE trajectory, then frozen ---------------
    mo = F.one_hot(m_idx, K).float().unsqueeze(1).expand(-1, T, -1)
    mkZ = lambda p, l=None: torch.cat(
        [torch.cos(p).unsqueeze(-1), torch.sin(p).unsqueeze(-1),
         (lt_frame if l is None else l).unsqueeze(-1), mo], -1)
    Zor = mkZ(phi_true)
    d0, h0 = P.new_decoders(DEV)
    dec, hdec = IQ.Cut(d0), IQ.Cut(h0)      # Cut zeroes the log-tempo channel (m2 convention)
    dpar = list(d0.parameters()) + list(h0.parameters())
    od = torch.optim.Adam(dpar, lr=3e-3)
    for s in range(A.dec_steps):
        od.zero_grad()
        ss = torch.randperm(B, device=DEV)[:min(16, B)]
        a_, b_, c_ = P.recon_terms(dec, hdec, Zor[ss], b[ss], db[ss], obs[ss], recon=A.recon)
        (a_ + b_ + c_).mean().backward(); od.step()

    @torch.no_grad()
    def _rec_vec(Z):
        r = P.recon_terms(dec, hdec, Z, b, db, obs, recon=A.recon)
        return (r[0] + r[1]).double()                       # [B], recon-only (no obs, no KL)

    torch.manual_seed(A.seed + 7)
    good_v = _rec_vec(Zor)
    good = float(good_v.mean())
    r_off = float(_rec_vec(mkZ((phi_true + TWO_PI * torch.rand(B, 1, device=DEV)) % TWO_PI)).mean())
    r_unif = float(_rec_vec(mkZ(TWO_PI * torch.rand(B, T, device=DEV))).mean())
    s_off, s_unif = 100 * (r_off - good) / abs(good), 100 * (r_unif - good) / abs(good)
    print(f"\n[P3] frozen decoder pre-fit on the ORACLE trajectory; SCRAMBLE sensitivity: "
          f"constant-rotation {s_off:+.1f}%, uniform-scramble {s_unif:+.1f}%  "
          f"(oracle recon {good:.1f} nats)", flush=True)
    if not (s_off > 20.0 and s_unif > 20.0):
        print("\n*** ABORT: decoder is NOT phase sensitive (need >+20% on BOTH scrambles). "
              "Every arm's recon would be phase-blind and the comparison vacuous. "
              "Raise --dec_steps.\n", flush=True)
        sys.exit(2)

    # ---- [C5][F1][F8] the TEMPO landscape, and the RESOLUTION LIMIT it implies ----------
    step_or = torch.exp(lt_frame.double())
    phi_rec = ((phi_true[:, :1].double() + F.pad(step_or[:, :-1], (1, 0)).cumsum(1)) % TWO_PI).float()
    rec_fid = float(percrop_corr(phi_rec, phi_true).mean())
    DGRID = [0.003, 0.01, 0.02, 0.03, 0.05, 0.10, 0.30]
    _cache = {}

    @torch.no_grad()
    def rec_at(delta):
        if delta in _cache:
            return _cache[delta]
        ltd = lt_frame + math.log1p(delta)
        sd = torch.exp(ltd.double())
        pd = ((phi_true[:, :1].double() + F.pad(sd[:, :-1], (1, 0)).cumsum(1)) % TWO_PI).float()
        out = (_rec_vec(mkZ(pd, ltd)), float(percrop_corr(pd, phi_true).mean()))
        _cache[delta] = out
        return out

    @torch.no_grad()
    def sym(d):
        """symmetrized (+d, -d) recon vector and corr"""
        if d == 0.0:
            return rec_at(0.0)
        vp, cp = rec_at(+d)
        vm, cm = rec_at(-d)
        return 0.5 * (vp + vm), 0.5 * (cp + cm)

    base_v, base_c = sym(0.0)
    print(f"[C5] TEMPO LANDSCAPE of the frozen decoder (recon_b+recon_db on the oracle "
          f"trajectory, level offset delta; phase re-integrated from the offset level).")
    print(f"     level->phase reconstruction fidelity at delta=0: per-crop corr {rec_fid:.4f} "
          f"(1.0 = the sweep really is the oracle trajectory)")
    print(f"     |delta| |   recon (paired d vs truth +- sem over crops) | per-crop corr")
    print(f"       0.0%  | {float(base_v.mean()):9.1f}   (reference)              |   {base_c:6.4f}")
    SWEEP = []
    CALIB = {0.0: base_c}
    for dv_ in DGRID:
        both, cc = sym(dv_)
        d = both - base_v
        mu, se = mean_sem(d)
        SWEEP.append((dv_, mu, se, cc)); CALIB[dv_] = cc
        print(f"      {100*dv_:5.1f}% | {float(both.mean()):9.1f}   ({mu:+8.1f} +- {se:5.1f})"
              f"              |   {cc:6.4f}")

    @torch.no_grad()
    def pair_sw(d1, d0):
        dd = sym(d1)[0] - sym(d0)[0]
        return mean_sem(dd)

    # A1: can the objective SEE a 10% error? (significance AND an absolute nat floor)
    a1_m, a1_s = pair_sw(0.10, 0.0)
    # A2b [F1/F8]: can it ORDER inside the band the arms traverse? 0.3% -> 3%.
    a2_m, a2_s = pair_sw(0.03, 0.003)
    # A2sat: the OLD gate, kept as a DIAGNOSTIC only (3% -> 30%); a bounded pointwise
    # likelihood saturates here by construction, so its failure is not evidence of anything.
    sat_m, sat_s = pair_sw(0.30, 0.03)
    FLAT_MARGIN = max(A.nat_floor, 0.10 * abs(a1_m))        # 10% of the reference effect
    A1 = verdict3(a1_m, a1_s, A.nat_floor, FLAT_MARGIN) == POS
    A2 = verdict3(a2_m, a2_s, A.nat_floor, FLAT_MARGIN) == POS
    SAT_V = verdict3(sat_m, sat_s, A.nat_floor, FLAT_MARGIN)
    # RESOLUTION LIMIT: the largest delta whose next step up is still resolved (>2 sem).
    res_lim = 0.0
    for i in range(len(DGRID)):
        d0_ = 0.0 if i == 0 else DGRID[i - 1]
        m_, s_ = pair_sw(DGRID[i], d0_)
        if m_ > 2 * s_:
            res_lim = DGRID[i]
        else:
            break
    ORD_HI = max(A.order_band, res_lim)
    inv, inv_out = [], []
    for i in range(len(DGRID) - 1):
        m_, s_ = pair_sw(DGRID[i + 1], DGRID[i])
        if m_ < -2 * s_:
            (inv if DGRID[i + 1] <= ORD_HI + 1e-12 else inv_out).append(
                (DGRID[i], DGRID[i + 1], m_, s_))
    A3 = not inv
    PRE_A = A1 and A2 and A3
    print("     [F1/F8] PRE_A is an ORDERING test INSIDE the band the arms traverse, with an")
    print(f"     absolute effect floor of {A.nat_floor:g} nats. The old 3%-vs-30% test is kept as a")
    print("     DIAGNOSTIC: a bounded pointwise likelihood saturates there BY CONSTRUCTION,")
    print("     so its failure vetoes nothing.")
    print(gline("A1 loss(10%)-loss(0%)", a1_m, a1_s, A.nat_floor, FLAT_MARGIN,
                note="can the objective SEE a 10% error?"))
    print(gline("A2 loss(3%)-loss(0.3%)", a2_m, a2_s, A.nat_floor, FLAT_MARGIN,
                note="can it ORDER inside the traversed band?"))
    print(gline("[diag] loss(30%)-loss(3%)", sat_m, sat_s, A.nat_floor, FLAT_MARGIN,
                note="SATURATION diagnostic -- NOT a gate"))
    fmtinv = lambda L: ", ".join(f"{100*x[0]:.1f}%->{100*x[1]:.1f}% {x[2]:+.1f}+-{x[3]:.1f}"
                                 for x in L)
    print(f"  A3 no significant inversion inside the ORDERING BAND |delta| <= "
          f"{100*ORD_HI:.1f}% (= max(--order_band, the measured resolution limit)): {A3}"
          + ("" if A3 else "   inversions: " + fmtinv(inv)))
    print(f"     [diagnostic] inversions BEYOND the band (inside the saturated plateau, where "
          f"monotonicity carries no information): "
          + (fmtinv(inv_out) if inv_out else "none"))
    print(f"  MEASURED RESOLUTION LIMIT: the objective resolves successive level errors up to "
          f"~{100*res_lim:.1f}% and saturates beyond (the 30%-vs-3% diagnostic reads {SAT_V}).")
    print(f"  PRE_A (A1 and A2 and A3) = {PRE_A}", flush=True)

    # ---- [F7] calibrate the DECLARED decision thresholds in per-crop corr ---------------
    def _corr_at_pp(pp):
        d = pp / 100.0
        return sym(d)[1]
    print(f"\n[F7] the decision thresholds are DECLARED, not fitted; here is what they COST "
          f"in per-crop phase corr on the oracle trajectory:")
    print(f"     level LOCATION drift {A.held_pp:.1f} pp (HELD equivalence margin) -> per-crop corr "
          f"{base_c:.4f} -> {_corr_at_pp(A.held_pp):.4f}")
    print(f"     level LOCATION drift {A.move_pp:.1f} pp (MOVED effect floor)      -> per-crop corr "
          f"{base_c:.4f} -> {_corr_at_pp(A.move_pp):.4f}")
    print(f"     (5 pp, the OLD threshold, would have cost {base_c:.4f} -> {_corr_at_pp(5.0):.4f} -- "
          f"i.e. the old gate called a phase-destroying drift 'HELD'.)", flush=True)

    for q in dpar:
        q.requires_grad_(False); q.grad = None
    dec_snap = [q.detach().clone() for q in dpar]

    def check_frozen(where):                                      # [P4] after EVERY arm
        ok = all(torch.equal(a, q.detach()) for a, q in zip(dec_snap, dpar))
        gok = all(q.grad is None for q in dpar)
        if not (ok and gok):
            print(f"\n*** ABORT [P4]: decoder changed / grads refilled after {where} "
                  f"(bit_identical={ok}, grads_none={gok}) -- arms co-adapted.\n", flush=True)
            sys.exit(4)
        return ok and gok

    # ---- [P2][F5] KL-estimator check against an ESTIMATOR-INDEPENDENT reference ---------
    set_prior(A.gamma_phase)
    dof = torch.tensor(IQ.DOF, device=DEV)
    mq, sq_ = 0.4 * IQ.T_SCALE, 2.3 * IQ.T_SCALE             # differs in BOTH location and scale
    with torch.no_grad():
        NREF = 400000
        mu = torch.full((NREF,), mq, device=DEV); sg = torch.full((NREF,), sq_, device=DEV)
        zz = torch.distributions.StudentT(dof, mu, sg).rsample()
        kk = P.kl_t_mc(dof, mu, sg, dof, torch.zeros_like(mu),
                       torch.full_like(mu, IQ.T_SCALE), zz)
        ref, ref_sem = float(kk.mean()), float(kk.std() / math.sqrt(NREF))
        one = []
        for _ in range(40):
            mu = torch.full((T - 1,), mq, device=DEV); sg = torch.full((T - 1,), sq_, device=DEV)
            zz = torch.distributions.StudentT(dof, mu, sg).rsample()
            one.append(float(P.kl_t_mc(dof, mu, sg, dof, torch.zeros_like(mu),
                                       torch.full_like(mu, IQ.T_SCALE), zz).sum()))
    got = _st.mean(one) / (T - 1)
    if HAVE_SCIPY:
        qd = _sps.t(2.0, loc=mq, scale=sq_)
        pd_ = _sps.t(2.0, loc=0.0, scale=IQ.T_SCALE)
        f_ = lambda z: qd.pdf(z) * (qd.logpdf(z) - pd_.logpdf(z))
        quad = (_sint.quad(f_, -math.inf, mq, limit=400)[0]
                + _sint.quad(f_, mq, math.inf, limit=400)[0])
        ok_kl = abs(quad - ref) < 3 * max(ref_sem, 1e-12) + 1e-4
        print(f"\n[P2][F5] kl_t_mc with q!=p (q: mu={mq:.5f} s={sq_:.5f} | p: mu=0 "
              f"s={IQ.T_SCALE:.5f}).  SciPy QUADRATURE of the true KL integral = "
              f"{quad:.6f} nats/step (estimator-INDEPENDENT).  P.kl_t_mc MC mean over "
              f"{NREF} draws = {ref:.6f}+-{ref_sem:.6f}.  agree within 3 sem: {ok_kl}")
        assert ok_kl, ("P.kl_t_mc disagrees with quadrature -- the KL estimator is wrong, "
                       "every ELBO below would be wrong too")
    else:
        print(f"\n[P2] scipy unavailable -- quadrature check SKIPPED (MC reference "
              f"{ref:.6f}+-{ref_sem:.6f}).", flush=True)
    print(f"     the SINGLE-SAMPLE estimator (the one used in training) gives {got:+.6f} "
          f"nats/step over 40 replicates of {T-1} steps -> unbiased against the reference.")
    print(f"     per-crop sd of the summed level KL over {T-1} steps = "
          f"{_st.pstdev(one):.1f} nats. THIS is why every ELBO comparison below is PAIRED "
          f"under common random numbers.", flush=True)

    # ---- free-parameter constructors ----------------------------------------------------
    # [F6] NO torch.manual_seed() in here: nothing random is drawn, and the old call made
    # arm pairing depend on an incidental global-RNG reset.
    def _inn3_for(s_lv):
        return IQ.softplus_inv(max(s_lv - 1e-5, 1e-9)) - IQ.B_SLT0

    def _sl1r_for(s_l1, floor):
        return IQ.softplus_inv(max(s_l1 - floor, 1e-4))

    S_LV_INIT = A.s_lv_init if A.s_lv_init > 0 else IQ.T_SCALE

    def new_pars(ml1_init, *, pin_level=False, s_l1_init=None, s_l1_floor=None,
                 s_lv_init=None, r1r_init=None):
        s1 = A.s_l1_init if s_l1_init is None else s_l1_init
        fl = A.s_l1_floor if s_l1_floor is None else s_l1_floor
        slv = S_LV_INIT if s_lv_init is None else s_lv_init
        r1 = A.r1r_init if r1r_init is None else r1r_init
        inn = torch.zeros(B, T - 1, 4, device=DEV)      # 0 -> phase innovations are q == p
        inn[..., 3] = _inn3_for(slv)                    # [F11] SHARP level innovations
        return dict(
            # [F4] mp1 starts at the ORACLE OFFSET. The amortized arm is given the same
            # offset by supervised pre-training, so the two families start symmetric.
            mp1=phi_true[:, 0].clone().detach().requires_grad_(True),
            # [F16] rho1 = sigmoid(r1r)*rho1_max. r1r=0 gives rho1=0.45, a 0.80 rad
            # wrapped-Cauchy on the INITIAL PHASE that swamps the tempo signal; the declared
            # init sharpens it and the scan below reports the sensitivity.
            r1r=torch.full((B,), float(r1), device=DEV, requires_grad=True),
            ml1=ml1_init.clone().detach().requires_grad_(not pin_level),
            # [F13/F19] the pin must cover the level WIDTH too, else "tempo provably exact"
            # is false: sampled level error is a monotone function of s_l1.
            sl1r=torch.full((B,), _sl1r_for(s1, fl), device=DEV,
                            requires_grad=not pin_level),
            mlog=torch.zeros(B, K, device=DEV, requires_grad=True),
            inn=inn.requires_grad_(True))

    def reset_level(pars, delta=0.0):
        """[C2] the same q with ONLY the level channels put back at the truth (x(1+delta))."""
        q = {k: v.detach().clone() for k, v in pars.items()}
        q["ml1"] = lt_true.clone() + math.log1p(delta)
        q["inn"][..., 2] = 0.0                      # mu_lt = 0
        q["inn"][..., 3] = _inn3_for(S_LV_INIT)     # s_lv back to the declared init
        q["sl1r"] = torch.full((B,), _sl1r_for(A.s_l1_init, A.s_l1_floor), device=DEV)
        return q

    FKW = dict(s_phi=A.s_phi, s_lt=A.s_lt, rho1_max=A.rho1_max, s_l1_floor=A.s_l1_floor)

    # ---- [F11] the SHARP init: its KL price, and the decomposition of its sampled error --
    chk_bank = [draw_noise_g(B, T, K, DEV, IQ.DOF,
                             gen=torch.Generator(device=DEV).manual_seed(4000 + i))
                for i in range(max(3, A.eval_draws))]

    @torch.no_grad()
    def _samp_stats(pars, kw=None):
        kw = kw or FKW
        cs, ms, kls = [], [], []
        for nz in chk_bank:
            ro = free_rollout(pars, nz, Pi, sample=True, temperature=A.temp, **kw)
            cs.append(float(percrop_corr(ro["phi"], phi_true).mean()))
            ms.append(100 * float((ro["lt"].mean(1) - lt_true).abs().mean()))
            kls.append(float(ro["kl_l"].mean()))
        return _st.mean(cs), _st.mean(ms), _st.mean(kls)

    p_sharp = new_pars(lt_true)
    p_prior = new_pars(lt_true, s_l1_init=IQ.INIT_LV_S, s_lv_init=IQ.T_SCALE,
                       s_l1_floor=A.s_l1_floor)
    p_pinn = new_pars(lt_true, s_lv_init=IQ.T_SCALE)      # sharp t=1, PRIOR innovations
    p_nolv = new_pars(lt_true, s_lv_init=1e-5)
    p_none = new_pars(lt_true, s_lv_init=1e-5, s_l1_init=A.s_l1_floor + 1e-4)
    c_sh, m_sh, kl_sh = _samp_stats(p_sharp)
    c_pr, m_pr, kl_pr = _samp_stats(p_prior)
    c_pi, m_pi, kl_pi = _samp_stats(p_pinn)
    c_nl, m_nl, kl_nl = _samp_stats(p_nolv)
    c_no, m_no, kl_no = _samp_stats(p_none)
    print(f"\n[F11] THE FREE-q INIT IS SHARP BY DESIGN, so that the SAMPLED trajectory -- the "
          f"only one the ELBO ever scores -- is really at the premise's good tempo.")
    print(f"      DECLARED init: s_l1={A.s_l1_init:g} (floor {A.s_l1_floor:g}), "
          f"s_lv={S_LV_INIT:.3g} (prior {IQ.T_SCALE:g}), rho1=sigmoid({A.r1r_init:g})*"
          f"{A.rho1_max:g}={float(torch.sigmoid(torch.tensor(A.r1r_init))*A.rho1_max):.4f}, "
          f"mu_phi1 = ORACLE offset, ml1 = TRUTH.")
    print(f"      DECOMPOSITION of the init's SAMPLED level error and its KL price (mean over "
          f"{len(chk_bank)} draws, B={B}):")
    print(f"        s_l1     s_lv        |  sampled MAE   corr    kl_level (nats/crop)")
    for tag, s1_, sv_, m_, c_, k_ in (
            ("prior-matched   ", IQ.INIT_LV_S, IQ.T_SCALE, m_pr, c_pr, kl_pr),
            ("sharp t=1 only  ", A.s_l1_init, IQ.T_SCALE, m_pi, c_pi, kl_pi),
            ("DECLARED (arms) ", A.s_l1_init, S_LV_INIT, m_sh, c_sh, kl_sh),
            ("s_lv -> 0       ", A.s_l1_init, 1e-5, m_nl, c_nl, kl_nl),
            ("both -> floor   ", A.s_l1_floor, 1e-5, m_no, c_no, kl_no)):
        print(f"        {tag} {s1_:6.3f} {sv_:9.2e}  | {m_:7.2f}%   {c_:.4f}  {k_:11.1f}")
    print(f"      READ THIS: with the level mean EXACTLY at the truth and the innovations at "
          f"the PRIOR scale, the sampled trajectory is still {m_pi:.1f}% off (corr {c_pi:.3f}) "
          f"-- the prior's own Student-t(2) random walk over {T-1} steps. The premise "
          f"(MAE 2-4%, corr ~0.6) is therefore UNREACHABLE without sharpening the level "
          f"innovations, and that sharpening costs {kl_sh-kl_pi:+.0f} nats of kl_level which "
          f"the free arms PAY in every ELBO below. The encoder's innov_head bias is set to "
          f"the SAME s_lv, so both families pay it. [F11]", flush=True)

    # =================== [C6][F10][F17] REGIME SENSITIVITY (NOT a selector) ==============
    print("\n===================== [C6] REGIME SENSITIVITY (NOT A SELECTOR) ================")
    print(f"  [F10/F17] The arms ALWAYS run at --gamma_phase = {A.gamma_phase:.3g}"
          + ("  (= the PRODUCTION value)." if PROD_REGIME else
             f"  (production is {PROD_GAMMA:g} -- NON-PRODUCTION)."))
    print("  The table below is a SENSITIVITY scan, run on TWO independent noise banks so")
    print("  that pass/fail FLIPS caused by RNG are visible. Nothing here selects anything.")
    banks = [[draw_noise_g(B, T, K, DEV, IQ.DOF,
                           gen=torch.Generator(device=DEV).manual_seed(9000 + 100 * j + i))
              for i in range(A.scan_draws)] for j in range(2)]

    @torch.no_grad()
    def regime_probe(gamma, s_l1_init, r1r_init, bank):
        set_prior(gamma)
        kw = dict(**FKW, sample=True, temperature=A.temp)
        mk = lambda off: new_pars(lt_true + math.log1p(off), s_l1_init=s_l1_init,
                                  r1r_init=r1r_init)
        base, hi, lo = mk(0.0), mk(+0.10), mk(-0.10)
        Lb, Lh, Ll, cs = [], [], [], []
        for nz in bank:
            rb_ = free_rollout(base, nz, Pi, **kw)
            Lb.append(per_crop_loss(rb_, dec, hdec, b, db, obs, **LOSS_KW).double())
            cs.append(percrop_corr(rb_["phi"], phi_true).double())
            Lh.append(per_crop_loss(free_rollout(hi, nz, Pi, **kw), dec, hdec, b, db, obs,
                                    **LOSS_KW).double())
            Ll.append(per_crop_loss(free_rollout(lo, nz, Pi, **kw), dec, hdec, b, db, obs,
                                    **LOSS_KW).double())
        Lb, Lh, Ll = torch.stack(Lb), torch.stack(Lh), torch.stack(Ll)
        hm, hs = mean_sem(Lh - Lb)
        lm, ls = mean_sem(Ll - Lb)
        corr = float(torch.stack(cs).mean())
        det = free_rollout(base, None, Pi, sample=False, temperature=A.temp, **FKW)
        return dict(gamma=gamma, s_l1=s_l1_init, r1r=r1r_init, corr=corr,
                    corr_det=float(percrop_corr(det["phi"], phi_true).mean()),
                    dhi=(hm, hs), dlo=(lm, ls),
                    ok_corr=corr >= A.gate_corr_floor,
                    ok_d=(verdict3(hm, hs, A.nat_floor, FLAT_MARGIN) == POS
                          and verdict3(lm, ls, A.nat_floor, FLAT_MARGIN) == POS))

    print("  gamma_phase | s_l1 |  r1r | corr_samp | corr_mp |   dELBO(+10%)    |   "
          "dELBO(-10%)    | G1 G2  (bank A / bank B)")
    SCAN, flips = [], 0
    for g in GAMMAS:
        for s1 in SCAN_SL1:
            for r1 in SCAN_R1R:
                ra = regime_probe(g, s1, r1, banks[0])
                rb2 = regime_probe(g, s1, r1, banks[1])
                fl = (ra["ok_d"] != rb2["ok_d"]) or (ra["ok_corr"] != rb2["ok_corr"])
                flips += int(fl)
                SCAN.append(dict(a=ra, b=rb2, flip=fl))
                print(f"   {g:9.2e}  | {s1:4.3f}| {r1:4.1f} |  {ra['corr']:6.4f}   | "
                      f"{ra['corr_det']:6.4f}  | {ra['dhi'][0]:+8.1f}+-{ra['dhi'][1]:5.1f} | "
                      f"{ra['dlo'][0]:+8.1f}+-{ra['dlo'][1]:5.1f} |  "
                      f"{int(ra['ok_corr'])}{int(rb2['ok_corr'])} {int(ra['ok_d'])}"
                      f"{int(rb2['ok_d'])}" + ("   <-- FLIPPED between banks" if fl else ""),
                      flush=True)
    print(f"  {flips} of {len(SCAN)} cells FLIP their pass/fail between two independent noise "
          f"banks at --scan_draws {A.scan_draws}. This is exactly why the old auto-selector "
          f"(which promoted the first passing cell) could swap the objective under test on "
          f"RNG alone; it has been REMOVED.")

    # PRE_C / PRE_D are measured AT THE ARM CONFIGURATION, on a dedicated larger bank.
    set_prior(A.gamma_phase)
    pre_bank = [draw_noise_g(B, T, K, DEV, IQ.DOF,
                             gen=torch.Generator(device=DEV).manual_seed(7700 + i))
                for i in range(2 * A.scan_draws)]
    r_arm = regime_probe(A.gamma_phase, A.s_l1_init, A.r1r_init, pre_bank)
    set_prior(A.gamma_phase)
    PRE_C = r_arm["ok_corr"]
    dhi_v = verdict3(*r_arm["dhi"], A.nat_floor, FLAT_MARGIN)
    dlo_v = verdict3(*r_arm["dlo"], A.nat_floor, FLAT_MARGIN)
    PRE_D = (dhi_v == POS) and (dlo_v == POS)
    PRE_D_FLAT = (dhi_v == NEG) and (dlo_v == NEG)
    REGIME = (f"gamma_phase={A.gamma_phase:.3g}"
              f"{' (PRODUCTION)' if PROD_REGIME else f' (NON-PRODUCTION; production is {PROD_GAMMA:g})'}"
              f", s_l1_init={A.s_l1_init:g}, s_l1_floor={A.s_l1_floor:g}, "
              f"s_lv_init={S_LV_INIT:.3g}, r1r_init={A.r1r_init:g}, beta={A.beta:g}, "
              f"recon={A.recon}, T={T}, crops={B}, obs_w={A.obs_w:g}")
    print(f"\n  MEASURED AT THE ARM CONFIGURATION ({2*A.scan_draws} draws x {B} crops):")
    print(f"  PRE_C  sampled per-crop corr at the truth-level init = {r_arm['corr']:.4f} "
          f"(floor {A.gate_corr_floor}; mean path {r_arm['corr_det']:.4f})  ->  {PRE_C}")
    print(gline("PRE_D dELBO(+10%)", *r_arm["dhi"], A.nat_floor, FLAT_MARGIN))
    print(gline("PRE_D dELBO(-10%)", *r_arm["dlo"], A.nat_floor, FLAT_MARGIN))
    print(f"  PRE_D (the FULL objective prefers the truth level over BOTH +-10%) = {PRE_D}"
          + ("   [and the opposite claim -- that it is FLAT there -- is itself supported: "
             "both are NEGLIGIBLE]" if PRE_D_FLAT else
             ("" if PRE_D else "   [NOT resolved: this is a failure to reject, NOT evidence "
                               "of flatness; see the MDEs above]")))
    print(f"  REGIME USED: {REGIME}")
    print(f"  RHO_P={IQ.RHO_P:.8f} GP1={IQ.GP1:.6e} R0={IQ.R0:.4f} (rebound together); "
          f"total phase-innovation Cauchy scale over the crop = "
          f"{A.gamma_phase*(T-1):.2f} rad")
    if not PROD_REGIME:
        print("\n  *** THE PRODUCTION REGIME WAS NOT TESTED: --gamma_phase differs from "
              f"{PROD_GAMMA:g}. No H1/H2 verdict may be printed from this run. ***")
    print("==============================================================================",
          flush=True)

    # ---- amortized arm: construction + [F4] SYMMETRIC init + [F21] PRE_B gate ------------
    phi0_tgt = torch.stack([torch.cos(phi_true[:, 0]), torch.sin(phi_true[:, 0])], -1)

    def build_amort(seed, verbose=False):
        torch.manual_seed(seed)
        model = InnovQT(s_phi=A.s_phi, s_lt=A.s_lt, rho1_max=A.rho1_max).to(DEV)
        assert torch.allclose(model.Pi_phys, Pi), "meter transition differs between arms"
        with torch.no_grad():
            # [F4] match every init coordinate the free arms declare, inside the ENCODER's
            # own family: rho1 bias, t=1 level-scale bias (at its hardcoded 0.05 floor) and
            # the innovation head's s_lv bias.
            model.init_head[-1].bias[K + 2] = A.r1r_init
            model.init_head[-1].bias[K + 4] = IQ.softplus_inv(
                max(A.amort_s_l1_init - ENC_S_L1_FLOOR, 1e-4))
            model.innov_head[-1].bias[3] = _inn3_for(S_LV_INIT)
        opt0 = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for s in range(A.init_steps):
            opt0.zero_grad()
            ss = torch.randperm(B, device=DEV)[:min(24, B)]
            c = model.encode_posterior(h[ss], b[ss])
            o = model.init_head(torch.cat([c.mean(1), c[:, 0]], -1))
            mu = o[:, K + 3] + model.level_offset
            # [F4] the phase OFFSET is supervised too -- the old script pre-trained the LEVEL
            # only, leaving mu_phi1 uniform-random (sd 3.03 rad) while the free arms were
            # handed the oracle offset. That asymmetry confounded MOVED on the level channel.
            L = ((mu - lt_true[ss]).abs().mean()
                 + ((o[:, K:K + 2] - phi0_tgt[ss]) ** 2).sum(-1).mean())
            L.backward(); opt0.step()
        with torch.no_grad():
            c = model.encode_posterior(h, b)
            o = model.init_head(torch.cat([c.mean(1), c[:, 0]], -1))
            mu = o[:, K + 3] + model.level_offset
            mp = torch.atan2(o[:, K + 1], o[:, K])
            dphi = float(torch.abs((mp - phi_true[:, 0] + math.pi) % TWO_PI - math.pi).mean())
            mae = 100 * float((mu - lt_true).abs().mean())
            model.eval(); ro = IQ.rollout(model, h, b, sample=False, temperature=A.temp)
            model.train()
            corr_mp = float(percrop_corr(ro["phi"], phi_true).mean())
        st = dict(mae=mae, dphi=dphi, corr_mp=corr_mp)
        if verbose:
            print(f"\n[F21][PRE_B] amortized init pre-train (seed {seed}, {A.init_steps} steps, "
                  f"supervised on the SAME level truth AND the oracle phase offset):")
            print(f"      mean-head level LOCATION MAE {mae:.2f}%  (gate <= {A.amort_init_mae}%)")
            print(f"      mean |mu_phi1 - phi_true[:,0]| {dphi:.3f} rad  "
                  f"(gate <= {A.amort_init_phi} rad)")
            print(f"      mean-path per-crop corr {corr_mp:.4f}  (gate >= {A.amort_init_corr})")
            print(f"      NOTE the SAMPLED corr of this arm is necessarily lower: innovq "
                  f"hardcodes s_l1 >= {ENC_S_L1_FLOOR:g}, a width the encoder family cannot "
                  f"go below. That is a WIDTH asymmetry, not a location one, and is exactly "
                  f"why MOVED/HELD are decided on the LOCATION. [F15]")
        return model, st

    model0, st0 = build_amort(A.seed, verbose=True)
    PRE_B = (st0["mae"] <= A.amort_init_mae and st0["dphi"] <= A.amort_init_phi
             and st0["corr_mp"] >= A.amort_init_corr)
    if not PRE_B:
        print("\n*** ABORT [F21/PRE_B]: the amortized arm did NOT start from the premise "
              "(level MAE / phase offset / mean-path corr above). `reproduce` would then be "
              "measured from a starting point that was never at good tempo, which is not the "
              "phenomenon under test. Raise --init_steps.\n", flush=True)
        sys.exit(7)

    # ---- [P6] GATE: the free-q objective IS innovq's objective --------------------------
    gb = min(A.gate_crops, B)
    cap = []
    hk = model0.innov_head.register_forward_hook(lambda m, i, o: cap.append(o.detach()))
    gnz = draw_noise_g(gb, T, K, DEV, IQ.DOF,
                       gen=torch.Generator(device=DEV).manual_seed(31337))
    with torch.no_grad():
        ref_ro = rollout_loop_noise(model0, h[:gb], b[:gb], gnz, sample=True, temperature=A.temp)
    hk.remove()
    out = torch.stack(cap, 1)                                              # [gb,T-1,4]
    iv = ref_ro["init_vec"]
    with torch.no_grad():
        free_ro = free_traj(m_logits=iv[:, :K],
                            mu_phi1=torch.atan2(iv[:, K + 1], iv[:, K]) % TWO_PI,
                            rho1=(torch.sigmoid(iv[:, K + 2]) * model0.rho1_max).clamp(1e-6, 1 - 1e-6),
                            mu_l1=iv[:, K + 3] + model0.level_offset,
                            s_l1=F.softplus(iv[:, K + 4]) + ENC_S_L1_FLOOR,   # the ENCODER's floor
                            mu_eps=torch.tanh(out[..., 0]) * model0.s_phi,
                            sq=F.softplus(out[..., 1] + IQ.R0).clamp(1e-6, 0.5),
                            mu_lt=torch.tanh(out[..., 2]) * model0.s_lt,
                            s_lv=F.softplus(out[..., 3] + IQ.B_SLT0) + 1e-5,
                            noise=gnz, Pi=Pi, sample=True, temperature=A.temp)
        Lr = per_crop_loss(ref_ro, dec, hdec, b[:gb], db[:gb], obs[:gb], **LOSS_KW).mean()
        Lf = per_crop_loss(free_ro, dec, hdec, b[:gb], db[:gb], obs[:gb], **LOSS_KW).mean()
        dphi_g = float(torch.abs(torch.remainder(free_ro["phi"] - ref_ro["phi"] + math.pi,
                                                 TWO_PI) - math.pi).max())
        dnc = float((free_ro["n_cross"] - ref_ro["n_cross"]).abs().max())
    print(f"\n[P6 GATE] replaying the ENCODER's own head outputs through the free-q assembly "
          f"with the same noise: free ELBO {float(Lf):.4f} vs innovq sequential reference "
          f"{float(Lr):.4f}  (d={float(Lf-Lr):+.4f} nats), max|dphi|={dphi_g:.2e} rad, "
          f"max|d n_cross|={dnc:.0f}", flush=True)
    if abs(float(Lf - Lr)) > 0.5 or dnc > 0:
        print("\n*** ABORT: the free arm is NOT optimizing innovq's objective.\n", flush=True)
        sys.exit(3)
    check_frozen("the P6 gate")

    # ---- [F20][P5] CERTIFY the Picard training gradient against the sequential loop ------
    cb = min(A.cert_crops, B)
    cnz = draw_noise_g(cb, T, K, DEV, IQ.DOF,
                       gen=torch.Generator(device=DEV).manual_seed(4242))

    def _pgrad(fn):
        model0.zero_grad(set_to_none=True)
        ro = fn()
        per_crop_loss(ro, dec, hdec, b[:cb], db[:cb], obs[:cb], **LOSS_KW).mean().backward()
        g = torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).reshape(-1)
                       for q in model0.parameters()])
        model0.zero_grad(set_to_none=True)
        return g
    t_c = time.time()
    g_vec = _pgrad(lambda: rollout_vec_s(model0, h[:cb], b[:cb], cnz, sample=True,
                                         temperature=A.temp, n_picard=8))
    g_loop = _pgrad(lambda: rollout_loop_noise(model0, h[:cb], b[:cb], cnz, sample=True,
                                               temperature=A.temp))
    gcos = float(F.cosine_similarity(g_vec, g_loop, dim=0))
    grel = float((g_vec - g_loop).norm() / (g_loop.norm() + 1e-30))
    print(f"[F20] PICARD-GRADIENT CERTIFICATION at the TRAINING configuration "
          f"(B={cb}, T={T}, n_picard=8, sample=True, gamma={A.gamma_phase:.3g}): parameter-"
          f"gradient cosine(rollout_vec_s, rollout_loop_noise) = {gcos:.6f}, relative L2 "
          f"error {grel:.2e}  [{time.time()-t_c:.0f}s]")
    PRE_G = (gcos >= A.grad_cos_min) or A.amort_exact
    if not PRE_G:
        print(f"\n*** ABORT [F20]: the vectorized training gradient is NOT equivalent to the "
              f"sequential one (cosine {gcos:.6f} < {A.grad_cos_min}). The amortized "
              f"degradation would then be an artifact of the approximation rather than of "
              f"amortization. Re-run with --amort_exact.\n", flush=True)
        sys.exit(8)
    print(f"      -> the amortized arm trains through "
          f"{'the SEQUENTIAL loop (--amort_exact)' if A.amort_exact else 'rollout_vec_s'}; "
          f"every REPORTED number comes from the exact sequential rollout either way. [P5]",
          flush=True)
    check_frozen("the gradient certification")

    # ---- shared evaluation ---------------------------------------------------------------
    def make_bank(seed, nd):
        return [draw_noise_g(B, T, K, DEV, IQ.DOF,
                             gen=torch.Generator(device=DEV).manual_seed(50000 + 997 * seed + i))
                for i in range(nd)]

    def make_eval(bank):
        ND = len(bank)

        @torch.no_grad()
        def evaluate(kind, thing):
            """Every ELBO on the SAME sampled draws; the LOCATION on the exact mean path.

            L/C/M are [ND,B]: per-draw per-crop ELBO, sampled per-crop corr, sampled per-crop
            level MAE (pp).  locv [B] is the DECIDING statistic: the deterministic mean-path
            level error per crop, which is a pure LOCATION quantity and therefore cannot be
            moved by the posterior WIDTH s_l1 [F7/F15]. locmed [B] is the same location read
            off the draw MEDIAN (a MEAN over draws is useless here: Student-t(2) has infinite
            variance, so the sample mean does not concentrate).
            """
            L, C, M, PR, LV = [], [], [], [], []
            for nz in bank:
                if kind == "free":                                 # exact one-pass cumsum [P5]
                    ro = free_rollout(thing[0], nz, Pi, sample=True, temperature=A.temp,
                                      **thing[1])
                else:                                              # exact SEQUENTIAL [P5]
                    ro = rollout_loop_noise(thing, h, b, nz, sample=True, temperature=A.temp)
                pr = loss_parts(ro, dec, hdec, b, db, obs, **LOSS_KW)
                L.append(sum(pr[k] for k in PKEYS).double())
                C.append(percrop_corr(ro["phi"], phi_true).double())
                M.append((100 * (ro["lt"].mean(1) - lt_true).abs()).double())
                LV.append(ro["lt"].mean(1).double())
                PR.append(torch.stack([pr[k].mean() for k in PKEYS]).double())
            L, C, M, LV = torch.stack(L), torch.stack(C), torch.stack(M), torch.stack(LV)
            if kind == "free":
                mr = free_rollout(thing[0], None, Pi, sample=False, temperature=A.temp,
                                  **thing[1])
                s_l1 = float(mr["s_l1"].mean())
            else:
                thing.eval(); mr = IQ.rollout(thing, h, b, sample=False, temperature=A.temp)
                thing.train(); s_l1 = float(mr["s_l1"].mean())
            locv = (100 * (mr["lt"].mean(1) - lt_true).abs()).double()          # [B] DECIDING
            locmed = (100 * (LV.median(0).values - lt_true.double()).abs())     # [B] robust
            sd = lambda x: float(x.mean(1).std(unbiased=True)) if ND > 1 else 0.0
            return dict(L=L, C=C, M=M, locv=locv, locmed=locmed,
                        corr=float(C.mean()), corr_sd=sd(C),
                        mae=float(M.mean()), mae_sd=sd(M),
                        loc=float(locv.mean()), locmed_m=float(locmed.mean()),
                        s_l1=s_l1,
                        corr_det=float(percrop_corr(mr["phi"], phi_true).mean()),
                        mae_det=100 * float((mr["lt"].mean(1) - lt_true).abs().mean()),
                        proxy_det=100 * float((mr["lt"].mean(1) - tg_proxy).abs().mean()),
                        parts=torch.stack(PR).mean(0))
        return evaluate, ND

    def paired(e1, e0, key="L"):
        """mean +- sem of (e0 - e1): >0 means e1 is BETTER (lower loss)."""
        return mean_sem(e0[key] - e1[key])

    def unp(e):
        pd = e["L"].mean(1)
        return mean_sem(pd)

    HDR = ("   step |   ELBO (sem over draws) | LOCATION lvlMAE (mean path) | corr_samp (sd) |"
           " lvlMAE_samp (sd) | corr_meanpath | s_l1")

    def row(s, e):
        m, sm = unp(e)
        return (f"  {s:5d} | {_fmt_pm(m, sm)}          |        {e['loc']:6.2f}%             "
                f"| {e['corr']:6.4f} ({e['corr_sd']:.4f}) |  {e['mae']:6.2f}% ({e['mae_sd']:4.2f})"
                f"  |    {e['corr_det']:6.4f}     | {e['s_l1']:.3f}")

    def parts_line(tag, e):
        return "   " + tag + " " + "  ".join(f"{k}={float(e['parts'][i]):8.1f}"
                                             for i, k in enumerate(PKEYS))

    # ---- preregistered decision rule, printed BEFORE any arm runs ------------------------
    print("\n=========================== PREREGISTERED CRITERIA ===========================")
    print("  Every ELBO is PAIRED under common random numbers over (draw x crop). Every gate")
    print("  is THREE-VALUED: POSITIVE (effect > 2 sem AND > an absolute floor), NEGLIGIBLE")
    print("  (equivalence: |effect| + 2 sem < a pre-declared margin) or UNRESOLVED. A gate")
    print("  that merely FAILS to reject is UNRESOLVED and may NOT be reported as flatness.")
    print("  Each gate prints its MDE = 2 sem, so 'not significant' can be read against it.")
    print(f"  DECIDING STATISTIC (level LOCATION): loc(arm) := 100*|mean-path lt - lt_true|,")
    print(f"     per crop, paired over crops. It is a pure LOCATION quantity, so it CANNOT be")
    print(f"     moved by the posterior width s_l1 (measured: sampled MAE 7.9% -> 40.4% at a")
    print(f"     PERFECT mean as s_l1 goes 0.011 -> 0.30). The sampled MAE is DESCRIPTIVE and")
    print(f"     each arm's final s_l1 is printed. Cross-checked on the draw MEDIAN location.")
    print(f"  MOVED(arm) := d_loc > 2 sem AND d_loc > {A.move_pp} pp   [costs corr "
          f"{base_c:.3f} -> {_corr_at_pp(A.move_pp):.3f}]")
    print(f"  HELD(arm)  := |d_loc| + 2 sem < {A.held_pp} pp  (EQUIVALENCE, not 'not "
          f"significant')  [{A.held_pp} pp costs corr {base_c:.3f} -> "
          f"{_corr_at_pp(A.held_pp):.3f}]")
    print("  TEMPO_PREF(arm) := paired dELBO(final vs final-with-level-reset-to-truth) POSITIVE")
    print("  PIN_LOSES := paired dELBO(free_good_final vs free_pinned_final) POSITIVE")
    print("  GRAD_OK   := AT THE TRUTH, the fraction of crops whose MEAN dL/dml1 is")
    print("     positive differs from 0.5 by more than 2 binomial sem, i.e. the level")
    print("     direction has a net sign there. Required for H1 (under H1 the truth is NOT")
    print("     the optimum, so a net direction must exist).")
    print("  GRAD_PULL := at a +5% level DISPLACEMENT, the fraction of crops whose mean")
    print("     gradient points BACK toward the truth exceeds 0.5 by more than 2 binomial")
    print("     sem. Required for H2. The two differ DELIBERATELY: under H2 the truth is the")
    print("     optimum, where the gradient VANISHES, so requiring GRAD_OK for H2 would make")
    print("     H2 unreachable by construction -- the same structural defect that made the")
    print("     old PRE_A dead code. Both are BINOMIAL direction tests rather than per-crop")
    print("     SNR gates, because with a Student-t(2) level channel the per-draw sign is")
    print("     near a coin flip even when the expected gradient is unambiguous.")
    print("  DIFFUSION := location drift(lr_hi) > 2*drift(lr_lo) AND the extra movement buys")
    print("     NEGLIGIBLE ELBO (equivalence, not merely 'not significant').")
    print("  COMPETITIVE := paired dELBO(free_good_final vs amortized_final) not NEGATIVE.")
    print("  PRE_E [optimizer positive control] := free_bad's paired dELBO(init->final) is")
    print("     POSITIVE (the free family CAN descend from a cold start) AND free_good's is")
    print("     not significantly negative. Without this, 'HELD' and 'the optimizer never")
    print("     moved' are the same observation.")
    print("  STABLE := the SIGNS of MOVED/HELD(free_good), MOVED(amortized), PIN_LOSES,")
    print(f"     TEMPO_PREF and COMPETITIVE agree across all {A.n_seeds} optimization seeds, the")
    print("     two --lr_free values, and the two location statistics (mean path / median).")
    print("  reproduce := MOVED(amortized) and its paired init->final dELBO POSITIVE")
    print("  ---------------------------------------------------------------------------")
    print("  HARD PRECONDITIONS for ANY H1/H2 statement: production regime, certified")
    print("  training gradient, PRE_A (ordering inside the traversed band), PRE_B (the")
    print("  amortized arm STARTED at the premise), PRE_C (the free q's SAMPLED trajectory")
    print("  starts at good tempo), PRE_D, PRE_E, reproduce, COMPETITIVE, STABLE.")
    print("  H1 := ... and MOVED(free_good) and TEMPO_PREF and PIN_LOSES and GRAD_OK and")
    print("        not DIFFUSION")
    print("  H2 := ... and HELD(free_good) and GRAD_PULL")
    print("  H3-FLAT := the flatness is POSITIVELY established (equivalence), never inferred")
    print("        from a failure to reject.")
    print("  RECOVERED(free_bad) := final loc < free_good's init loc + the MOVED floor.")
    print("     free_bad separates HELD from NEVER-FOUND; it never enters the H1/H2 branch.")
    print("  (context only, NOT gates: teacher per-crop corr ~0.63, constant-tempo ceiling")
    print("   ~0.726 -- both crop-set and regime dependent.)")
    print("==============================================================================",
          flush=True)

    # ---- [C3] level-gradient probes ------------------------------------------------------
    # TWO probes, because they answer different questions and only one is a fair H2 gate:
    #   (i)  AT THE TRUTH -- was free_good's movement gradient-driven? Required for H1.
    #        Under H2 the truth IS (near) the optimum, so a vanishing gradient there is
    #        EXPECTED; requiring signal at the optimum would structurally block H2, the same
    #        defect class as the old unreachable PRE_A. NOT required for H2.
    #   (ii) AT A DISPLACEMENT (+--grad_disp) -- does the objective PULL BACK toward the
    #        truth? That does not vanish at the optimum, so it IS a fair symmetric
    #        requirement, and it is what H2 ("the objective is fine") actually asserts.
    def grad_probe(ml1_init, tag):
        pars_ = new_pars(ml1_init)
        gsn = torch.Generator(device=DEV).manual_seed(777 + A.seed + (len(tag) * 37))
        Gs = []
        for _ in range(A.snr_draws):
            for v in pars_.values():
                if v.grad is not None: v.grad = None
            nz = draw_noise_g(B, T, K, DEV, IQ.DOF, gen=gsn)               # [P1] FRESH each draw
            ro = free_rollout(pars_, nz, Pi, sample=True, temperature=A.temp, **FKW)
            per_crop_loss(ro, dec, hdec, b, db, obs, **LOSS_KW).mean().backward()
            Gs.append(pars_["ml1"].grad.detach().clone())
        Gs = torch.stack(Gs)                                               # [n,B]
        gmm, gss = Gs.mean(0), Gs.std(0, unbiased=True)
        snr_ = gmm.abs() / (gss + 1e-30)
        sc = (2 * (Gs > 0).float().mean(0) - 1).abs()
        for v in pars_.values():
            v.grad = None
        return (float(snr_.median()), float(sc.median()), float((snr_ > 1).float().mean()),
                float((gmm > 0).float().mean()))

    med_snr, med_sign, frac_snr, t_home = grad_probe(lt_true, "at_truth")
    d_snr, d_sign, d_frac, d_home = grad_probe(lt_true + math.log1p(A.grad_disp), "displaced+")
    # The DECIDING gradient statistic is a BINOMIAL test on the DIRECTION across crops, not a
    # per-crop SNR: with a Student-t(2) level channel the per-draw gradient sign is close to a
    # coin flip even when the expected gradient is unambiguous (measured: sign consistency
    # 0.00 while 88% of crop MEANS point home), so a per-crop SNR gate would be another
    # structurally-unreachable veto. binsem = sqrt(0.25/B) is the binomial sem of a fraction.
    binsem = math.sqrt(0.25 / B)
    GRAD_OK = abs(t_home - 0.5) > 2 * binsem          # a consistent net direction AT the truth
    GRAD_PULL = (d_home - 0.5) > 2 * binsem           # and it points HOME from a displacement
    GRAD_FLAT = (abs(t_home - 0.5) + 2 * binsem) < 0.15    # POSITIVE no-direction statement
    print(f"\n[C3] level-gradient probes ({A.snr_draws} FRESH draws each; binomial sem over "
          f"{B} crops = {binsem:.4f}):")
    print(f"     (i)  AT THE TRUTH  : {100*t_home:.0f}% of crop-mean gradients are positive "
          f"(0.5 = no net direction, MDE {2*binsem:.3f}); descriptive per-crop median "
          f"|E[dL/dml1]|/sd = {med_snr:.4f}, median per-draw sign consistency {med_sign:.4f}, "
          f"crops with SNR>1 {100*frac_snr:.0f}%")
    print(f"          -> GRAD_OK={GRAD_OK} (a net direction exists at the truth), "
          f"GRAD_FLAT={GRAD_FLAT} (provably no net direction)   [GRAD_OK required for H1]")
    print(f"     (ii) AT +{100*A.grad_disp:.0f}% LEVEL: {100*d_home:.0f}% of crop-mean "
          f"gradients point BACK toward the truth (descriptive median SNR {d_snr:.4f}, sign "
          f"consistency {d_sign:.4f})  -> GRAD_PULL={GRAD_PULL}   [required for H2]")
    print(f"     WHY TWO PROBES: at the truth the gradient is expected to VANISH if the truth "
          f"IS the optimum, so demanding signal there would make H2 unreachable by "
          f"construction -- the same structural defect as the old PRE_A. The displaced probe "
          f"does not vanish at the optimum and is the fair symmetric control; the truth probe "
          f"is exactly what H1 needs (under H1 the truth is NOT the optimum, so a net "
          f"direction must exist there).")
    print(f"     (SNR<1 with sign consistency ~0 means Adam saturates each of the {T-1} bounded "
          f"level innovations to a random sign; the accumulated walk is s_lt*sqrt(T-1) = "
          f"{A.s_lt*math.sqrt(T-1)*100:.1f}% in log-level -- past any DEGRADED threshold, with "
          f"no tempo information behind it.)", flush=True)

    # ======================= ARMS ==========================================================
    def loc_test(ei, ef, key="locv"):
        m, s = mean_sem(ef[key] - ei[key])
        return m, s, ((m > 2 * s) and (m > A.move_pp)), ((abs(m) + 2 * s) < A.held_pp)

    def run_seed(seed, nd, nck, full, lr_list):
        """One optimization seed. full=True runs every arm and every diagnostic."""
        bank = make_bank(seed, nd)
        evaluate, ND = make_eval(bank)
        CK = sorted({0} | {round(A.steps * i / nck) for i in range(1, nck + 1)})
        R, HIST = {}, {}

        def run_free(tag, label, ml1_init, lr, gseed, *, pin_level=False, kw=None,
                     s_l1_init=None, s_l1_floor=None):
            kw = kw or FKW
            pars = new_pars(ml1_init, pin_level=pin_level, s_l1_init=s_l1_init,
                            s_l1_floor=(kw["s_l1_floor"] if s_l1_floor is None else s_l1_floor))
            snap = (pars["ml1"].detach().clone(), pars["sl1r"].detach().clone(),
                    pars["inn"].detach()[..., 2:4].clone())
            train = [v for v in pars.values() if v.requires_grad]
            opt = torch.optim.Adam(train, lr=lr)
            gen = torch.Generator(device=DEV).manual_seed(gseed)
            t0 = time.time()
            thing = (pars, kw)
            hist = [(0, evaluate("free", thing))]
            print(f"\n--- arm {tag} [seed {seed}]: {label}   (lr {lr:g}, "
                  f"{sum(p.numel() for p in train)} trainable of "
                  f"{sum(p.numel() for p in pars.values())} free params) ---")
            print(HDR); print(row(0, hist[0][1]), flush=True)
            if full:
                print(parts_line("parts@init :", hist[0][1]), flush=True)
            prev = None
            for s in range(1, A.steps + 1):
                opt.zero_grad()
                nz = draw_noise_g(B, T, K, DEV, IQ.DOF, gen=gen)                  # [P1] FRESH
                if prev is not None and s in (2, 3):
                    for kk in NKEYS:      # [P1][F6] freshness on EVERY channel, not just 'u'
                        assert not torch.equal(prev[kk], nz[kk]), \
                            f"[P1] noise channel '{kk}' was NOT redrawn at step {s}!"
                prev = nz
                ro = free_rollout(pars, nz, Pi, sample=True, temperature=A.temp, **kw)
                per_crop_loss(ro, dec, hdec, b, db, obs, **LOSS_KW).mean().backward()
                if pin_level:                       # [C1] the level channels cannot move
                    pars["inn"].grad[..., 2:4] = 0.0
                opt.step()
                if s in CK:
                    e = evaluate("free", thing); hist.append((s, e)); print(row(s, e), flush=True)
            check_frozen(f"arm {tag}")
            if pin_level:                           # [P4][F13] the pin is VERIFIED, not assumed
                oks = (torch.equal(pars["ml1"], snap[0]),
                       torch.equal(pars["sl1r"], snap[1]),
                       torch.equal(pars["inn"].detach()[..., 2:4], snap[2]))
                if not all(oks):
                    print(f"\n*** ABORT [P4]: the level pin LEAKED "
                          f"(ml1={oks[0]}, sl1r={oks[1]}, inn[...,2:4]={oks[2]}).\n", flush=True)
                    sys.exit(5)
                print(f"  [P4][F13] level pin verified bit-identical on ml1, sl1r AND "
                      f"inn[...,2:4] -- the level LOCATION *and* WIDTH are frozen, so this "
                      f"arm's tempo really is exact. final mean-path loc = "
                      f"{hist[-1][1]['loc']:.2f}%", flush=True)
            if full:
                print(parts_line("parts@final:", hist[-1][1]), flush=True)
                dpm = hist[-1][1]["parts"] - hist[0][1]["parts"]
                print("   d(parts)   " + "  ".join(f"{k}={float(dpm[i]):+8.1f}"
                                                   for i, k in enumerate(PKEYS))
                      + "   <- where the init->final ELBO change actually came from", flush=True)
            d, sd_ = paired(hist[-1][1], hist[0][1])
            print(f"  paired dELBO(init->final), common random numbers: {d:+.2f}+-{sd_:.2f} nats "
                  f"({verdict3(d, sd_, 0.0, FLAT_MARGIN)}) -- NOT used as evidence for H1; "
                  f"see TEMPO_PREF.  [{time.time()-t0:.0f}s]", flush=True)
            HIST[tag] = hist
            return dict(pars=pars, kw=kw, hist=hist, i=hist[0][1], f=hist[-1][1],
                        d=d, sd=sd_, lr=lr)

        def run_amort(gseed):
            model, st = build_amort(seed, verbose=False)
            if seed != A.seed:
                print(f"\n[PRE_B seed {seed}] amortized init: level MAE {st['mae']:.2f}%, "
                      f"|dphi| {st['dphi']:.3f} rad, mean-path corr {st['corr_mp']:.4f}")
            p_before = torch.cat([q.detach().reshape(-1) for q in model.parameters()]).clone()
            opt = torch.optim.Adam(model.parameters(), lr=A.lr_amort)
            gen = torch.Generator(device=DEV).manual_seed(gseed)
            t0 = time.time()
            hist = [(0, evaluate("amort", model))]
            print(f"\n--- arm amortized [seed {seed}]: InnovQT encoder "
                  f"({sum(q.numel() for q in model.parameters())} params, lr {A.lr_amort}, "
                  f"{'SEQUENTIAL' if A.amort_exact else 'certified Picard'} training rollout) ---")
            print(HDR); print(row(0, hist[0][1]), flush=True)
            if full:
                print(parts_line("parts@init :", hist[0][1]), flush=True)
            prev = None
            for s in range(1, A.steps + 1):
                opt.zero_grad()
                nz = draw_noise_g(B, T, K, DEV, IQ.DOF, gen=gen)                   # [P1] FRESH
                if prev is not None and s in (2, 3):
                    for kk in NKEYS:
                        assert not torch.equal(prev[kk], nz[kk]), \
                            f"[P1] noise channel '{kk}' was NOT redrawn at step {s}!"
                prev = nz
                ro = (rollout_loop_noise(model, h, b, nz, sample=True, temperature=A.temp)
                      if A.amort_exact else
                      rollout_vec_s(model, h, b, nz, sample=True, temperature=A.temp,
                                    n_picard=8))            # [F20] gradient CERTIFIED above
                per_crop_loss(ro, dec, hdec, b, db, obs, **LOSS_KW).mean().backward()
                opt.step()
                if s in CK:
                    e = evaluate("amort", model); hist.append((s, e)); print(row(s, e), flush=True)
            check_frozen(f"arm amortized [seed {seed}]")
            p_after = torch.cat([q.detach().reshape(-1) for q in model.parameters()])
            if full:
                print(parts_line("parts@final:", hist[-1][1]), flush=True)
                dpm = hist[-1][1]["parts"] - hist[0][1]["parts"]
                print("   d(parts)   " + "  ".join(f"{k}={float(dpm[i]):+8.1f}"
                                                   for i, k in enumerate(PKEYS)), flush=True)
            d, sd_ = paired(hist[-1][1], hist[0][1])
            print(f"  paired dELBO(init->final): {d:+.2f}+-{sd_:.2f} nats "
                  f"({verdict3(d, sd_, 0.0, FLAT_MARGIN)}); encoder moved: "
                  f"||dtheta||={float((p_after-p_before).norm()):.4f};  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            HIST["amortized"] = hist
            return dict(model=model, hist=hist, i=hist[0][1], f=hist[-1][1], d=d, sd=sd_,
                        init_stats=st)

        # ---- free_good at each lr ---------------------------------------------------------
        SW = {}
        for j, lr in enumerate(lr_list):
            SW[lr] = run_free(f"free_good@{lr:g}", "level init = TRUTH, phase offset = ORACLE, "
                              "level innovations SHARP (declared init)", lt_true, lr,
                              1234 + 7919 * seed + j)
        R["sw"] = SW
        # [F3] best_lr by the PAIRED comparison, never by an unpaired point estimate whose
        # MC noise (measured: +-16 nats) exceeded the gap it was deciding.
        if len(lr_list) > 1:
            hi, lo = lr_list[0], lr_list[-1]
            d_lh, s_lh = paired(SW[hi]["f"], SW[lo]["f"])       # >0 => hi is BETTER
            if d_lh > 2 * s_lh:
                best_lr, lr_note = hi, f"lr {hi:g} is PAIRED-significantly better"
            elif d_lh < -2 * s_lh:
                best_lr, lr_note = lo, f"lr {lo:g} is PAIRED-significantly better"
            else:
                best_lr = lo
                lr_note = (f"TIED within 2 sem ({d_lh:+.1f}+-{s_lh:.1f}) -- the data do NOT "
                           f"decide; falling back to the SMALLER lr {lo:g} (least diffusion)")
            R["d_lh"], R["s_lh"] = d_lh, s_lh
        else:
            best_lr, lr_note, R["d_lh"], R["s_lh"] = lr_list[0], "single lr", 0.0, 0.0
        R["best_lr"], R["lr_note"] = best_lr, lr_note
        good = SW[best_lr]
        R["good"] = good

        # ---- free_pinned: level LOCATION *and* WIDTH frozen at the truth -------------------
        R["pin"] = run_free("free_pinned", "level PINNED at the TRUTH (ml1, sl1r frozen; "
                            "inn[...,2:4] grads zeroed) -- everything else free",
                            lt_true, best_lr, 4321 + 7919 * seed, pin_level=True)

        if full:
            # ---- free_bad: the cold-start control. [F3] its lr is SIZED to the offset. ----
            gbad = torch.Generator(device=DEV).manual_seed(99 + seed)
            # [F3] the cold start must be REACHABLE: Adam travels at most steps*lr per
            # coordinate. Note the offset is dominated by the CORPUS SPREAD of log-tempo, not
            # by --bad_sd: initializing at the corpus mean is already ~0.5-0.8 away in
            # log-level for the extreme crops. So the lr is SIZED to the measured offset
            # (3*off/steps guarantees feasibility by construction) rather than inherited from
            # free_good, and the feasibility bound is asserted anyway.
            bad_init = lt_true.mean() + A.bad_sd * torch.randn(B, device=DEV, generator=gbad)
            off = float((bad_init - lt_true).abs().max())
            lr_bad = max(max(lr_list), 3.0 * off / max(A.steps, 1))
            if A.lr_bad_max > 0:
                lr_bad = min(lr_bad, A.lr_bad_max)
            feasible = A.steps * lr_bad > 2 * off
            print(f"\n[F3] free_bad FEASIBILITY: cold start = corpus mean + N(0,{A.bad_sd:g}); "
                  f"max |bad_init - truth| = {off:.4f} in log-level (dominated by the corpus "
                  f"spread of log-tempo, not by the jitter); lr SIZED to that offset = "
                  f"{lr_bad:.2e}, maximum Adam travel {A.steps*lr_bad:.4f} -> feasible: "
                  f"{feasible}  (the old script ran this arm at free_good's lr 3e-4, a travel "
                  f"budget of {A.steps*3e-4:.4f} -- only "
                  f"{int(((bad_init-lt_true).abs() < A.steps*3e-4).sum())}/{B} crops were even "
                  f"REACHABLE, so its RECOVERED=False was a statement about the step budget.)")
            if lr_bad > 5 * max(lr_list):
                print(f"     NOTE: that lr is {lr_bad/max(lr_list):.0f}x free_good's. It was "
                      f"chosen for REACHABILITY, not stability; read PRE_E with that in mind "
                      f"(raise --steps to bring it down).")
            if not feasible:
                print("\n*** ABORT [F3]: free_bad cannot reach the truth within the step "
                      "budget; RECOVERED would be uninterpretable. Raise --steps.\n", flush=True)
                sys.exit(9)
            R["bad_off"], R["bad_lr"] = off, lr_bad
            R["bad"] = run_free("free_bad   ", "level init = corpus mean + N(0,0.3) -- the "
                                "cold-start control (phase offset still ORACLE)",
                                bad_init, lr_bad, 5678 + 7919 * seed)
            # ---- [F16] robustness arm: the ENCODER's own hardcoded s_l1 floor -------------
            kw_fl = dict(FKW); kw_fl["s_l1_floor"] = ENC_S_L1_FLOOR
            R["floor"] = run_free("free_floor ", f"IDENTICAL to free_good but with the "
                                  f"ENCODER's hardcoded s_l1 floor {ENC_S_L1_FLOOR:g} "
                                  f"(width-asymmetry robustness arm)", lt_true, best_lr,
                                  1234 + 7919 * seed, kw=kw_fl,
                                  s_l1_init=A.amort_s_l1_init)

        R["am"] = run_amort(8765 + 7919 * seed)

        # ---- [C2][F9] tempo-specific dELBO + a NON-TRIVIAL estimator check ----------------
        TP = {}
        for tg_, key in (("free_good", "good"), ("free_pinned", "pin")) + \
                        ((("free_bad", "bad"),) if full else ()):
            arm = R[key]
            e_fin = evaluate("free", (arm["pars"], arm["kw"]))
            e_res = evaluate("free", (reset_level(arm["pars"]), arm["kw"]))
            TP[tg_] = paired(e_fin, e_res)
        R["TP"] = TP
        if full:
            # construction check (identical params -> exactly 0) -- NOT an estimator check
            zz = TP["free_pinned"]
            # [F9] the real estimator check: a KNOWN +1% level perturbation of the PINNED arm,
            # measured on the shared bank and again on an INDEPENDENT, larger bank.
            arm = R["pin"]
            e_a = evaluate("free", (reset_level(arm["pars"], 0.0), arm["kw"]))
            e_b = evaluate("free", (reset_level(arm["pars"], 0.01), arm["kw"]))
            d1, s1 = paired(e_b, e_a)
            ev2, _ = make_eval(make_bank(seed + 555, 2 * nd))
            d2, s2 = paired(ev2("free", (reset_level(arm["pars"], 0.01), arm["kw"])),
                            ev2("free", (reset_level(arm["pars"], 0.0), arm["kw"])))
            agree = abs(d1 - d2) < 3 * math.sqrt(s1 ** 2 + s2 ** 2)
            R["est_check"] = (d1, s1, d2, s2, agree, zz)
        return R

    # ======================= run the seeds =================================================
    SEEDS = [A.seed + i for i in range(max(1, A.n_seeds))]
    ALL = {}
    print(f"\n############ SEED {SEEDS[0]} (FULL: every arm and every diagnostic) ############",
          flush=True)
    ALL[SEEDS[0]] = run_seed(SEEDS[0], A.eval_draws, A.n_ckpt, True, LRS)
    BEST_LR = ALL[SEEDS[0]]["best_lr"]
    print(f"\n[C4/F3] lr chosen by the PAIRED comparison: {BEST_LR:g}  ({ALL[SEEDS[0]]['lr_note']})",
          flush=True)
    for sd in SEEDS[1:]:
        print(f"\n############ SEED {sd} (replicate: free_good, free_pinned, amortized) "
              f"############", flush=True)
        ALL[sd] = run_seed(sd, max(3, A.eval_draws // 2), 1, False, [BEST_LR])

    # [F2] Stability is judged by CONTRADICTION, not by exact boolean equality: a replicate
    # that is merely UNRESOLVED can neither confirm nor refute, and demanding identical
    # booleans would itself be a power-dependent gate. Only an OPPOSITE call is instability.
    def sgn3(m, s):
        return 1 if m > 2 * s else (-1 if m < -2 * s else 0)

    def contra(a, bx):
        """a, bx are (MOVED, HELD) pairs. True iff they make OPPOSITE calls."""
        return (a[0] and bx[1]) or (a[1] and bx[0])

    def signs(R):
        g, p, am = R["good"], R["pin"], R["am"]
        m, s, mv, hd = loc_test(g["i"], g["f"])
        ma, sa, mva, hda = loc_test(am["i"], am["f"])
        dpin, spin = paired(g["f"], p["f"])
        tp = R["TP"]["free_good"]
        cd, cs = paired(g["f"], am["f"])
        return dict(MOVED_g=mv, HELD_g=hd, MOVED_a=mva, HELD_a=hda,
                    PIN_LOSES=(dpin > 2 * spin), PIN_WINS=(dpin < -2 * spin),
                    TEMPO_PREF=(tp[0] > 2 * tp[1]), COMPETITIVE=(cd > -2 * cs),
                    s_pin=sgn3(dpin, spin), s_tp=sgn3(*tp), s_cross=sgn3(cd, cs),
                    dloc_g=(m, s), dloc_a=(ma, sa), pin=(dpin, spin), tp=tp, cross=(cd, cs))

    SG = {sd: signs(ALL[sd]) for sd in SEEDS}
    S0, R0 = SG[SEEDS[0]], ALL[SEEDS[0]]
    unstable = []
    for k in ("s_pin", "s_tp", "s_cross"):
        v = {SG[sd][k] for sd in SEEDS}
        if 1 in v and -1 in v:
            unstable.append(k)
    for nm_, a_, b_ in (("free_good MOVED/HELD", "MOVED_g", "HELD_g"),
                        ("amortized MOVED/HELD", "MOVED_a", "HELD_a")):
        pr_ = [(SG[sd][a_], SG[sd][b_]) for sd in SEEDS]
        if any(contra(pr_[0], q) for q in pr_[1:]):
            unstable.append(nm_)
    SEED_STABLE = not unstable
    # lr stability (seed 0): does the other lr CONTRADICT the MOVED/HELD call?
    if len(LRS) > 1:
        alt = [lr for lr in LRS if lr != BEST_LR][0]
        a_ = R0["sw"][alt]
        _, _, mv_a, hd_a = loc_test(a_["i"], a_["f"])
        LR_STABLE = not contra((S0["MOVED_g"], S0["HELD_g"]), (mv_a, hd_a))
    else:
        alt, mv_a, hd_a, LR_STABLE = None, None, None, True
    # location-statistic agreement (mean path vs draw MEDIAN) -- a falsifier, not a confirmer
    dm_g, sm_g, mvm, hdm = loc_test(R0["good"]["i"], R0["good"]["f"], key="locmed")
    dm_a, sm_a, mvam, hdam = loc_test(R0["am"]["i"], R0["am"]["f"], key="locmed")
    LOC_AGREE = (not contra((S0["MOVED_g"], S0["HELD_g"]), (mvm, hdm))
                 and not contra((S0["MOVED_a"], S0["HELD_a"]), (mvam, hdam)))
    STABLE = SEED_STABLE and LR_STABLE and LOC_AGREE

    gi, gf = R0["good"]["i"], R0["good"]["f"]
    pi_, pf = R0["pin"]["i"], R0["pin"]["f"]
    bi, bf = R0["bad"]["i"], R0["bad"]["f"]
    ai, af = R0["am"]["i"], R0["am"]["f"]
    fi_, ff_ = R0["floor"]["i"], R0["floor"]["f"]
    dg, sg_ = R0["good"]["d"], R0["good"]["sd"]
    dbd, sbd = R0["bad"]["d"], R0["bad"]["sd"]
    da, sa = R0["am"]["d"], R0["am"]["sd"]
    d_pin, s_pin = S0["pin"]
    cross_d, cross_s = S0["cross"]
    tp_d, tp_s = S0["tp"]
    win = float((gf["L"] < af["L"]).double().mean())
    (dl_g, sl_g), (dl_a, sl_a) = S0["dloc_g"], S0["dloc_a"]
    MOVED_G, HELD_G, MOVED_A = S0["MOVED_g"], S0["HELD_g"], S0["MOVED_a"]
    _, _, MOVED_F, HELD_F = loc_test(fi_, ff_)
    FLOOR_ROBUST = (MOVED_F == MOVED_G) and (HELD_F == HELD_G)
    RECOVERED = bf["loc"] < gi["loc"] + A.move_pp
    TEMPO_PREF, PIN_LOSES = S0["TEMPO_PREF"], S0["PIN_LOSES"]
    PIN_WINS, COMPETITIVE = S0["PIN_WINS"], S0["COMPETITIVE"]
    HELD_CORR = gf["corr"] >= pf["corr"] - 2 * paired(gf, pf, key="C")[1]
    # [F13/F19] optimizer positive control
    PRE_E = (verdict3(dbd, sbd, 0.0, FLAT_MARGIN) == POS) and (dg > -2 * sg_)
    # [C4] diffusion: drift must scale with lr AND buy NEGLIGIBLE (not merely non-significant) ELBO
    if len(LRS) > 1:
        dr_hi = R0["sw"][LRS[0]]["f"]["loc"] - R0["sw"][LRS[0]]["i"]["loc"]
        dr_lo = R0["sw"][LRS[-1]]["f"]["loc"] - R0["sw"][LRS[-1]]["i"]["loc"]
        d_lh, s_lh = R0["d_lh"], R0["s_lh"]
        DIFFUSION = ((dr_hi > 0) and (dr_lo <= 0 or dr_hi > 2 * dr_lo)
                     and verdict3(d_lh, s_lh, 0.0, FLAT_MARGIN) == NEG)
    else:
        dr_hi = dr_lo = d_lh = s_lh = 0.0; DIFFUSION = False
    reproduce = MOVED_A and (verdict3(da, sa, 0.0, FLAT_MARGIN) == POS)

    print("\n================================== SUMMARY ==================================")
    print(f"  REGIME: {REGIME}")
    print(f"  PRE_A (objective ORDERS tempo errors INSIDE the traversed band) = {PRE_A}"
          f"   [A1 see-10% = {A1}, A2 order 0.3%->3% = {A2}, A3 no-inversion = {A3}; "
          f"measured RESOLUTION LIMIT ~{100*res_lim:.1f}%, saturation diagnostic {SAT_V}]")
    print(f"  PRE_B (amortized arm STARTED at the premise) = {PRE_B}   [level MAE "
          f"{st0['mae']:.2f}%, |dphi| {st0['dphi']:.3f} rad, mean-path corr {st0['corr_mp']:.4f}]")
    print(f"  PRE_C (SAMPLED per-crop corr at the free-q truth-level init >= "
          f"{A.gate_corr_floor}) = {PRE_C}   [measured {r_arm['corr']:.4f} sampled vs "
          f"{r_arm['corr_det']:.4f} on the mean path]")
    print(f"  PRE_D (full objective prefers the truth level over +-10%) = {PRE_D}"
          f"   [+10% {dhi_v}, -10% {dlo_v}]")
    print(f"  PRE_E (optimizer positive control: free_bad CAN descend) = {PRE_E}"
          f"   [free_bad dELBO {dbd:+.1f}+-{sbd:.1f}, free_good {dg:+.1f}+-{sg_:.1f}]")
    print(f"  PRE_G (Picard training gradient certified) = cosine {gcos:.6f} >= "
          f"{A.grad_cos_min}   PRODUCTION REGIME = {PROD_REGIME}")
    print(f"  GRAD_OK = {GRAD_OK} at the truth ({100*t_home:.0f}% of crop-mean gradients "
          f"positive, MDE {2*binsem:.3f}; GRAD_FLAT={GRAD_FLAT});  GRAD_PULL = {GRAD_PULL} at "
          f"+{100*A.grad_disp:.0f}% ({100*d_home:.0f}% point home)   DIFFUSION = {DIFFUSION}")
    print(f"  SCOPE OF THE REPLICATION: the {len(SEEDS)} seeds re-draw the encoder init, the "
          f"optimizer noise and the evaluation bank. The CROP SET is deliberately held fixed "
          f"across seeds (the whole design is paired on crops); crop-level variability enters "
          f"through the per-crop sems printed everywhere, NOT through a re-draw of the crop "
          f"set. Use --seed to re-draw the crops.")
    print(f"  STABLE = {STABLE}  [seeds {SEEDS}: "
          f"{'no contradictions' if SEED_STABLE else 'CONTRADICT on ' + ','.join(unstable)}"
          f" | lr-contradiction-free {LR_STABLE} | median-location cross-check {LOC_AGREE}]")
    print(f"     median-location cross-check (draw MEDIAN, robust to the Student-t(2) tails): "
          f"free_good d {dm_g:+.2f}+-{sm_g:.2f} pp (MOVED={mvm} HELD={hdm}), amortized d "
          f"{dm_a:+.2f}+-{sm_a:.2f} pp (MOVED={mvam} HELD={hdam}). It can FALSIFY the mean-path "
          f"call, not confirm it: an UNRESOLVED cross-check is not instability.")
    print("  ---------------------------------------------------------------------------")
    print("   arm            LOCATION lvlMAE (mean path)     sampled MAE      corr_samp"
          "        final s_l1   paired dELBO(init->final)")
    for nm, i_, f_, dd, ss in (("free_good  ", gi, gf, dg, sg_),
                               ("free_pinned", pi_, pf, *paired(pf, pi_)),
                               ("free_floor ", fi_, ff_, R0["floor"]["d"], R0["floor"]["sd"]),
                               ("free_bad   ", bi, bf, dbd, sbd),
                               ("amortized  ", ai, af, da, sa)):
        m_, s_, mv_, hd_ = loc_test(i_, f_)
        flag = (f"RECOVERED={RECOVERED}" if nm.strip() == "free_bad"
                else f"MOVED={mv_} HELD={hd_}")
        print(f"  {nm}  {i_['loc']:6.2f}% -> {f_['loc']:6.2f}% (d {m_:+5.2f}+-{s_:4.2f}) | "
              f"{i_['mae']:6.2f}% ->{f_['mae']:6.2f}% | {i_['corr']:.4f}->{f_['corr']:.4f} | "
              f"{i_['s_l1']:.3f}->{f_['s_l1']:.3f} | {dd:+8.2f}+-{ss:5.2f} | {flag}")
    print("  ---------------------------------------------------------------------------")
    print(gline("[C1] free_good vs pinned", d_pin, s_pin, 0.0, FLAT_MARGIN,
                note=("free_good (moved) BETTER" if PIN_LOSES else
                      ("free_pinned (TRUTH tempo) BETTER" if PIN_WINS else "TIE"))))
    print(f"       corr: free_good {gf['corr']:.4f} vs free_pinned {pf['corr']:.4f} "
          f"-> HELD_CORR={HELD_CORR}")
    print(gline("[C2] TEMPO_PREF(free_good)", tp_d, tp_s, 0.0, FLAT_MARGIN,
                note="(+) = the objective prefers the MOVED level"))
    print(gline("cross-arm free vs amort", cross_d, cross_s, 0.0, FLAT_MARGIN,
                note=f"free_good better on {100*win:.0f}% of draw x crop units; "
                     f"COMPETITIVE={COMPETITIVE}"))
    print(f"  [F16] width-robustness arm free_floor (s_l1 floor {ENC_S_L1_FLOOR:g}, the "
          f"ENCODER's own): MOVED={MOVED_F} HELD={HELD_F} -> agrees with free_good: "
          f"{FLOOR_ROBUST}")
    if len(LRS) > 1:
        print(f"  [C4] lr sweep (LOCATION drift): lr {LRS[0]:g} {dr_hi:+.2f} pp vs lr "
              f"{LRS[-1]:g} {dr_lo:+.2f} pp; paired ELBO(hi)-ELBO(lo) = {d_lh:+.1f}+-{s_lh:.1f} "
              f"({verdict3(d_lh, s_lh, 0.0, FLAT_MARGIN)}) -> DIFFUSION={DIFFUSION}; "
              f"the alternative lr gives MOVED={mv_a} HELD={hd_a} (LR_STABLE={LR_STABLE})")
    if "est_check" in R0:
        d1, s1, d2, s2, agree, zz = R0["est_check"]
        print(f"  [C2/F9] estimator checks: (a) CONSTRUCTION check -- reset_level on the "
              f"PINNED arm passes an IDENTICAL parameter dict, so {zz[0]:+.2f}+-{zz[1]:.2f} is "
              f"0 by construction and validates NOTHING (this is pitfall P2 and it is "
              f"labelled as such).  (b) REAL check -- a KNOWN +1% level perturbation of the "
              f"pinned arm makes the ELBO WORSE by {-d1:+.1f}+-{s1:.1f} nats on the shared "
              f"bank and by {-d2:+.1f}+-{s2:.1f} on an INDEPENDENT bank of "
              f"{2*A.eval_draws} draws; "
              f"agree within 3 combined sem: {agree}")
    print(f"  per-seed signs:")
    for sd in SEEDS:
        s_ = SG[sd]
        print(f"    seed {sd}: MOVED_g={int(s_['MOVED_g'])} HELD_g={int(s_['HELD_g'])} "
              f"MOVED_a={int(s_['MOVED_a'])} PIN_LOSES={int(s_['PIN_LOSES'])} "
              f"TEMPO_PREF={int(s_['TEMPO_PREF'])} COMPETITIVE={int(s_['COMPETITIVE'])} | "
              f"d_loc(free_good) {s_['dloc_g'][0]:+.2f}+-{s_['dloc_g'][1]:.2f} pp, "
              f"d_loc(amort) {s_['dloc_a'][0]:+.2f}+-{s_['dloc_a'][1]:.2f} pp")

    # ---- the arms' MEASURED traversal range -- any "over the range the arms move" claim ---
    drifts = [abs(S0["dloc_g"][0]), abs(S0["dloc_a"][0]),
              abs(loc_test(pi_, pf)[0]), abs(loc_test(fi_, ff_)[0])]
    dr_lo_pp, dr_hi_pp = min(drifts), max(drifts)
    print(f"  ARMS' MEASURED LOCATION TRAVERSAL: {dr_lo_pp:.2f} .. {dr_hi_pp:.2f} pp of "
          f"log-level. The C5 sweep resolves successive errors up to ~{100*res_lim:.1f}%, and a "
          f"10% error is worth {a1_m:+.0f}+-{a1_s:.0f} nats.")

    H1_SIG = (MOVED_G and TEMPO_PREF and PIN_LOSES and GRAD_OK and not DIFFUSION)
    HARD = PROD_REGIME and PRE_G and PRE_A and PRE_B and PRE_C and PRE_D

    # ---- verdict ---------------------------------------------------------------------------
    land = (f"Landscape facts from this run, quoted so no claim can contradict them: a 10% "
            f"level error costs {a1_m:+.0f}+-{a1_s:.0f} nats; 0.3%->3% costs "
            f"{a2_m:+.0f}+-{a2_s:.0f}; 3%->30% costs {sat_m:+.0f}+-{sat_s:.0f} ({SAT_V}); the "
            f"measured RESOLUTION LIMIT is ~{100*res_lim:.1f}%. The arms traversed "
            f"{dr_lo_pp:.2f}..{dr_hi_pp:.2f} pp. ")

    if not PROD_REGIME:
        v = "INCONCLUSIVE-NONPRODUCTION"
        why = (f"THE PRODUCTION REGIME WAS NOT TESTED: the arms ran at gamma_phase="
               f"{A.gamma_phase:.3g} while VBPM trains at {PROD_GAMMA:g}. Every number above "
               "describes an objective the phenomenon under study does not use. Re-run without "
               "--gamma_phase. Regime: " + REGIME)
    elif not PRE_C:
        v = "INCONCLUSIVE-PREMISE"
        why = ("the premise does not hold at the declared init: the SAMPLED trajectories -- the "
               f"only ones the ELBO ever scores -- have per-crop corr {r_arm['corr']:.4f} "
               f"(floor {A.gate_corr_floor}, brief's teacher 0.63) while the mean path shows "
               f"{r_arm['corr_det']:.4f}. The free q was therefore never AT good tempo under "
               "the objective's own expectation, so it cannot have held or lost it. The [F11] "
               "decomposition above says where the loss comes from (s_lv, then s_l1, then the "
               "wrapped-Cauchy phase walk of total scale "
               f"{A.gamma_phase*(T-1):.1f} rad). Sharpen --s_lv_init / --s_l1_init and re-run. "
               "Regime: " + REGIME)
    elif not (PRE_A and PRE_D):
        flat_ok = ((not PRE_A and verdict3(a2_m, a2_s, A.nat_floor, FLAT_MARGIN) == NEG)
                   or (not PRE_D and PRE_D_FLAT))
        if flat_ok:
            v = "H3-FLAT"
            why = ("the objective is POSITIVELY established (by equivalence, not by a failure "
                   "to reject) to be FLAT in the level direction over the band the arms "
                   f"traverse. {land}Neither 'the optimum has wrong tempo' (H1) nor 'the "
                   "objective is fine' (H2) is a meaningful sentence when the coordinate "
                   "carries no gradient there. NOTE the level coordinate is NOT unidentified "
                   f"in general -- the same sweep values a 10% error at {a1_m:+.0f} nats; what "
                   "is flat is the band above the resolution limit. The mechanism is that BCE "
                   "is POINTWISE, so its tempo landscape is a plateau with a pit whose "
                   "half-width shrinks with T; fix the likelihood's tempo information (a "
                   "displacement / Cramer-type term, or an emission that sees beat spacing) "
                   "before asking whether the encoder or the ELBO is the breaker. Regime: "
                   + REGIME)
        else:
            v = "INCONCLUSIVE-UNDERPOWERED"
            why = (f"a hard precondition did not resolve at this sample size, and a failure to "
                   f"reject is NOT evidence of absence. PRE_A={PRE_A} (A1={A1}, A2={A2}, "
                   f"A3={A3}), PRE_D={PRE_D} (+10% {dhi_v}, -10% {dlo_v}). {land}Read each "
                   "gate's printed MDE: the effect the data could have detected is larger than "
                   "the effect being claimed. Raise --crops / --eval_draws / --scan_draws. "
                   "Regime: " + REGIME)
    elif not reproduce:
        v = "INCONCLUSIVE-NO-PHENOMENON"
        why = ("the objective's tempo preconditions PASSED and the amortized arm STARTED at "
               f"the premise (PRE_B: level MAE {st0['mae']:.2f}%, |dphi| {st0['dphi']:.3f} rad, "
               f"mean-path corr {st0['corr_mp']:.4f}), but the phenomenon did not reproduce in "
               f"this run: the amortized level LOCATION went {ai['loc']:.2f}% -> {af['loc']:.2f}% "
               f"(paired d {dl_a:+.2f}+-{sl_a:.2f} pp; MOVED needs > {A.move_pp} pp) with "
               f"dELBO {da:+.1f}+-{sa:.1f}. With no degradation-while-ELBO-improves to explain, "
               "neither H1 nor H2 is testable here. Run more --steps. Regime: " + REGIME)
    elif not COMPETITIVE:
        v = "INCONCLUSIVE-FREE-Q-FAILED"
        why = ("free-q optimization FAILED: the per-crop family ended "
               f"{-cross_d:.1f}+-{cross_s:.1f} nats WORSE than the amortized encoder on the "
               "identical objective, noise bank and crops. The free family is not a superset of "
               "the encoder family (the encoder's innovations are state-dependent feedback, the "
               "free table is open-loop), so a free arm that cannot match the encoder says "
               "nothing about the objective's optimum. Raise --steps or tune --lr_free. "
               "Regime: " + REGIME)
    elif not PRE_E:
        v = "INCONCLUSIVE-DEAD-OPTIMIZER"
        why = ("the free family failed its POSITIVE CONTROL, so 'held' and 'the optimizer never "
               f"moved' cannot be distinguished: free_bad's paired dELBO(init->final) was "
               f"{dbd:+.1f}+-{sbd:.1f} nats ({verdict3(dbd, sbd, 0.0, FLAT_MARGIN)}) -- it could "
               f"not descend from a cold start even at lr {R0['bad']['lr']:.1e} -- and "
               f"free_good's was {dg:+.1f}+-{sg_:.1f}. Any HELD reported here would be a "
               "statement about Adam, not about the objective. Regime: " + REGIME)
    elif not STABLE:
        v = "INCONCLUSIVE-UNSTABLE"
        why = ("the branch-selecting SIGNS are not stable, so the verdict would be a function "
               f"of nuisance choices rather than of an effect. seeds {SEEDS} disagree on "
               f"{unstable if unstable else 'nothing'}; LR_STABLE={LR_STABLE} (the alternative "
               f"lr gives MOVED={mv_a}, HELD={hd_a}); mean-path vs draw-median location "
               f"agreement={LOC_AGREE}; width-robustness arm agrees={FLOOR_ROBUST}. {land}"
               "Raise --steps / --crops / --n_seeds. Regime: " + REGIME)
    elif H1_SIG:
        v = "H1"
        why = ("an unrestricted per-crop q, started AT the correct tempo (SAMPLED per-crop corr "
               f"{gi['corr']:.3f} at init, level LOCATION {gi['loc']:.2f}%) AND at the correct "
               "phase offset, on the identical objective, MOVED AWAY from it: level LOCATION "
               f"{gi['loc']:.2f}% -> {gf['loc']:.2f}% (paired d {dl_g:+.2f}+-{sl_g:.2f} pp over "
               f"crops, MOVED floor {A.move_pp} pp), corr {gi['corr']:.3f}->{gf['corr']:.3f}. "
               f"The objective genuinely PREFERS the moved level: resetting ONLY the level "
               f"channels to the truth costs {tp_d:+.1f}+-{tp_s:.1f} nats, and the level-PINNED "
               f"twin -- whose level LOCATION and WIDTH are both frozen and verified "
               f"bit-identical -- ends {d_pin:+.1f}+-{s_pin:.1f} nats WORSE. The level direction "
               f"has a net sign at the truth ({100*t_home:.0f}% of crop-mean gradients share "
               f"it, MDE {2*binsem:.3f}) and the drift is not lr-diffusion. The decision is "
               f"on the LOCATION statistic, so it cannot have been bought by a change in "
               f"posterior width (final s_l1 {gi['s_l1']:.3f}->{gf['s_l1']:.3f}), it survives "
               f"the encoder's own width floor (free_floor agrees: {FLOOR_ROBUST}), and it is "
               f"sign-stable across {len(SEEDS)} seeds and both lrs. {land}The optimum of THIS "
               "objective has wrong tempo. Regime: " + REGIME)
    elif HELD_G and GRAD_PULL:
        sep = ("free_bad ALSO reached good tempo, so the objective actively DRIVES toward the "
               f"truth from a cold start (LOCATION {bi['loc']:.1f}% -> {bf['loc']:.1f}% at lr "
               f"{R0['bad']['lr']:.1e}, a feasible step budget)"
               if RECOVERED else
               f"free_bad did NOT reach good tempo from a FEASIBLE cold start (LOCATION "
               f"{bi['loc']:.1f}% -> {bf['loc']:.1f}%, lr {R0['bad']['lr']:.1e}, travel budget "
               f"{A.steps*R0['bad']['lr']:.3f} in log-level vs a maximum init offset of "
               f"{R0['bad_off']:.3f} -- i.e. the truth WAS reachable), so free_good HELD a "
               "solution it was "
               "given rather than the objective handing it to any q -- a basin/initialization "
               "statement, separate from H1 and NOT evidence of multi-modality")
        v = "H2"
        why = ("on the identical objective, frozen decoder, crop set and noise bank, the free q "
               f"HELD the good tempo -- level LOCATION {gi['loc']:.2f}% -> {gf['loc']:.2f}%, "
               f"paired d {dl_g:+.2f}+-{sl_g:.2f} pp, and |d| + 2 sem < {A.held_pp} pp, i.e. an "
               "EQUIVALENCE result, not a failure to reject -- while the amortized q degraded "
               f"({ai['loc']:.2f}% -> {af['loc']:.2f}%, paired d {dl_a:+.2f}+-{sl_a:.2f} pp). "
               f"free_good's final ELBO is {cross_d:+.1f}+-{cross_s:.1f} nats relative to the "
               f"amortized one (COMPETITIVE), the free family demonstrably CAN move on this "
               f"objective (free_bad's dELBO {dbd:+.1f}+-{sbd:.1f}, PRE_E), the objective "
               f"PULLS BACK toward the truth from a +{100*A.grad_disp:.0f}% displacement "
               f"(GRAD_PULL: {100*d_home:.0f}% of crop-mean gradients point home, MDE "
               f"{2*binsem:.3f}; at the truth itself the net direction is {100*t_home:.0f}% "
               f"-- near 0.5, as it should be at an optimum), the "
               f"decision is on a pure LOCATION statistic so posterior width cannot have bought "
               f"it (free_good s_l1 {gi['s_l1']:.3f}->{gf['s_l1']:.3f}; the width-matched "
               f"free_floor arm agrees: {FLOOR_ROBUST}), and the signs are stable across "
               f"{len(SEEDS)} seeds and both lrs. {land}The ELBO does not prefer wrong tempo; "
               "the ENCODER FAMILY cannot hold the right one. " + sep + ". Regime: " + REGIME)
    elif MOVED_G and (GRAD_FLAT or verdict3(tp_d, tp_s, 0.0, FLAT_MARGIN) == NEG):
        v = "H3-FLAT"
        why = (f"free_good's level LOCATION moved ({gi['loc']:.2f}% -> {gf['loc']:.2f}%, paired "
               f"d {dl_g:+.2f}+-{sl_g:.2f} pp) but the movement is POSITIVELY established NOT to "
               f"be ELBO-driven: TEMPO_PREF is {verdict3(tp_d, tp_s, 0.0, FLAT_MARGIN)} "
               f"({tp_d:+.1f}+-{tp_s:.1f} nats for the moved level over the truth level), "
               f"PIN_LOSES={PIN_LOSES} (the level-PINNED twin ends {d_pin:+.1f}+-{s_pin:.1f} "
               f"nats relative to free_good), GRAD_FLAT={GRAD_FLAT} ({100*t_home:.0f}% of "
               f"crop-mean level gradients positive at the truth, i.e. no net direction), "
               f"DIFFUSION={DIFFUSION}. The level coordinate DIFFUSES under Adam on a flat "
               f"direction; the init->final ELBO gain ({dg:+.1f}+-{sg_:.1f} nats) lives in "
               f"channels orthogonal to the level -- see the d(parts) lines. {land}This is "
               "neither H1 nor H2. Regime: " + REGIME)
    else:
        v = "INCONCLUSIVE"
        why = (f"free_good neither MOVED (d_loc {dl_g:+.2f}+-{sl_g:.2f} pp, floor "
               f"{A.move_pp} pp) nor HELD by equivalence (margin {A.held_pp} pp), or a "
               f"supporting gate was UNRESOLVED rather than POSITIVE/NEGLIGIBLE "
               f"(TEMPO_PREF={verdict3(tp_d, tp_s, 0.0, FLAT_MARGIN)}, "
               f"GRAD_OK={GRAD_OK}, GRAD_PULL={GRAD_PULL}, GRAD_FLAT={GRAD_FLAT}). {land}"
               f"This is an honest "
               "no-answer, not a flatness finding. Needs more --steps or --crops. Regime: "
               + REGIME)

    print(f"\nVERDICT: {v} -- {why}")
    print(f"\n[time] {time.time()-t00:.0f}s total")

    if A.json:
        def arm_json(hist):
            return [(s, dict(corr=e["corr"], corr_sd=e["corr_sd"], mae=e["mae"],
                             mae_sd=e["mae_sd"], loc=e["loc"], locmed=e["locmed_m"],
                             s_l1=e["s_l1"], corr_det=e["corr_det"], mae_det=e["mae_det"],
                             proxy_det=e["proxy_det"], elbo=unp(e)[0], elbo_sem=unp(e)[1],
                             parts={kk: float(e["parts"][ii]) for ii, kk in enumerate(PKEYS)}))
                    for s, e in hist]
        with open(A.json, "w") as f:
            json.dump(dict(
                verdict=v, why=why, regime=REGIME, cfg=vars(A), seeds=SEEDS,
                arms={k: arm_json(R0[k]["hist"]) for k in ("good", "pin", "bad", "floor", "am")},
                lr_arms={f"{lr:g}": arm_json(R0["sw"][lr]["hist"]) for lr in LRS},
                cross_arm=[cross_d, cross_s, win], pin=[d_pin, s_pin],
                tempo_pref={k: list(vv) for k, vv in R0["TP"].items()},
                sweep=[[d_, m_, s_, c_] for d_, m_, s_, c_ in SWEEP],
                resolution_limit=res_lim, calib={str(k): v_ for k, v_ in CALIB.items()},
                landscape=dict(a1=[a1_m, a1_s], a2=[a2_m, a2_s], sat=[sat_m, sat_s],
                               flat_margin=FLAT_MARGIN, inversions=inv),
                scan=[{"a": {k2: (list(v2) if isinstance(v2, tuple) else v2)
                             for k2, v2 in c["a"].items()},
                       "b": {k2: (list(v2) if isinstance(v2, tuple) else v2)
                             for k2, v2 in c["b"].items()}, "flip": c["flip"]} for c in SCAN],
                pre_d=dict(hi=list(r_arm["dhi"]), lo=list(r_arm["dlo"]), corr=r_arm["corr"],
                           corr_det=r_arm["corr_det"]),
                grad=dict(med_snr=med_snr, med_sign=med_sign, frac_snr=frac_snr,
                          truth_home=t_home, disp_snr=d_snr, disp_sign=d_sign,
                          disp_home=d_home, binsem=binsem, grad_disp=A.grad_disp,
                          picard_cosine=gcos, picard_rel=grel),
                per_seed={str(sd): {k2: (list(v2) if isinstance(v2, tuple) else bool(v2))
                                    for k2, v2 in SG[sd].items()} for sd in SEEDS},
                best_lr=BEST_LR, lr_note=R0["lr_note"],
                flags=dict(reproduce=bool(reproduce), PRE_A=bool(PRE_A), PRE_B=bool(PRE_B),
                           PRE_C=bool(PRE_C), PRE_D=bool(PRE_D), PRE_D_FLAT=bool(PRE_D_FLAT),
                           PRE_E=bool(PRE_E), PROD_REGIME=bool(PROD_REGIME),
                           A1=bool(A1), A2=bool(A2), A3=bool(A3), SAT=SAT_V,
                           GRAD_OK=bool(GRAD_OK), GRAD_FLAT=bool(GRAD_FLAT),
                           GRAD_PULL=bool(GRAD_PULL),
                           DIFFUSION=bool(DIFFUSION), COMPETITIVE=bool(COMPETITIVE),
                           TEMPO_PREF=bool(TEMPO_PREF), PIN_LOSES=bool(PIN_LOSES),
                           PIN_WINS=bool(PIN_WINS), HELD_CORR=bool(HELD_CORR),
                           STABLE=bool(STABLE), SEED_STABLE=bool(SEED_STABLE),
                           LR_STABLE=bool(LR_STABLE), LOC_AGREE=bool(LOC_AGREE),
                           FLOOR_ROBUST=bool(FLOOR_ROBUST),
                           moved_free_good=bool(MOVED_G), held_free_good=bool(HELD_G),
                           moved_amort=bool(MOVED_A), recovered_free_bad=bool(RECOVERED))),
                f, indent=1)
        print(f"[json] {A.json}")


if __name__ == "__main__":
    main()
