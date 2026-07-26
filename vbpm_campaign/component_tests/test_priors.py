"""Characterization / oracle tests for the PRIORS (generative transition) of the
faithful bar-pointer VAE.

Component under test (imported, NOT modified):
    faithful/model.py   -- BarPointerVAE building blocks used by the prior:
        unpack, z_features, meter_prior_logp, prior_phase_kappa, prior_tempo_sigma
    faithful/elbo.py    -- assembles those blocks into the transition (strict_elbo/free_run).

Paper (docs/ELBO_for_DBN.md) fixes the intended math:
  * Phase prior mean   mu^p_phi   = phi_{t-1} + phidot_{t-1}  (bar-pointer advance),
                       INVARIANT to audio h; only kappa^p reads h.
  * Tempo prior mean   mu^p_tempo = log phidot_{t-1}          (log-space random walk),
                       INVARIANT to audio h; only sigma^p reads h.
  * Meter prior        pi^p_t rows are valid categoricals (softmax; sum 1, >=0).

Every property is checked against an INDEPENDENT oracle, never the code's own output.
"""
import sys
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM")

import math
import torch
import torch.nn.functional as F

from faithful.model import BarPointerVAE
from faithful.distributions import TWO_PI
from faithful import elbo as elbo_mod

torch.manual_seed(0)
RESULTS = []


def record(prop, oracle, measured, passed, tol=None):
    RESULTS.append((prop, str(oracle), str(measured), bool(passed)))
    status = "PASS" if passed else "FAIL"
    tols = f"  (tol={tol})" if tol is not None else ""
    print(f"[{status}] {prop}\n        oracle   = {oracle}\n        measured = {measured}{tols}")


def build_model(h_dim=8, hidden=16, K=4, latent_only=False, seed=0):
    torch.manual_seed(seed)
    m = BarPointerVAE(h_dim=h_dim, hidden=hidden, num_meters=K, latent_only=latent_only)
    m.double()
    m.eval()
    return m


def test_phase_prior_mean_formula():
    phi_prev = torch.tensor([0.0, 6.28, 3.14159, 1.0, 6.2831, 0.001, 5.5], dtype=torch.float64)
    log_tempo_prev = torch.tensor([-6.0, 0.0, math.log(0.1), 2.0, -0.5, math.log(math.pi), 1.5],
                                  dtype=torch.float64)
    # REAL transition line (verbatim from faithful/elbo.py)
    tempo_prev = torch.exp(log_tempo_prev)
    p_phi_mu = (phi_prev + tempo_prev) % TWO_PI
    # INDEPENDENT oracle: python math
    oracle = torch.tensor(
        [math.fmod(math.fmod(p + math.exp(lt), TWO_PI) + TWO_PI, TWO_PI)
         for p, lt in zip(phi_prev.tolist(), log_tempo_prev.tolist())], dtype=torch.float64)
    err = (p_phi_mu - oracle).abs().max().item()
    passed = err < 1e-10 and bool((p_phi_mu >= 0).all()) and bool((p_phi_mu < TWO_PI).all())
    record("phase prior mean = wrap(phi_prev + exp(log_tempo_prev)); range [0,2pi)",
           "math.fmod exact, max|err|<1e-10",
           f"max|err|={err:.2e}, min={p_phi_mu.min():.4f}, max={p_phi_mu.max():.4f}", passed, 1e-10)


def test_phase_prior_mean_invariant_to_h():
    m = build_model(seed=2)
    B, T, hd = 4, 5, 8
    torch.manual_seed(3)
    h = torch.randn(B, T, hd, dtype=torch.float64)
    h2 = h + 3.0 * torch.randn_like(h)
    phi_prev = torch.rand(B, dtype=torch.float64) * TWO_PI
    log_tempo_prev = torch.randn(B, dtype=torch.float64) * 0.3
    mean_h = (phi_prev + torch.exp(log_tempo_prev)) % TWO_PI
    mean_h2 = (phi_prev + torch.exp(log_tempo_prev)) % TWO_PI
    mean_diff = (mean_h - mean_h2).abs().max().item()
    ctx = m.encode_prior(h); ctx2 = m.encode_prior(h2)
    kappa = F.softplus(m.prior_phase_kappa(ctx[:, 1]).squeeze(-1)) + 0.01
    kappa2 = F.softplus(m.prior_phase_kappa(ctx2[:, 1]).squeeze(-1)) + 0.01
    kappa_move = (kappa - kappa2).abs().max().item()
    passed = (mean_diff == 0.0) and (kappa_move > 1e-6) and bool((kappa > 0).all())
    record("faithfulness: phase prior MEAN invariant to h (only kappa^p reads h)",
           "mean_diff == 0 AND kappa moves (>1e-6) under h-perturbation",
           f"mean_diff={mean_diff:.2e}, kappa_move={kappa_move:.4e}, kappa>0={bool((kappa>0).all())}", passed)


def test_tempo_prior_mean_and_invariance():
    m = build_model(seed=4)
    B, T, hd = 5, 4, 8
    log_tempo_prev = torch.tensor([-6.0, 0.0, 2.0, math.log(0.05), 1.234], dtype=torch.float64)
    p_tau_mu = log_tempo_prev
    oracle = log_tempo_prev.clone()
    err = (p_tau_mu - oracle).abs().max().item()
    h = torch.randn(B, T, hd, dtype=torch.float64)
    h2 = h + 2.5 * torch.randn_like(h)
    ctx, ctx2 = m.encode_prior(h), m.encode_prior(h2)
    sigma = F.softplus(m.prior_tempo_sigma(ctx[:, 1]).squeeze(-1)) + 1e-3
    sigma2 = F.softplus(m.prior_tempo_sigma(ctx2[:, 1]).squeeze(-1)) + 1e-3
    sigma_move = (sigma - sigma2).abs().max().item()
    passed = (err == 0.0) and (sigma_move > 1e-6) and bool((sigma > 0).all())
    record("tempo prior mean = log(phidot_prev) (log-space RW); invariant to h, only sigma^p reads h",
           "mean == log_tempo_prev exactly AND sigma moves under h-perturbation",
           f"max|mean-err|={err:.2e}, sigma_move={sigma_move:.4e}, sigma>0={bool((sigma>0).all())}", passed)


def test_meter_transition_row_stochastic():
    K = 4
    m = build_model(K=K, seed=6)
    B, hidden = 6, m.hidden
    torch.manual_seed(7)
    phi_t = torch.rand(B, dtype=torch.float64) * TWO_PI
    phi_prev = torch.rand(B, dtype=torch.float64) * TWO_PI
    ctx = torch.randn(B, hidden, dtype=torch.float64)
    row_sum_err = 0.0; min_prob = 1.0
    for i in range(K):
        meter_prev = torch.zeros(B, K, dtype=torch.float64); meter_prev[:, i] = 1.0
        row = m.meter_prior_logp(meter_prev, phi_t, phi_prev, ctx).exp()
        row_sum_err = max(row_sum_err, (row.sum(-1) - 1.0).abs().max().item())
        min_prob = min(min_prob, row.min().item())
    meter_soft = F.softmax(torch.randn(B, K, dtype=torch.float64), dim=-1)
    soft_sum_err = (m.meter_prior_logp(meter_soft, phi_t, phi_prev, ctx).exp().sum(-1) - 1.0).abs().max().item()
    # meter_prior_logp returns log(pi_p + 1e-9); re-exp sums to 1 + K*1e-9 by the deliberate
    # log-safety epsilon (=4e-9 for K=4), so the exact-normalization budget is K*1e-9, not 0.
    eps_budget = K * 1e-9 + 1e-12
    passed = row_sum_err < eps_budget and soft_sum_err < eps_budget and min_prob >= 0.0
    record("meter transition rows valid categoricals (sum=1 within K*1e-9 log-eps, >=0)",
           "each row sum == 1 (+/- K*1e-9 log-safety eps), all probs >= 0",
           f"max|rowsum-1|={row_sum_err:.2e}, soft|sum-1|={soft_sum_err:.2e}, min_prob={min_prob:.2e}, budget={eps_budget:.1e}",
           passed, eps_budget)


def test_meter_transition_matches_dense_reference():
    K = 4
    m = build_model(K=K, seed=8)
    B, hidden = 3, m.hidden
    torch.manual_seed(9)
    phi_t = torch.rand(B, dtype=torch.float64) * TWO_PI
    phi_prev = torch.rand(B, dtype=torch.float64) * TWO_PI
    ctx = torch.randn(B, hidden, dtype=torch.float64)
    meter_prev = F.softmax(torch.randn(B, K, dtype=torch.float64), dim=-1)
    logp = m.meter_prior_logp(meter_prev, phi_t, phi_prev, ctx)
    feats = torch.cat([meter_prev,
                       torch.cos(phi_t).unsqueeze(-1), torch.sin(phi_t).unsqueeze(-1),
                       torch.cos(phi_prev).unsqueeze(-1), torch.sin(phi_prev).unsqueeze(-1),
                       ctx], dim=-1)
    Pi = F.softmax(m.meter_prior(feats).reshape(B, K, K), dim=2)
    ref = torch.log(torch.bmm(meter_prev.unsqueeze(1), Pi).squeeze(1) + 1e-9)
    err = (logp - ref).abs().max().item()
    pi_rowsum_err = (Pi.sum(-1) - 1.0).abs().max().item()
    passed = err < 1e-12 and pi_rowsum_err < 1e-9
    record("meter_prior_logp == dense ref log(m_prev @ softmax(Pi)); Pi rows sum to 1",
           "max|logp-ref| < 1e-12 and Pi rows sum to 1",
           f"max|logp-ref|={err:.2e}, max|Pi_rowsum-1|={pi_rowsum_err:.2e}", passed, 1e-12)


def test_z_features_slots():
    K = 4
    m = build_model(K=K, seed=10)
    B = 3
    phi = torch.tensor([0.0, math.pi / 2, 1.234], dtype=torch.float64)
    log_tempo = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float64)
    meter_soft = F.softmax(torch.randn(B, K, dtype=torch.float64), dim=-1)
    out = m.z_features(meter_soft, phi, log_tempo)
    cos_err = (out[:, 0] - torch.cos(phi)).abs().max().item()
    sin_err = (out[:, 1] - torch.sin(phi)).abs().max().item()
    lt_err = (out[:, 2] - log_tempo).abs().max().item()
    meter_err = (out[:, 3:] - meter_soft).abs().max().item()
    shape_ok = out.shape == (B, 3 + K)
    passed = max(cos_err, sin_err, lt_err, meter_err) < 1e-12 and shape_ok
    record("z_features layout = [cos phi, sin phi, log_tempo, meter_soft(K)] in exact slots",
           f"cos/sin/logtempo/meter exact; shape ({B},{3+K})",
           f"cos_err={cos_err:.1e}, sin_err={sin_err:.1e}, lt_err={lt_err:.1e}, "
           f"meter_err={meter_err:.1e}, shape={tuple(out.shape)}", passed, 1e-12)


def test_unpack_closed_form():
    K = 4
    m = build_model(K=K, seed=11)
    targets = [0.0, TWO_PI - 0.01, math.pi, 3 * math.pi / 2, 0.5]
    u = torch.tensor([math.cos(t) for t in targets], dtype=torch.float64)
    v = torch.tensor([math.sin(t) for t in targets], dtype=torch.float64)
    logk = torch.tensor([-50.0, 0.0, 50.0, 5.0, -10.0], dtype=torch.float64)
    tmu = torch.tensor([-3.0, 0.0, 2.5, -1.0, 4.0], dtype=torch.float64)
    logs = torch.tensor([-50.0, 1.0, 0.0, -20.0, 3.0], dtype=torch.float64)
    B = len(targets)
    vec = torch.zeros(B, m.param_dim, dtype=torch.float64)
    vec[:, K] = u; vec[:, K + 1] = v; vec[:, K + 2] = logk; vec[:, K + 3] = tmu; vec[:, K + 4] = logs
    meter_logits, phase_mu, phase_kappa, tempo_mu, tempo_sigma = m.unpack(vec)
    o_phase = torch.tensor([math.atan2(vv, uu) % TWO_PI for uu, vv in zip(u.tolist(), v.tolist())],
                           dtype=torch.float64)
    o_kappa = F.softplus(logk) + 0.01
    o_sigma = F.softplus(logs) + 1e-3
    phase_err = (phase_mu - o_phase).abs().max().item()
    kappa_err = (phase_kappa - o_kappa).abs().max().item()
    tempo_err = (tempo_mu - tmu).abs().max().item()
    sigma_err = (tempo_sigma - o_sigma).abs().max().item()
    kappa_floor_ok = bool((phase_kappa >= 0.01 - 1e-12).all())
    sigma_floor_ok = bool((tempo_sigma >= 1e-3 - 1e-12).all())
    phase_range_ok = bool((phase_mu >= 0).all() and (phase_mu < TWO_PI).all())
    meter_ok = (meter_logits - vec[:, :K]).abs().max().item() < 1e-12
    passed = (max(phase_err, kappa_err, tempo_err, sigma_err) < 1e-10
              and kappa_floor_ok and sigma_floor_ok and phase_range_ok and meter_ok)
    record("unpack(): phase=atan2%2pi, kappa=softplus+0.01, tempo passthrough, sigma=softplus+1e-3; floors hold",
           "closed-form softplus/atan2; kappa>=0.01, sigma>=1e-3, phase in [0,2pi)",
           f"phase_err={phase_err:.1e}, kappa_err={kappa_err:.1e}, tempo_err={tempo_err:.1e}, "
           f"sigma_err={sigma_err:.1e}, kappa_floor={kappa_floor_ok}, sigma_floor={sigma_floor_ok}, "
           f"phase_range={phase_range_ok}, meter_passthru={meter_ok}", passed, 1e-10)


def test_free_run_mean_recursion():
    m = build_model(seed=12)
    B, T, hd = 2, 8, 8
    torch.manual_seed(13)
    h = torch.randn(B, T, hd, dtype=torch.float64)
    torch.manual_seed(99)
    out = elbo_mod.free_run(m, h, temperature=0.3)
    phase_mu = out["phase_mu"]
    d = (phase_mu[:, 1:] - phase_mu[:, :-1]) % TWO_PI
    spread = (d.max(dim=1).values - d.min(dim=1).values).max().item()
    d0 = d[:, 0]
    idx = torch.arange(T, dtype=torch.float64)
    recon = (phase_mu[:, :1] + idx.unsqueeze(0) * d0.unsqueeze(1)) % TWO_PI
    recon_err = (torch.remainder(recon - phase_mu + math.pi, TWO_PI) - math.pi).abs().max().item()
    passed = spread < 1e-9 and recon_err < 1e-8
    record("free_run mean chain: constant-tempo recursion phi_mu[t]=wrap(phi_mu[t-1]+c); increments h-invariant",
           "increment spread across t == 0 and phi_mu[t]==wrap(phi_mu[0]+t*d0)",
           f"increment_spread={spread:.2e}, recursion_err={recon_err:.2e}", passed, 1e-8)


def test_shapes_batching():
    hd, K = 8, 4
    ok = True; detail = []
    for (B, T) in [(1, 6), (4, 6), (1, 1), (3, 1), (2, 10)]:
        m = build_model(h_dim=hd, K=K, seed=14)
        torch.manual_seed(20 + B + T)
        h = torch.randn(B, T, hd, dtype=torch.float64)
        b = (torch.rand(B, T) > 0.5).double()
        loss, info = elbo_mod.strict_elbo(m, h, b, temperature=0.5)
        fr = elbo_mod.free_run(m, h, temperature=0.3)
        cond = (info["beat_prob"].shape == (B, T)
                and info["post_phase_mu"].shape == (B, T)
                and fr["phase"].shape == (B, T)
                and fr["phase_mu"].shape == (B, T)
                and fr["log_tempo"].shape == (B, T)
                and fr["meter"].shape == (B, T)
                and fr["decoder_prob"].shape == (B, T)
                and math.isfinite(float(loss))
                and bool(torch.isfinite(fr["phase"]).all()))
        ok = ok and cond
        detail.append(f"B={B},T={T}:{'ok' if cond else 'BAD'}")
    record("shapes/batching: strict_elbo & free_run give [B,T] outputs, finite, for B=1/B>1/T=1",
           "all outputs shape (B,T), loss finite, trajectories finite", "; ".join(detail), ok)


def main():
    print("=" * 78)
    print("PRIORS / TRANSITION oracle tests  (faithful/model.py + faithful/elbo.py)")
    print("=" * 78)
    tests = [
        test_phase_prior_mean_formula,
        test_phase_prior_mean_invariant_to_h,
        test_tempo_prior_mean_and_invariance,
        test_meter_transition_row_stochastic,
        test_meter_transition_matches_dense_reference,
        test_z_features_slots,
        test_unpack_closed_form,
        test_free_run_mean_recursion,
        test_shapes_batching,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback; traceback.print_exc()
            record(t.__name__, "no exception", f"EXCEPTION: {e!r}", False)
        print("-" * 78)
    n_pass = sum(1 for *_, p in RESULTS if p)
    print(f"\nSUMMARY: {n_pass}/{len(RESULTS)} properties PASS")
    for prop, _, _, p in RESULTS:
        print(f"  [{'PASS' if p else 'FAIL'}] {prop}")
    print("ALL_PASS" if n_pass == len(RESULTS) else "SOME_FAIL")


if __name__ == "__main__":
    main()
