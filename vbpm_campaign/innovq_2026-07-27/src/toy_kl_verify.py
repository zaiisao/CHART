"""INNOVATION-SPACE POSTERIOR -- toy verification of the KL accounting (SPEC.md part b).

Claims verified here (all SIMULATED -- no training; seeds stated inline):
 V1 EXACTNESS   with q sharing the prior recursion, pm_common.elbo_run's certified per-step
                phase accounting kl_wc(q_mu, rho_q, p_mu, rho_p) collapses EXACTLY to the
                innovation form kl_wc(mu_eps_t, rho_q, 0, rho_p) (+ the t=1 term), because
                q_mu - p_mu = mu_eps_t by construction.  Checked by feeding elbo_run
                (q_mode="free", sample=False, prior_mode="physical") the innovation-rollout
                means and comparing its kl_phase to the closed-form innovation sum.
 V2 MC=CLOSED   the single-sample MC estimator (same family as the certified elbo) of the
                per-step innovation KL converges to the closed form kl_wrapped_cauchy.
 V3 STRUCTURE   under SAMPLING, an absolute-mean (pinned/free) q pays the analytic floor
                (min_r 2log(2+r)-log(4r) = log 2 = 0.693 nats/frame even with a PERFECT mean
                trajectory) because the prior mean is built from the sampled prev state while
                q's mean is fixed; the innovation q pays exactly the closed innovation KL
                (0 when mu_eps=0, rho_q=rho_p) because its conditional mean moves WITH the
                sampled prev state.  Executed contrast on identical nominal trajectories.
 V4 MAGNITUDES  per-crop (T=256) innovation kl_phase on the REAL train crops for: truth
                innovations, raw PF teacher, smoothed PF teacher (window sweep) + corr-to-
                truth of each target; rho_q grid; s_phi saturation cap; t=1 KLs; tempo
                (log-level) innovation KL magnitudes via the certified MC estimator.
"""
import json, math, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P
from m2_targets import pf_targets
from vbpm.distributions import kl_wrapped_cauchy, TWO_PI

dev = "cuda:0"
torch.manual_seed(0)
OUT = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq"
RHO_P = math.exp(-P.PHYS["gamma_phase"])          # 0.99944515
GAMMA_P = P.PHYS["gamma_phase"]

def wrap(x): return (x + math.pi) % TWO_PI - math.pi

def logit(p): return math.log(p / (1 - p))

def softplus_inv(y): return math.log(math.expm1(y))

model = P.load_model(dev)
tr = P.build_crops(P.load_songs("train"), n_per_song=2, seed=0, dev=dev)
PF = pf_targets("train", 2, 0, 256, lt_src="phase", dev=dev)
dec, hdec = P.new_decoders(dev)
B, T = tr["phi"].shape
print(f"crops {B}x{T}  rho_p={RHO_P:.6f}", flush=True)

def smooth_track(phi, win):
    d = wrap(phi[:, 1:] - phi[:, :-1])
    k = torch.ones(1, 1, win, device=phi.device) / win
    ds = F.conv1d(F.pad(d.unsqueeze(1), (win // 2, win // 2), mode="replicate"), k).squeeze(1)
    out = torch.cat([phi[:, :1], phi[:, :1] + torch.cumsum(ds, 1)], 1)
    lt = torch.log(ds.clamp(1e-4, 1.0))
    lt = torch.cat([lt[:, :1], lt], 1)[:, :phi.shape[1]]
    return out % TWO_PI, lt

def innovations(phi, lt):
    """mu_eps_t = wrap(phi_t - (phi_{t-1} + exp(lt_{t-1}))), t=1..T-1.  float64."""
    pred = phi[:, :-1].double() + torch.exp(lt[:, :-1].double().clamp(-12, 6))
    return wrap(phi[:, 1:].double() - pred)

def kl_innov_closed(mu_eps, rho_q, rho_p=RHO_P):
    """closed-form per-step innovation KL, float64. mu_eps [B,T-1]."""
    rq = torch.as_tensor(rho_q, dtype=torch.float64, device=mu_eps.device)
    rp = torch.tensor(min(rho_p, 1 - 1e-6), dtype=torch.float64, device=mu_eps.device)
    num = 1 - 2 * rq * rp * torch.cos(mu_eps) + (rq * rp) ** 2
    return torch.log(num / ((1 - rq ** 2) * (1 - rp ** 2)))

def kl_t1_phase(rho_q1):
    """t=1: q = WC(mu, rho_q1) vs physical init WC(pi, 1e-6) -- mu-independent (rp~0)."""
    rq = torch.tensor(rho_q1, dtype=torch.float64)
    rp = torch.tensor(1e-6, dtype=torch.float64)
    num = 1 - 2 * rq * rp + (rq * rp) ** 2     # worst case cos=1
    return float(torch.log(num / ((1 - rq ** 2) * (1 - rp ** 2))))

R = {}

# ---------------- V1: exactness vs elbo_run's certified accounting -----------------------
def make_free(phi_traj, lt_traj, rho_q, rho_q1=0.9):
    rr = torch.full((B, T), logit(min(rho_q, 1 - 2e-4) / (1 - 1e-4)), device=dev)
    rr[:, 0] = logit(rho_q1 / (1 - 1e-4))
    return dict(mlog=F.one_hot(tr["m"], model.K).float() * 10.0,
                phi=phi_traj.float(), rho_raw=rr,
                lv=lt_traj.float(),
                slv_raw=torch.full((B, T), softplus_inv(P.PHYS["t_scale"] - 1e-3 + 1e-6), device=dev),
                dv=torch.zeros(B, T, device=dev),
                sdv_raw=torch.full((B, T), -20.0, device=dev))

for name, (phi_t, lt_t) in dict(
        pf_raw=(PF["phi"], PF["lt"]),
        pf_sm51=smooth_track(PF["phi"], 51)).items():
    free = make_free(phi_t, lt_t, RHO_P)
    with torch.no_grad():
        _, info = P.elbo_run(model, tr, dec, hdec, q_mode="free", prior_mode="physical",
                             free=free, sample=False, beta=1.0)
    # closed-form innovation accountant on the SAME trajectory (float64)
    mu_eps = innovations(phi_t.float(), lt_t.float())
    kl_closed = kl_innov_closed(mu_eps, RHO_P).sum(1).mean().item() + kl_t1_phase(0.9)
    R[f"V1_{name}"] = dict(elbo_run_kl_phase=info["kl_phase"], innovation_closed=kl_closed,
                           abs_diff=abs(info["kl_phase"] - kl_closed),
                           rel_diff=abs(info["kl_phase"] - kl_closed) / max(kl_closed, 1e-9))
    print(f"V1[{name}] elbo_run={info['kl_phase']:.3f}  closed={kl_closed:.3f}  "
          f"rel_diff={R[f'V1_{name}']['rel_diff']:.2e}", flush=True)

# ---------------- V2: single-sample MC == closed form ------------------------------------
def wc_logpdf(x, mu, rho):
    rho = torch.as_tensor(rho, dtype=torch.float64, device=x.device)
    return torch.log((1 - rho ** 2) / (TWO_PI * (1 + rho ** 2 - 2 * rho * torch.cos(x - mu))))

g = torch.Generator(device=dev).manual_seed(1)
V2 = []
for mu_e, rq in [(0.0, RHO_P), (0.005, RHO_P), (0.005, 0.995), (0.05, RHO_P), (0.0005, 0.9995)]:
    n = 200000
    gam = -math.log(rq)
    u = torch.rand(n, generator=g, device=dev, dtype=torch.float64)
    eps = mu_e + gam * torch.tan(math.pi * (u - 0.5))
    mc = (wc_logpdf(eps, torch.tensor(mu_e, dtype=torch.float64, device=dev), rq)
          - wc_logpdf(eps, torch.tensor(0.0, dtype=torch.float64, device=dev), RHO_P))
    cf = kl_innov_closed(torch.tensor([[mu_e]], dtype=torch.float64, device=dev), rq).item()
    V2.append(dict(mu_eps=mu_e, rho_q=rq, mc=float(mc.mean()),
                   mc_se=float(mc.std() / math.sqrt(n)), closed=cf))
    print(f"V2 mu={mu_e:7.4f} rho_q={rq:.5f}  MC={V2[-1]['mc']:.4f}+-{V2[-1]['mc_se']:.4f}  "
          f"closed={cf:.4f}", flush=True)
R["V2_mc_vs_closed"] = V2

# ---------------- V3: sampling contrast -- absolute-mean q floor vs innovation q ---------
# perfect nominal trajectory = a prior-mean rollout (zero innovations) from truth init.
with torch.no_grad():
    phi_perf = torch.empty(B, T, device=dev); phi_perf[:, 0] = tr["phi"][:, 0]
    lt_perf = tr["lt"][:, :1].expand(B, T).contiguous()
    for t in range(1, T):
        phi_perf[:, t] = (phi_perf[:, t - 1] + torch.exp(lt_perf[:, t - 1].clamp(-12, 6))) % TWO_PI
V3 = {}
for rq_name, rq in [("rho_p", RHO_P), ("rho_opt_pinned", math.exp(-GAMMA_P / 2))]:
    # (a) absolute-mean q, sample=True through the certified estimator, 5 seeds
    kls = []
    for s in range(5):
        torch.manual_seed(100 + s)
        free = make_free(phi_perf, lt_perf, rq)
        with torch.no_grad():
            _, info = P.elbo_run(model, tr, dec, hdec, q_mode="free", prior_mode="physical",
                                 free=free, sample=True, beta=1.0)
        kls.append(info["kl_phase"])
    # (b) innovation q on the same nominal trajectory: mu_eps = 0 -> closed value, exact
    #     (per-step KL is sample-path independent because q's conditional mean tracks the
    #      sampled prev state); per-frame floor formula for the pinned case:
    r = -math.log(rq) / GAMMA_P if rq < 1 else 0.0
    floor = 2 * math.log(2 + r) - math.log(4 * r) if r > 0 else float("nan")
    innov_kl = kl_innov_closed(torch.zeros(B, T - 1, dtype=torch.float64, device=dev), rq
                               ).sum(1).mean().item() + kl_t1_phase(0.9)
    V3[rq_name] = dict(rho_q=rq, pinned_sampled_kl_phase=float(np.mean(kls)),
                       pinned_sampled_sd=float(np.std(kls)),
                       pinned_perframe=float(np.mean(kls)) / (T - 1),
                       analytic_floor_perframe=floor,
                       innovation_kl_phase=innov_kl, innovation_perframe=innov_kl / (T - 1))
    print(f"V3[{rq_name}] pinned-sampled={np.mean(kls):8.1f}/crop ({np.mean(kls)/(T-1):.3f}/fr, "
          f"analytic floor {floor:.3f}/fr) vs innovation={innov_kl:.2f}/crop", flush=True)
R["V3_sampling_contrast"] = V3

# ---------------- V4: magnitudes on real crops -------------------------------------------
def corr_truth(phi): return float(torch.abs(torch.exp(1j * (phi - tr["phi"]))).mean(1).mean() * 0
                                  ) if False else float(
    torch.abs(torch.exp(1j * (phi.double() - tr["phi"].double())).mean(1)).mean())

rows = []
cands = {"truth": (tr["phi"], tr["lt"]), "pf_raw": (PF["phi"], PF["lt"])}
for w in (51, 101, 201, 401):
    cands[f"pf_sm{w}"] = smooth_track(PF["phi"], w)
for name, (phi_t, lt_t) in cands.items():
    mu = innovations(phi_t.float(), lt_t.float())
    row = dict(name=name, corr_truth=corr_truth(phi_t),
               mean_abs_innov=float(mu.abs().mean()),
               p99_innov=float(torch.quantile(mu.abs().flatten(), 0.99)))
    for rq in (RHO_P, 0.999, 0.9995, 0.9998):
        row[f"kl_crop_rq{rq:.5f}"] = float(kl_innov_closed(mu, rq).sum(1).mean())
    row["kl_crop_best"] = min(v for k, v in row.items() if k.startswith("kl_crop_rq"))
    rows.append(row)
    print(f"V4[{name:8s}] corr={row['corr_truth']:.4f} |innov|={row['mean_abs_innov']:.5f} "
          f"kl/crop@rho_p={row[f'kl_crop_rq{RHO_P:.5f}']:8.1f}  best={row['kl_crop_best']:8.1f}",
          flush=True)
R["V4_targets"] = rows
# s_phi saturation cap + t=1 + tempo-innovation KL
R["V4_sphi_cap"] = dict(
    s_phi=0.05, kl_crop_all_saturated=float(kl_innov_closed(
        torch.full((1, T - 1), 0.05, dtype=torch.float64, device=dev), RHO_P).sum()))
R["V4_t1"] = dict(phase_rq09=kl_t1_phase(0.9), phase_rq05=kl_t1_phase(0.5),
                  meter_max=math.log(4))
# tempo level innovation: certified MC estimator kl_t_mc, q Student-t(dof_p, mu_e, s) vs prior
dof = torch.tensor(P.PHYS["t_dof"], device=dev)
sp = torch.tensor(P.PHYS["t_scale"], device=dev)
V4t = []
for mu_e, sq in [(0.0, float(sp)), (float(sp), float(sp)), (0.0, 2 * float(sp))]:
    n = 200000
    torch.manual_seed(7)
    z = torch.distributions.StudentT(dof, torch.full((n,), mu_e, device=dev),
                                     torch.full((n,), sq, device=dev)).sample()
    mc = P.kl_t_mc(dof, torch.full((n,), mu_e, device=dev), torch.full((n,), sq, device=dev),
                   dof, torch.zeros(n, device=dev), sp.expand(n), z)
    V4t.append(dict(mu_eps_lt=mu_e, s_q=sq, kl_perframe=float(mc.mean()),
                    se=float(mc.std() / math.sqrt(n)), kl_crop=float(mc.mean()) * (T - 1)))
    print(f"V4t mu={mu_e:.5f} sq={sq:.5f}  KL={V4t[-1]['kl_perframe']:.4f}/fr "
          f"({V4t[-1]['kl_crop']:.1f}/crop)", flush=True)
R["V4_tempo_innov"] = V4t

json.dump(R, open(f"{OUT}/toy_kl_verify.json", "w"), indent=1, default=float)
print("WROTE toy_kl_verify.json", flush=True)
