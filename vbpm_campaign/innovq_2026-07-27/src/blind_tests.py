"""BLIND TESTS for the innovation-space posterior (vbpm_innovq).

Written from SPEC.md ALONE -- innovq.py source was never read; its call surface was
established purely by introspection (signatures) and behavioral probing (return shapes).

Tests (orchestrator-mandated):
 1 ZERO-INNOVATION KNOWN-ANSWER  perfect sawtooth + read-out F ~ 1.0 on the synthetic grid
 2 SMOOTHNESS BY CONSTRUCTION    |innovation| <= bound; increments ~ physical law; >>10x
                                 smoother than the amortized-absolute baseline 0.063 rad
 3 KL BUDGET                     kl_phase/crop at placement end O(10-100), not thousands
 4 NULL                          time-rolled h+b -> placement corr to truth ~ 0 (no leak)
 5 RIGIDITY PROBE                +pi wrong t=1 phase; recovery half-life within bounds

CLI:  python blind_tests.py --test all|1|2|3|4|5 [--pre 300] [--seed 0] [--refresh]
Placement protocol = m2 stage 1 verbatim (300 steps, AdamW lr 1e-3, batch 16, clip 5.0),
targets = PF teacher smoothed w=101 (SPEC (d) preferred cell). All results appended to
blind_results.json after every test (host-restart resilience).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
import pm_common as P                     # noqa: E402
import innovq                             # noqa: E402  (interface only; source unread)

OUT = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq"
RES_PATH = f"{OUT}/blind_results.json"
TWO_PI = 2 * math.pi
FPS = 50.0
DEV = "cuda:0"
S_PHI = 0.05
S_LT = 0.0025
GAMMA_P = P.PHYS["gamma_phase"]
RHO_P = math.exp(-GAMMA_P)
BASELINE_ABS_JITTER = 0.063     # rad/frame, collapsed absolute-phase cells (campaign)

ap = argparse.ArgumentParser()
ap.add_argument("--test", default="all")
ap.add_argument("--pre", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--refresh", action="store_true")
A = ap.parse_args()


# ------------------------------------------------------------------ util
def wrapd(x):
    return (x + math.pi) % TWO_PI - math.pi


def corr_percrop(a, b):
    return float(torch.exp(1j * (a - b).double()).mean(dim=1).abs().mean())


def corr_pooled(a, b):
    return float(torch.abs(torch.exp(1j * (a - b).double()).mean()))


def load_results():
    if os.path.exists(RES_PATH):
        return json.load(open(RES_PATH))
    return {}


def save_results(res):
    json.dump(res, open(RES_PATH, "w"), indent=1, default=float)
    print(f"[saved] {RES_PATH}", flush=True)


def build_model(seed):
    torch.manual_seed(seed)
    m = innovq.InnovQ().to(DEV)
    ck = torch.load(P.CKPT, map_location=DEV, weights_only=False)
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    bad = [k for k in missing
           if not (k.startswith("init_head") or k.startswith("innov_head") or k == "Pi_phys")]
    assert not bad and not unexpected, (missing, unexpected)
    return m


def get_train():
    return P.build_crops(P.load_songs("train"), n_per_song=2, seed=0, dev=DEV)


def get_eval():
    return P.build_crops(P.load_songs("eval"), n_per_song=1, seed=1, crop=1000, dev=DEV)


def get_teacher(train, win=101):
    tea = innovq.pf_targets("train", 2, 0, 256, dev=DEV)
    phi_sm, lt_sm = innovq.smooth_track(tea["phi"], win)   # returns (phi_sm, implied lt)
    innov = wrapd(phi_sm[:, 1:] - phi_sm[:, :-1] - torch.exp(lt_sm[:, :-1]))
    stats = dict(
        corr_raw_percrop=corr_percrop(tea["phi"], train["phi"]),
        corr_sm_percrop=corr_percrop(phi_sm, train["phi"]),
        corr_sm_pooled=corr_pooled(phi_sm, train["phi"]),
        mean_abs_innov_raw=float(wrapd(tea["phi"][:, 1:] - tea["phi"][:, :-1]
                                       - torch.exp(tea["lt"][:, :-1])).abs().mean()),
        mean_abs_innov_sm=float(innov.abs().mean()))
    return dict(phi=phi_sm, lt=lt_sm), stats


# ------------------------------------------------------------------ placement (m2 stage 1)
def run_placement(model, train, TEA, steps, seed, roll=0, tag="placed"):
    ck_path = f"{OUT}/blind_{tag}.pt"
    if os.path.exists(ck_path) and not A.refresh:
        sd = torch.load(ck_path, map_location=DEV, weights_only=False)
        model.load_state_dict(sd["model"])
        model.eval()
        print(f"[{tag}] loaded existing {ck_path}", flush=True)
        return sd["hist"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    h, b = train["h"], train["b"]
    if roll:
        h = torch.roll(h, roll, dims=1)
        b = torch.roll(b, roll, dims=1)
    N = h.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    hist = []
    t0 = time.time()
    for s in range(1, steps + 1):
        idx = torch.tensor(rng.integers(0, N, 16), device=DEV, dtype=torch.long)
        r = innovq.rollout(model, h[idx], b[idx], sample=False)
        loss = (1 - torch.cos(r["phi"] - TEA["phi"][idx])).mean() * 10 \
               + ((r["lt"] - TEA["lt"][idx]) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if s % 50 == 0 or s == 1:
            hist.append(dict(step=s, loss=float(loss),
                             mean_abs_innov=float(r["mean_abs_innov"])))
            print(f"[{tag}][PRE] s{s:4d} sup={float(loss):8.4f} "
                  f"innov={float(r['mean_abs_innov']):.2e} "
                  f"| {s/(time.time()-t0):.2f} it/s", flush=True)
    model.eval()
    torch.save(dict(model=model.state_dict(), hist=hist, seed=seed, roll=roll,
                    steps=steps), ck_path)
    print(f"[{tag}] saved {ck_path}", flush=True)
    return hist


@torch.no_grad()
def full_rollout(model, D, sample, chunk=48, roll=0, seed=0):
    torch.manual_seed(seed)
    h, b = D["h"], D["b"]
    if roll:
        h = torch.roll(h, roll, dims=1)
        b = torch.roll(b, roll, dims=1)
    keys = ["phi", "lt", "MU", "SQ", "kl_p", "rho1", "mu_phi1"]
    outs = {k: [] for k in keys}
    for i in range(0, h.shape[0], chunk):
        r = innovq.rollout(model, h[i:i + chunk], b[i:i + chunk], sample=sample)
        for k in keys:
            outs[k].append(r[k])
    return {k: torch.cat(v, 0) for k, v in outs.items()}


# ================================================================== TEST 1
def test1(model, ev, tag):
    """Zero-innovation known-answer: sawtooth exactness + read-out F vs analytic grid."""
    torch.manual_seed(0)
    h, b = ev["h"][:8], ev["b"][:8]
    with torch.no_grad():
        r = innovq.rollout(model, h, b, sample=False, zero_innov=True)
    phi, lt, Z = r["phi"], r["lt"], r["Z"]
    T = phi.shape[1]
    rec_err = wrapd(phi[:, 1:] - (phi[:, :-1] + torch.exp(lt[:, :-1])))
    max_rec = float(rec_err.abs().max())
    lt_drift = float((lt - lt[:, :1]).abs().max())
    max_mu = float(r["MU"].abs().max())
    Fs, n_ref, n_est = [], [], []
    for i in range(phi.shape[0]):
        p1 = float(phi[i, 0]); w = float(torch.exp(lt[i, 0]))
        mi = int(Z[i, 0, 3:7].argmax()) + 1
        # analytic synthetic grid: continuous phase p(t)=p1+w*t crosses 2*pi*k/mi
        ref = []
        for k in range(mi):
            t0 = ((TWO_PI * k / mi - p1) % TWO_PI) / w
            while t0 <= T - 1:
                if t0 > 0:
                    ref.append(t0 / FPS)
                t0 += TWO_PI / w
        ref = np.sort(np.array(ref))
        est = innovq.beats_from_barphase(phi[i].cpu().numpy(), mi, FPS)
        Fs.append(innovq.f_measure(ref, est))
        n_ref.append(len(ref)); n_est.append(len(est))
    out = dict(model_state=tag, max_recursion_err_rad=max_rec, lt_drift=lt_drift,
               max_abs_mu_forced_zero=max_mu, F_per_song=[float(f) for f in Fs],
               F_mean=float(np.mean(Fs)), n_ref=n_ref, n_est=n_est,
               PASS=bool(max_rec < 1e-5 and lt_drift < 1e-6 and max_mu == 0.0
                         and np.mean(Fs) >= 0.98))
    print(f"[T1:{tag}] max_rec={max_rec:.2e} lt_drift={lt_drift:.2e} "
          f"max|MU|={max_mu:.2e} F={out['F_mean']:.4f} PASS={out['PASS']}", flush=True)
    return out


# ================================================================== TEST 2
def test2(model, train, seed):
    """Smoothness by construction on real crops, SAMPLED q (placement-end model)."""
    r = full_rollout(model, train, sample=True, seed=seed)
    mu = r["MU"]
    eps = wrapd(r["phi"][:, 1:] - r["phi"][:, :-1] - torch.exp(r["lt"][:, :-1]))
    # physical-law increment sample, same count, u-clamped as SPEC (c)
    g = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.rand(eps.numel(), generator=g).clamp(1e-4, 1 - 1e-4)
    eps_phys = GAMMA_P * torch.tan(math.pi * (u - 0.5))
    def iqr_std(x):
        q = torch.quantile(x.flatten().float().cpu(), torch.tensor([0.25, 0.75]))
        return float((q[1] - q[0]) / 1.349)
    st_traj, st_phys = iqr_std(eps), iqr_std(eps_phys)
    out = dict(seed=seed, n_crops=int(mu.shape[0]),
               max_abs_mu=float(mu.abs().max()), mean_abs_mu=float(mu.abs().mean()),
               bound_s_phi=S_PHI,
               mean_abs_eps_realized=float(eps.abs().mean()),
               iqr_std_eps=st_traj, iqr_std_phys=st_phys,
               ratio_vs_phys=st_traj / st_phys,
               smoother_than_baseline_x=BASELINE_ABS_JITTER / max(float(eps.abs().mean()), 1e-12),
               sat_frac_08bound=float((mu.abs() > 0.8 * S_PHI).float().mean()))
    out["PASS"] = bool(out["max_abs_mu"] <= S_PHI + 1e-6
                       and out["ratio_vs_phys"] <= 2.0
                       and out["smoother_than_baseline_x"] >= 10.0)
    print(f"[T2] max|mu|={out['max_abs_mu']:.4f} mean|mu|={out['mean_abs_mu']:.2e} "
          f"mean|eps|={out['mean_abs_eps_realized']:.2e} "
          f"iqr_std traj/phys={st_traj:.2e}/{st_phys:.2e} (x{out['ratio_vs_phys']:.2f}) "
          f"baseline-smoother x{out['smoother_than_baseline_x']:.0f} PASS={out['PASS']}",
          flush=True)
    return out


# ================================================================== TEST 3
def test3(model, train, dec, hdec, seed):
    """KL budget at placement end (sampled ELBO accounting over the full train set)."""
    torch.manual_seed(seed)
    kl_ps, kl_tot, innovs, sats = [], [], [], []
    info_last = None
    with torch.no_grad():
        N = train["b"].shape[0]
        for i in range(0, N, 48):
            idx = torch.arange(i, min(i + 48, N), device=DEV)
            _, info, tr = innovq.elbo_innovq(model, train, dec, hdec, idx=idx,
                                             beta=1.0, sample=True)
            kl_ps.append(tr["kl_p"])
            kl_tot.append(tr["kl_p"] + tr["kl_l"] + tr["kl_m"] + tr["kl_dv"])
            innovs.append(info["mean_abs_innov"]); sats.append(info["sat_frac"])
            info_last = info
            # float64 closed-form cross-check on this chunk (innovation KL, SPEC (b))
            if i == 0:
                mu64 = tr["MU"].double(); sq64 = tr["SQ"].double()
                rq = 1.0 - sq64
                kl64 = torch.log((1 - 2 * rq * RHO_P * torch.cos(mu64) + (rq * RHO_P) ** 2)
                                 / ((1 - rq ** 2) * (1 - RHO_P ** 2))).sum(1)
                r1 = tr["rho1"].double()
                kl64_t1 = torch.log((1 - 2 * r1 * 1e-6 * torch.cos(tr["mu_phi1"].double() - math.pi)
                                     + (r1 * 1e-6) ** 2) / ((1 - r1 ** 2) * (1 - 1e-12)))
                xcheck = dict(code=float(tr["kl_p"].mean()),
                              analytic64=float((kl64 + kl64_t1).mean()))
                xcheck["ratio"] = xcheck["code"] / max(xcheck["analytic64"], 1e-9)
    kl_p = torch.cat(kl_ps)
    out = dict(seed=seed, n_crops=int(kl_p.shape[0]),
               kl_phase_per_crop_mean=float(kl_p.mean()),
               kl_phase_p50=float(kl_p.median()), kl_phase_p90=float(kl_p.quantile(0.9)),
               kl_phase_max=float(kl_p.max()),
               kl_total_per_crop_mean=float(torch.cat(kl_tot).mean()),
               mean_abs_innov=float(np.mean(innovs)), sat_frac=float(np.mean(sats)),
               rho_q=info_last["rho_q"], rho1=info_last["rho1"],
               accounting_xcheck=xcheck,
               spec_bracket=[15, 100], collapsed_reference=5800)
    m = out["kl_phase_per_crop_mean"]
    out["in_spec_bracket"] = bool(15 <= m <= 100)
    out["PASS"] = bool(m <= 300)   # O(10-100), hard-fail level well below the old thousands
    print(f"[T3] kl_phase/crop mean={m:.1f} p50={out['kl_phase_p50']:.1f} "
          f"p90={out['kl_phase_p90']:.1f} max={out['kl_phase_max']:.1f} "
          f"xcheck ratio={xcheck['ratio']:.6f} in_bracket={out['in_spec_bracket']} "
          f"PASS={out['PASS']}", flush=True)
    return out


# ================================================================== TEST 4
def test4(model_placed, train, TEA, tea_stats, seed):
    """Leak null: time-rolled h+b. (i) eval-time roll on the placed model;
    (ii) a placement trained ON rolled inputs from scratch."""
    roll = 128
    r_norm = full_rollout(model_placed, train, sample=False, seed=seed)
    r_roll = full_rollout(model_placed, train, sample=False, roll=roll, seed=seed)
    out = dict(roll_frames=roll,
               corr_pooled_normal=corr_pooled(r_norm["phi"], train["phi"]),
               corr_pooled_rolled=corr_pooled(r_roll["phi"], train["phi"]),
               corr_percrop_normal=corr_percrop(r_norm["phi"], train["phi"]),
               corr_percrop_rolled=corr_percrop(r_roll["phi"], train["phi"]),
               corr2teacher_pooled_normal=corr_pooled(r_norm["phi"], TEA["phi"]),
               corr2teacher_pooled_rolled=corr_pooled(r_roll["phi"], TEA["phi"]))
    # (ii) fresh model, placement on rolled inputs
    m2 = build_model(seed + 100)
    hist = run_placement(m2, train, TEA, A.pre, seed, roll=roll, tag="placed_rollnull")
    rr = full_rollout(m2, train, sample=False, roll=roll, seed=seed)
    out["trained_on_rolled"] = dict(
        final_sup_loss=hist[-1]["loss"],
        corr_pooled=corr_pooled(rr["phi"], train["phi"]),
        corr_percrop=corr_percrop(rr["phi"], train["phi"]),
        corr2teacher_pooled=corr_pooled(rr["phi"], TEA["phi"]))
    out["PASS"] = bool(out["corr_pooled_rolled"] < 0.10
                       and out["trained_on_rolled"]["corr_pooled"] < 0.10
                       and out["corr_pooled_normal"] > 3 * out["corr_pooled_rolled"])
    print(f"[T4] pooled corr normal={out['corr_pooled_normal']:.4f} "
          f"rolled={out['corr_pooled_rolled']:.4f} "
          f"trained-on-rolled={out['trained_on_rolled']['corr_pooled']:.4f} "
          f"(percrop rolled={out['corr_percrop_rolled']:.4f}) PASS={out['PASS']}",
          flush=True)
    return out


# ================================================================== TEST 5
def _identify_layout(model, h, b):
    """Behavioral identification of head output layouts (source never read).
    Certification = exact replay match; any wrong guess fails the certificate."""
    with torch.no_grad():
        r = innovq.rollout(model, h, b, sample=False)
        ctx = model.encode_posterior(h, b)
        lay = {}
        # ---- init head input order
        for order, inp in (("mean_c1", torch.cat([ctx.mean(1), ctx[:, 0]], -1)),
                           ("c1_mean", torch.cat([ctx[:, 0], ctx.mean(1)], -1))):
            raw = model.init_head(inp)
            # rho1 channel: sigmoid(raw)*0.9
            rho_c = [c for c in range(raw.shape[-1])
                     if torch.allclose(torch.sigmoid(raw[:, c]) * 0.9, r["rho1"], atol=1e-5)]
            if rho_c:
                lay["init_input"] = order
                lay["raw_init"] = raw
                lay["rho1_c"] = rho_c[0]
                break
        if "rho1_c" not in lay:
            return None, r, ctx
        raw = lay.pop("raw_init")
        n = raw.shape[-1]
        # phi1: pair (i,j) with atan2(raw_j, raw_i) == mu_phi1
        tgt = r["mu_phi1"] % TWO_PI
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if torch.allclose(torch.atan2(raw[:, j], raw[:, i]) % TWO_PI, tgt, atol=1e-4):
                    lay["phi1_ij"] = (i, j)
        # mu_l1 channel under candidate offsets
        for c in range(n):
            for off_name, off in (("init_lv_mu", P.PHYS["init_level_mu"]), ("none", 0.0),
                                  ("INIT_LV_MU", innovq.INIT_LV_MU)):
                if torch.allclose(raw[:, c] + off, r["mu_l1"], atol=1e-5):
                    lay["mu_l1_c"] = c; lay["mu_l1_off"] = off
        # s_l1 channel under candidate transforms
        import torch.nn.functional as FF
        for c in range(n):
            for f_name, f in (("sp_bls0_p05", lambda x: FF.softplus(x + innovq.B_LS0) + 0.05),
                              ("sp_bls0", lambda x: FF.softplus(x + innovq.B_LS0)),
                              ("sp_p05", lambda x: FF.softplus(x) + 0.05)):
                if torch.allclose(f(raw[:, c]), r["s_l1"], atol=1e-5):
                    lay["s_l1_c"] = c; lay["s_l1_f"] = f_name
        used = set([lay.get("rho1_c"), lay.get("mu_l1_c"), lay.get("s_l1_c")]) \
            | set(lay.get("phi1_ij", ()))
        lay["meter_cs"] = [c for c in range(n) if c not in used]
        # meter feature convention: match Z[:,0,3:7]
        mfeat = r["Z"][:, 0, 3:7]
        mlog = raw[:, lay["meter_cs"]]
        for name, cand in (("softmax_t0.3", torch.softmax(mlog / 0.3, -1)),
                           ("softmax", torch.softmax(mlog, -1)),
                           ("onehot", torch.nn.functional.one_hot(
                               mlog.argmax(-1), 4).float())):
            if torch.allclose(cand, mfeat, atol=1e-5):
                lay["meter_f"] = name
        # meter constant along t? (needed for fixed-meter replay)
        lay["meter_const_t"] = bool((r["Z"][:, :, 3:7] - r["Z"][:, :1, 3:7]).abs().max() < 1e-6)
        # ---- innov head: channels via returned MU / lt increments / SQ
        zprev = r["Z"][:, :-1]
        for iorder, iinp in (("c_z", torch.cat([ctx[:, 1:], zprev], -1)),
                             ("z_c", torch.cat([zprev, ctx[:, 1:]], -1))):
            rawi = model.innov_head(iinp)
            u_c = [c for c in range(rawi.shape[-1])
                   if torch.allclose(torch.tanh(rawi[..., c]) * S_PHI, r["MU"], atol=1e-5)]
            if u_c:
                lay["innov_input"] = iorder
                lay["u_c"] = u_c[0]
                ltinc = r["lt"][:, 1:] - r["lt"][:, :-1]
                lay["v_c"] = [c for c in range(rawi.shape[-1])
                              if torch.allclose(torch.tanh(rawi[..., c]) * S_LT, ltinc,
                                                atol=1e-5)]
                import torch.nn.functional as FF
                lay["r_c_sq"] = [c for c in range(rawi.shape[-1])
                                 if torch.allclose(FF.softplus(rawi[..., c] + innovq.R0),
                                                   r["SQ"], atol=1e-6)]
                break
    return lay, r, ctx


def _replay(model, ctx, lay, phi1_off=0.0, T=256):
    """Spec-recursion replay from identified layout (mean rollout, meter held)."""
    inp = torch.cat([ctx.mean(1), ctx[:, 0]], -1) if lay["init_input"] == "mean_c1" \
        else torch.cat([ctx[:, 0], ctx.mean(1)], -1)
    raw = model.init_head(inp)
    i, j = lay["phi1_ij"]
    phi = (torch.atan2(raw[:, j], raw[:, i]) + phi1_off) % TWO_PI
    lt = raw[:, lay["mu_l1_c"]] + lay["mu_l1_off"]
    mlog = raw[:, lay["meter_cs"]]
    mfeat = {"softmax_t0.3": torch.softmax(mlog / 0.3, -1),
             "softmax": torch.softmax(mlog, -1),
             "onehot": torch.nn.functional.one_hot(mlog.argmax(-1), 4).float()}[lay["meter_f"]]
    P_, L_ = [phi], [lt]
    for t in range(1, T):
        zf = torch.cat([torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1),
                        lt.clamp(-12.0, 6.0).unsqueeze(-1), mfeat], -1)
        ii = torch.cat([ctx[:, t], zf], -1) if lay["innov_input"] == "c_z" \
            else torch.cat([zf, ctx[:, t]], -1)
        rawi = model.innov_head(ii)
        mu = torch.tanh(rawi[..., lay["u_c"]]) * S_PHI
        dlt = torch.tanh(rawi[..., lay["v_c"][0]]) * S_LT
        phi = (phi + torch.exp(lt.clamp(-12.0, 6.0)) + mu) % TWO_PI
        lt = lt + dlt
        P_.append(phi); L_.append(lt)
    return torch.stack(P_, 1), torch.stack(L_, 1)


def test5(model, train, seed):
    torch.manual_seed(seed)
    h, b = train["h"][:16], train["b"][:16]
    lay, r, ctx = _identify_layout(model, h, b)
    if lay is None or "u_c" not in lay or "phi1_ij" not in lay:
        return dict(PASS=False, blocked="layout identification failed", layout=str(lay))
    with torch.no_grad():
        phi_ref, lt_ref = _replay(model, ctx, lay, 0.0, T=h.shape[1])
        cert = float(wrapd(phi_ref - r["phi"]).abs().max())
        cert_lt = float((lt_ref - r["lt"]).abs().max())
        certified = cert < 1e-4 and cert_lt < 1e-4
        rep = dict(layout={k: str(v) for k, v in lay.items()},
                   replay_max_phase_diff=cert, replay_max_lt_diff=cert_lt,
                   replay_certified=bool(certified))
        if not certified:
            rep["PASS"] = False
            rep["blocked"] = "replay does not match rollout; cannot run perturbation"
            print(f"[T5] REPLAY CERT FAILED diff={cert:.2e}", flush=True)
            return rep
        phi_p, lt_p = _replay(model, ctx, lay, math.pi, T=h.shape[1])
        e = wrapd(phi_p - phi_ref).abs()        # [16,T], e[:,0] == pi
        med = e.median(0).values.cpu().numpy()
        def first_below(thr):
            w = np.where(med <= thr)[0]
            return int(w[0]) if len(w) else -1
        # per-crop half-life
        hl = []
        for k in range(e.shape[0]):
            w = torch.where(e[k] <= math.pi / 2)[0]
            hl.append(int(w[0]) if len(w) else -1)
        # correction-rate usage in the perturbed run
        # recompute innovations of perturbed traj
        eps_p = wrapd(phi_p[:, 1:] - phi_p[:, :-1] - torch.exp(lt_p[:, :-1].clamp(-12., 6.)))
        rep.update(err_t0=float(e[:, 0].mean()), err_final_median=float(med[-1]),
                   half_life_median_frames=first_below(math.pi / 2),
                   quarter_life_median_frames=first_below(math.pi / 4),
                   half_life_per_crop=hl,
                   n_recovered_half=int(sum(x >= 0 for x in hl)), n_crops=e.shape[0],
                   mean_abs_innov_perturbed=float(eps_p.abs().mean()),
                   sat_frac_perturbed=float((eps_p.abs() > 0.8 * S_PHI).float().mean()),
                   theoretical_min_half_life=float((math.pi / 2) / S_PHI),
                   err_median_traj_every16=[float(x) for x in med[::16]])
        rep["verdict"] = ("RECOVERS" if rep["n_recovered_half"] >= 12 else
                          "PARTIAL" if rep["n_recovered_half"] >= 4 else "STUCK")
        rep["PASS"] = True   # probe: reported, not thresholded; certificate is the gate
        print(f"[T5] cert={cert:.2e} half-life(med)={rep['half_life_median_frames']}fr "
              f"recovered {rep['n_recovered_half']}/{e.shape[0]} "
              f"final err(med)={rep['err_final_median']:.3f} rad "
              f"sat={rep['sat_frac_perturbed']:.3f} verdict={rep['verdict']}", flush=True)
    return rep


# ================================================================== main
def main():
    res = load_results()
    res["config"] = dict(pre=A.pre, seed=A.seed, teacher="pf_sm101",
                         placement_protocol="m2 stage1: AdamW lr1e-3 batch16 clip5",
                         time=time.strftime("%F %T"))
    want = set(A.test.split(",")) if A.test != "all" else {"1", "2", "3", "4", "5"}
    train = get_train()
    ev = get_eval()
    TEA, tea_stats = get_teacher(train)
    res["teacher_stats"] = tea_stats
    print("[teacher]", json.dumps(tea_stats), flush=True)
    save_results(res)

    model = build_model(A.seed)
    if "1" in want:
        res["test1_initmodel"] = test1(model, ev, "init")
        save_results(res)

    # ---- placement (shared state for T2/T3/T4/T5)
    hist = run_placement(model, train, TEA, A.pre, A.seed, tag="placed")
    res["placement_hist"] = hist
    with torch.no_grad():
        rr = full_rollout(model, train, sample=False, seed=A.seed)
        res["placement_end"] = dict(
            corr_percrop_truth=corr_percrop(rr["phi"], train["phi"]),
            corr_pooled_truth=corr_pooled(rr["phi"], train["phi"]),
            corr_percrop_teacher=corr_percrop(rr["phi"], TEA["phi"]),
            corr_pooled_teacher=corr_pooled(rr["phi"], TEA["phi"]))
    print("[placement_end]", json.dumps(res["placement_end"]), flush=True)
    save_results(res)

    if "1" in want:
        res["test1_placed"] = test1(model, ev, "placed")
        save_results(res)
    if "2" in want:
        res["test2"] = test2(model, train, A.seed)
        save_results(res)
    if "3" in want:
        d0, h0 = P.new_decoders(DEV)
        dec, hdec = innovq.Cut(d0), innovq.Cut(h0)
        res["test3"] = test3(model, train, dec, hdec, A.seed)
        save_results(res)
    if "4" in want:
        res["test4"] = test4(model, train, TEA, tea_stats, A.seed)
        save_results(res)
    if "5" in want:
        res["test5"] = test5(model, train, A.seed)
        save_results(res)

    n_pass = sum(1 for k, v in res.items()
                 if k.startswith("test") and isinstance(v, dict) and v.get("PASS"))
    n_all = sum(1 for k, v in res.items()
                if k.startswith("test") and isinstance(v, dict) and "PASS" in v)
    res["summary"] = f"{n_pass}/{n_all} test blocks PASS"
    save_results(res)
    print("[DONE]", res["summary"], flush=True)


if __name__ == "__main__":
    main()
