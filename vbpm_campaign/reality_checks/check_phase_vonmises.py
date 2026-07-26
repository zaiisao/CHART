"""
Reality check: PHASE_VONMISES assumption (ELBO_for_DBN.md §3, §5.2).

Paper's generative model for phase:
    phi_t in [0, 2pi) is BAR phase ("0 = start of bar, ->2pi = end of bar").
    phi_t ~ vM( phi_{t-1} + phi_dot_{t-1}, kappa^p_phi )   -- a von Mises transition
    whose MEAN is a uniform (sawtooth) bar-pointer advance at rate phi_dot (tempo),
    and beats are emitted (decoder) at concentrated sub-bar phase. Under a bar-pointer
    model with meter M, beats sit at fixed sub-bar positions k*2pi/M, k=0..M-1.

Two coupled claims we test on REAL annotations (datasets WITH col1 = beat-in-bar):

  (A) SAWTOOTH / uniform advance. Reconstruct phi(t) by CONSTANT-RATE (time-linear)
      interpolation between consecutive downbeats (col1==1): phi(t) = 2pi * (t - t_db0)
      / (t_db1 - t_db0). This is exactly the paper's "mean advances uniformly across the
      bar" assumption. If tempo is constant across the bar (sawtooth), inter-beat
      intervals within a bar are equal.
        -> Test: within-bar IBI coefficient of variation (CV). CV=0 => perfect sawtooth.

  (B) CONCENTRATED BEAT PHASE. Under uniform advance, beat k should land at the exact
      grid phase g_k = 2pi*k/M. The residual
          r = wrap( phi_actual(beat) - g_k )  in (-pi, pi]
      is the phase noise that the prior's von Mises kappa must absorb. Pool residuals
      over all non-downbeat beats; empirical resultant length R -> kappa_hat (Fisher
      approx); von Mises log-likelihood vs uniform (LLR). High kappa / small circular
      std = beats sit tightly on the grid = assumption HOLDS. The downbeat itself is
      excluded (r==0 by construction).

We also report:
  - kappa_hat per beat-in-bar POSITION (systematic swing pushes some positions off-grid).
  - circular std of residual expressed as a FRACTION OF ONE BEAT (2pi/M), the intuitive
    "how far off the metric grid, in beats" number.
  - von-Mises adequacy: is the residual actually von Mises, or heavier-tailed (expressive
    timing outliers)? We compare per-obs LL of a fitted von Mises vs a wrapped Cauchy
    (heavy-tailed circular alternative) at matched resultant length.
"""
import glob
import numpy as np
from scipy import special, stats

ANN = "/home/sogang/jaehoon/VBPM/dataset_store/beat_this_annotations/{d}/annotations/beats/*.beats"
# datasets WITH col1 (beat-in-bar); smc has none -> excluded from phase test.
DATASETS = ["ballroom", "beatles", "asap", "hjdb", "gtzan", "hainsworth", "rwc"]

# sane bar durations (s): drop degenerate/gap bars. 0.3s..8s covers 30..400 BPM x M.
BAR_MIN, BAR_MAX = 0.3, 8.0
M_MIN, M_MAX = 2, 12  # beats per bar we accept


def wrap_pi(x):
    """wrap angle to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def kappa_from_R(R):
    """Fisher (1993) approximation to the vM MLE kappa from resultant length R."""
    if R < 1e-9:
        return 0.0
    if R < 0.53:
        return 2 * R + R ** 3 + 5 * R ** 5 / 6
    elif R < 0.85:
        return -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        return 1.0 / (R ** 3 - 4 * R ** 2 + 3 * R)


def load_song_residuals(path):
    """Return (residuals, positions, M_list, ibi_cv_list) for one .beats file.

    residuals: wrapped phase residual of each non-downbeat beat vs its metric grid.
    positions: the beat-in-bar index k (0-based) of each residual.
    M_list:    beats-per-bar of the bar each residual came from.
    ibi_cv_list: per-bar within-bar IBI coefficient of variation.
    """
    arr = np.loadtxt(path, ndmin=2)
    if arr.shape[1] < 2 or arr.shape[0] < 4:
        return None
    t = arr[:, 0].astype(float)
    col1 = arr[:, 1]
    order = np.argsort(t)
    t = t[order]
    col1 = col1[order]
    # downbeat indices = where col1 == 1
    db_idx = np.where(np.round(col1) == 1)[0]
    if db_idx.size < 2:
        return None
    res, pos, Ms, cvs = [], [], [], []
    for a, b in zip(db_idx[:-1], db_idx[1:]):
        # bar spans beats a..b (b exclusive is next downbeat). M beats in bar = b-a.
        M = b - a
        if M < M_MIN or M > M_MAX:
            continue
        t0, t1 = t[a], t[b]
        dur = t1 - t0
        if dur < BAR_MIN or dur > BAR_MAX:
            continue
        # within-bar IBI CV (sawtooth test)
        bar_t = t[a:b + 1]  # M+1 times (incl both downbeats)
        ibi = np.diff(bar_t)
        if np.all(ibi > 0):
            cvs.append(ibi.std(ddof=0) / ibi.mean())
        # phase residual for each beat k=1..M-1 (k=0 downbeat excluded, r==0)
        for k in range(1, M):
            phi_actual = 2 * np.pi * (t[a + k] - t0) / dur
            g_k = 2 * np.pi * k / M
            res.append(wrap_pi(phi_actual - g_k))
            pos.append(k)
            Ms.append(M)
    if not res:
        return None
    return (np.array(res), np.array(pos), np.array(Ms), np.array(cvs))


def vm_llr(res):
    """von Mises vs uniform log-likelihood ratio (total and per-obs) at MLE."""
    n = res.size
    C = np.cos(res).sum()
    S = np.sin(res).sum()
    R = np.hypot(C, S) / n
    mu = np.arctan2(S, C)
    kappa = kappa_from_R(R)
    # LL_vM = kappa*sum cos(res-mu) - n*log(2pi I0(kappa))
    #       = kappa*n*R - n*log(2pi) - n*log I0(kappa)
    # LL_unif = -n*log(2pi)
    # LLR = kappa*n*R - n*log I0(kappa)
    logI0 = np.log(special.i0e(kappa)) + kappa  # stable log I0
    llr = kappa * n * R - n * logI0
    circ_std = np.sqrt(-2 * np.log(R)) if R > 0 else np.inf  # radians
    return dict(n=n, R=R, mu=mu, kappa=kappa, llr=llr, llr_per=llr / n,
                circ_std=circ_std)


def wrapped_cauchy_ll(res, rho):
    """per-obs mean log-lik of wrapped Cauchy at concentration rho, centered at MLE mean."""
    C = np.cos(res).sum(); S = np.sin(res).sum()
    mu = np.arctan2(S, C)
    # f(theta) = (1/2pi) (1-rho^2)/(1+rho^2-2rho cos(theta-mu))
    d = res - mu
    ll = np.log((1 - rho ** 2) / (2 * np.pi) / (1 + rho ** 2 - 2 * rho * np.cos(d)))
    return ll.mean()


def main():
    per_ds_res, per_ds_pos, per_ds_M, per_ds_cv = {}, {}, {}, {}
    for d in DATASETS:
        R, P, M, CV = [], [], [], []
        for f in sorted(glob.glob(ANN.format(d=d))):
            out = load_song_residuals(f)
            if out is None:
                continue
            r, p, m, cv = out
            R.append(r); P.append(p); M.append(m); CV.append(cv)
        if R:
            per_ds_res[d] = np.concatenate(R)
            per_ds_pos[d] = np.concatenate(P)
            per_ds_M[d] = np.concatenate(M)
            per_ds_cv[d] = np.concatenate(CV) if CV else np.array([])

    print("=" * 108)
    print("PHASE_VONMISES reality check: do beats sit at concentrated sub-bar phase under uniform (sawtooth) advance?")
    print("phi reconstructed by TIME-LINEAR interpolation between downbeats (col1); residual = beat_phase - metric grid k*2pi/M")
    print("=" * 108)
    hdr = (f"{'dataset':<12}{'Nbeats':>9}{'R':>8}{'kappa':>9}{'circSD_rad':>12}"
           f"{'SD_/beat':>10}{'LLR/obs':>10}{'IBI_CV%':>9}{'mode_M':>8}")
    print(hdr)
    print("-" * len(hdr))

    pooled = []
    rows = {}
    for d in DATASETS:
        if d not in per_ds_res:
            continue
        res = per_ds_res[d]
        pooled.append(res)
        st = vm_llr(res)
        M = per_ds_M[d]
        mode_M = int(stats.mode(M, keepdims=False).mode)
        # circular std as fraction of one beat spacing (2pi/mode_M)
        sd_beat = st["circ_std"] / (2 * np.pi / mode_M)
        cv = per_ds_cv[d]
        cvm = np.median(cv) * 100 if cv.size else np.nan
        rows[d] = st
        print(f"{d:<12}{st['n']:>9}{st['R']:>8.3f}{st['kappa']:>9.2f}"
              f"{st['circ_std']:>12.3f}{sd_beat:>10.3f}{st['llr_per']:>10.3f}"
              f"{cvm:>9.2f}{mode_M:>8}")

    pooled = np.concatenate(pooled)
    stp = vm_llr(pooled)
    Mall = np.concatenate([per_ds_M[d] for d in per_ds_res])
    mode_M = int(stats.mode(Mall, keepdims=False).mode)
    sd_beat = stp["circ_std"] / (2 * np.pi / mode_M)
    cvall = np.concatenate([per_ds_cv[d] for d in per_ds_res if per_ds_cv[d].size])
    print("-" * len(hdr))
    print(f"{'POOLED':<12}{stp['n']:>9}{stp['R']:>8.3f}{stp['kappa']:>9.2f}"
          f"{stp['circ_std']:>12.3f}{sd_beat:>10.3f}{stp['llr_per']:>10.3f}"
          f"{np.median(cvall)*100:>9.2f}{mode_M:>8}")

    # ---- per beat-in-bar position (systematic swing) on pooled 4/4 bars ----
    print()
    print("Per beat-in-bar position, pooled over 4/4 bars (M=4): mean residual (deg) = systematic swing; kappa = spread")
    print(f"{'k (beat)':<10}{'grid_deg':>10}{'N':>9}{'meanRes_deg':>13}{'kappa':>9}{'circSD_deg':>12}")
    for d in per_ds_res:
        pass
    resM4 = np.concatenate([per_ds_res[d][per_ds_M[d] == 4] for d in per_ds_res])
    posM4 = np.concatenate([per_ds_pos[d][per_ds_M[d] == 4] for d in per_ds_res])
    for k in range(1, 4):
        rk = resM4[posM4 == k]
        if rk.size < 10:
            continue
        C = np.cos(rk).sum(); S = np.sin(rk).sum(); Rk = np.hypot(C, S) / rk.size
        mu = np.degrees(np.arctan2(S, C)); kap = kappa_from_R(Rk)
        csd = np.degrees(np.sqrt(-2 * np.log(Rk))) if Rk > 0 else np.inf
        print(f"{k:<10}{90*k:>10}{rk.size:>9}{mu:>13.2f}{kap:>9.2f}{csd:>12.2f}")

    # ---- von Mises adequacy WITHIN each homogeneous corpus ----
    print()
    print("von Mises vs wrapped Cauchy WITHIN each dataset (separates mixture heavy-tails from genuine ones):")
    print(f"{'dataset':<12}{'llVM/obs':>10}{'llWC/obs':>10}{'WC-VM':>9}   heavier-tailed-than-vM?")
    for d in per_ds_res:
        res = per_ds_res[d]
        C = np.cos(res).sum(); S = np.sin(res).sum(); Rd = np.hypot(C, S) / res.size
        kap = kappa_from_R(Rd)
        llvm = kap * Rd - (np.log(special.i0e(kap)) + kap) - np.log(2 * np.pi)
        llwc = wrapped_cauchy_ll(res, Rd)
        print(f"{d:<12}{llvm:>10.3f}{llwc:>10.3f}{llwc-llvm:>+9.3f}   {'YES' if llwc>llvm else 'no'}")

    # ---- von Mises adequacy vs heavy-tailed wrapped Cauchy (pooled) ----
    print()
    R = stp["R"]
    kap = stp["kappa"]
    ll_vm = (kap * R - (np.log(special.i0e(kap)) + kap) - np.log(2 * np.pi))
    # wrapped Cauchy with SAME resultant length rho=R (its mean resultant length = rho)
    ll_wc = wrapped_cauchy_ll(pooled, R)
    print("von Mises adequacy (pooled residuals), both matched to empirical resultant length R:")
    print(f"   per-obs mean logLik  von Mises = {ll_vm:.4f}   wrapped Cauchy = {ll_wc:.4f}"
          f"   (Cauchy - vM = {ll_wc-ll_vm:+.4f} nats)")
    print("   wrapped Cauchy > von Mises  =>  residual heavier-tailed than von Mises (expressive-timing outliers)")

    # tail mass of residual vs von Mises expectation
    print()
    print("Residual tail mass vs a von Mises with the SAME R (how often beats stray far off grid):")
    # simulate vM tail probabilities
    from numpy.random import default_rng
    rng = default_rng(0)
    sim = stats.vonmises.rvs(kap, size=500000, random_state=rng)
    for thr_deg in (30, 45, 60):
        thr = np.radians(thr_deg)
        emp = np.mean(np.abs(pooled - stp["mu"]) > thr) if False else np.mean(np.abs(wrap_pi(pooled - stp["mu"])) > thr)
        vm = np.mean(np.abs(wrap_pi(sim)) > thr)
        ratio = emp / vm if vm > 0 else np.inf
        print(f"   P(|residual| > {thr_deg:>2} deg): empirical {emp:.4f}   vonMises(kappa={kap:.1f}) {vm:.4f}   ratio {ratio:5.2f}x")

    # ---- meter distribution (context) ----
    print()
    print("Beats-per-bar (M) distribution per dataset (share of bars):")
    for d in per_ds_res:
        M = per_ds_M[d]
        # M here is per-residual; approximate bar share via unique
        vals, cnts = np.unique(M, return_counts=True)
        share = {int(v): round(c / (v - 1) / (np.sum(cnts / (vals - 1))), 3) for v, c in zip(vals, cnts)}
        top = sorted(share.items(), key=lambda kv: -kv[1])[:4]
        print(f"   {d:<12} " + "  ".join(f"M={v}:{s}" for v, s in top))

    # summary
    print()
    kaps = [rows[d]["kappa"] for d in rows]
    sds = [rows[d]["circ_std"] for d in rows]
    print(f"kappa_hat across datasets: {min(kaps):.2f} .. {max(kaps):.2f}  (pooled {stp['kappa']:.2f})")
    print(f"circular SD across datasets (rad): {min(sds):.3f} .. {max(sds):.3f}  (pooled {stp['circ_std']:.3f} rad "
          f"= {np.degrees(stp['circ_std']):.1f} deg = {stp['circ_std']/(2*np.pi/mode_M):.3f} of a beat)")
    print(f"Every-dataset LLR/obs vs uniform > 0: {all(rows[d]['llr_per']>0 for d in rows)} "
          f"(beats ARE concentrated, not uniform on the circle)")


if __name__ == "__main__":
    main()
