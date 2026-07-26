"""
Reality check: TEMPO_RW assumption (ELBO_for_DBN.md §3, §5.3).

Paper's generative model for tempo:
    log phi_dot_t ~ N( log phi_dot_{t-1}, sigma^2 )
i.e. a Gaussian random walk in LOG-tempo. Since phi_dot (angular advance / beat
rate) is proportional to 1/IBI, log-tempo increments equal (up to sign) the
per-beat log-IBI increments:
    Delta_t = log(IBI_t) - log(IBI_{t-1})

The paper asserts these increments are GAUSSIAN (log-Normal RW). We test that on
real beat annotations.

Tests:
  1. Excess kurtosis of Delta (Gaussian => 0). Heavy tails => misspecified.
  2. MLE fit of Gaussian vs Laplace to pooled+per-dataset Delta; compare by
     total log-likelihood (LLR) and AIC. Laplace = madmom's exp(-lambda|.|) kernel.
  3. Anderson-Darling normality test.
  4. Lag-1 autocorrelation of Delta: is the walk truly Markov-1 (white increments)
     or is there structure (e.g. mean-reversion => negative autocorr) the RW ignores?

Beat rate vs IBI: phi_dot proportional to 1/IBI, so log phi_dot = const - log IBI.
A Gaussian RW on log phi_dot <=> Gaussian RW on log IBI (sign flip only; symmetric
tests are invariant). We work in log-IBI.
"""
import glob
import os
import numpy as np
from scipy import stats

ANN = "/home/sogang/jaehoon/VBPM/dataset_store/beat_this_annotations/{d}/annotations/beats/*.beats"
DATASETS = ["ballroom", "beatles", "asap", "hjdb", "gtzan", "hainsworth", "rwc", "smc"]

# Physiological / annotation sanity bounds on IBI (sec). 0.15s = 400 BPM,
# 2.0s = 30 BPM. Increments outside these arise from gaps / missing beats, not
# tempo dynamics; keep them out so we test the RW, not annotation holes.
IBI_MIN, IBI_MAX = 0.15, 2.0


def load_deltas(path):
    """Return per-song log-IBI increments Delta for one .beats file."""
    arr = np.loadtxt(path, ndmin=2)
    t = arr[:, 0]
    t = np.sort(t)
    if t.size < 3:
        return np.empty(0)
    ibi = np.diff(t)
    ok = (ibi > IBI_MIN) & (ibi < IBI_MAX)
    logibi = np.log(ibi)
    d = np.diff(logibi)
    # only keep increments where BOTH adjacent IBIs are valid (contiguous)
    valid = ok[:-1] & ok[1:]
    return d[valid]


def gauss_ll(x, mu, sigma):
    return np.sum(stats.norm.logpdf(x, mu, sigma))


def laplace_ll(x, loc, b):
    return np.sum(stats.laplace.logpdf(x, loc, b))


def analyze(deltas, name):
    n = deltas.size
    if n < 30:
        return None
    # Gaussian MLE
    mu = deltas.mean()
    sigma = deltas.std(ddof=0)
    # Laplace MLE: loc = median, b = mean|x - median|
    loc = np.median(deltas)
    b = np.mean(np.abs(deltas - loc))
    if sigma <= 0 or b <= 0:
        return None
    llg = gauss_ll(deltas, mu, sigma)
    lll = laplace_ll(deltas, loc, b)
    exkurt = stats.kurtosis(deltas, fisher=True, bias=False)
    # Anderson-Darling normality (standardized)
    ad = stats.anderson((deltas - mu) / sigma, dist="norm")
    ad_stat = ad.statistic
    # lag-1 autocorrelation of increments
    dm = deltas - mu
    if dm.size > 2 and np.sum(dm[:-1] ** 2) > 0:
        r1 = np.sum(dm[1:] * dm[:-1]) / np.sum(dm ** 2)
    else:
        r1 = np.nan
    # AIC: both 2 params
    aic_g = 2 * 2 - 2 * llg
    aic_l = 2 * 2 - 2 * lll
    return dict(
        name=name, n=n, mu=mu, sigma=sigma,
        exkurt=exkurt, ad=ad_stat,
        llg=llg, lll=lll, llr=lll - llg,
        aic_g=aic_g, aic_l=aic_l, daic=aic_g - aic_l,
        r1=r1, laplace_wins=(lll > llg),
    )


def main():
    per_ds = {}
    pooled = []
    for d in DATASETS:
        files = sorted(glob.glob(ANN.format(d=d)))
        ds_deltas = []
        for f in files:
            dd = load_deltas(f)
            if dd.size:
                ds_deltas.append(dd)
        if ds_deltas:
            cat = np.concatenate(ds_deltas)
            per_ds[d] = cat
            pooled.append(cat)
    pooled = np.concatenate(pooled)

    print("=" * 100)
    print("TEMPO_RW reality check: are per-beat log-IBI increments Gaussian (paper) or heavy-tailed?")
    print(f"IBI kept in [{IBI_MIN}, {IBI_MAX}] s; increments require both adjacent IBIs valid+contiguous")
    print("=" * 100)
    hdr = f"{'dataset':<12}{'N':>8}{'exKurt':>10}{'AD':>9}{'LLR(L-G)':>12}{'dAIC(G-L)':>12}{'ac1':>8}  winner"
    print(hdr)
    print("-" * len(hdr))

    rows = {}
    for d in DATASETS:
        if d not in per_ds:
            continue
        r = analyze(per_ds[d], d)
        rows[d] = r
        if r is None:
            print(f"{d:<12}  (insufficient)")
            continue
        win = "Laplace" if r["laplace_wins"] else "Gaussian"
        print(f"{d:<12}{r['n']:>8}{r['exkurt']:>10.2f}{r['ad']:>9.1f}"
              f"{r['llr']:>12.1f}{r['daic']:>12.1f}{r['r1']:>8.3f}  {win}")

    rp = analyze(pooled, "POOLED")
    rows["POOLED"] = rp
    print("-" * len(hdr))
    win = "Laplace" if rp["laplace_wins"] else "Gaussian"
    print(f"{'POOLED':<12}{rp['n']:>8}{rp['exkurt']:>10.2f}{rp['ad']:>9.1f}"
          f"{rp['llr']:>12.1f}{rp['daic']:>12.1f}{rp['r1']:>8.3f}  {win}")

    # AD critical value for normality at 1% is ~1.09 (n large). Report.
    print()
    print("Anderson-Darling normal critical values (from scipy, pooled):")
    ad = stats.anderson((pooled - pooled.mean()) / pooled.std(), dist="norm")
    for sl, cv in zip(ad.significance_level, ad.critical_values):
        print(f"   sig {sl:>4}%  crit {cv:.3f}")
    print(f"   -> pooled AD statistic = {ad.statistic:.1f} (reject Gaussian if > crit)")

    # Tail mass comparison: fraction of |Delta| beyond k*sigma vs Gaussian expectation
    print()
    s = pooled.std()
    for k in (3, 4, 5):
        emp = np.mean(np.abs(pooled - pooled.mean()) > k * s)
        gaus = 2 * stats.norm.sf(k)
        ratio = emp / gaus if gaus > 0 else np.inf
        print(f"   P(|Delta| > {k} sigma): empirical {emp:.2e}  Gaussian {gaus:.2e}  ratio {ratio:6.1f}x")

    # How much heavier tail does Laplace need at same variance? report per-obs LL gain
    print()
    print(f"Pooled per-observation LL gain (Laplace - Gaussian): {rp['llr']/rp['n']:.4f} nats/increment")
    print(f"Pooled Gaussian LL {rp['llg']:.1f}  Laplace LL {rp['lll']:.1f}  N={rp['n']}")

    # summary line
    n_lwin = sum(1 for d in DATASETS if d in rows and rows[d] and rows[d]["laplace_wins"])
    n_tot = sum(1 for d in DATASETS if d in rows and rows[d])
    print()
    print(f"Laplace beats Gaussian in {n_lwin}/{n_tot} datasets by log-likelihood.")
    kurts = [rows[d]["exkurt"] for d in DATASETS if d in rows and rows[d]]
    print(f"Excess kurtosis range across datasets: {min(kurts):.2f} .. {max(kurts):.2f} (Gaussian=0)")
    ac1s = [rows[d]["r1"] for d in DATASETS if d in rows and rows[d]]
    print(f"Lag-1 autocorrelation of increments range: {min(ac1s):.3f} .. {max(ac1s):.3f} (RW/Markov-1 => ~0)")


if __name__ == "__main__":
    main()
