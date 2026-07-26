"""
alt_tempo_prior.py  --  IN-MODEL replacements for the VBPM tempo transition prior.

Faithful default (docs/ELBO_for_DBN.md 5.3):
    log phidot_t ~ N( log phidot_{t-1}, sigma_p^2 )   (Gaussian on log-tempo increments)
    KL = Gaussian KL in log-space (closed form); reparam log_tempo = mu + sigma*eps.

Real log-IBI increments are heavy-tailed (kurtosis ~13); a Gaussian increment law is
mis-specified in the tails. Since phidot ~ 1/IBI, the log-tempo increment equals MINUS the
log-IBI increment; every symmetric candidate below is invariant to that sign, so we fit the
real per-song log-IBI increments delta = log IBI_{i+1} - log IBI_i directly.

Candidates (all reparameterizable, all keep a tempo TRANSITION law on log-tempo):
    Gaussian      (baseline / current)               2 params  closed-form KL
    Laplace                                           2 params  closed-form KL
    Student-t (learnable dof)                         3 params  MC KL (reparam rsample)
    NIG (Normal-Inverse-Gaussian)                     4 params  MC KL
    GaussMix2 (2-component Gaussian mixture)          5 params  MC KL
    ARCH-Gauss  sig_t^2 = w + a*delta_{t-1}^2         3 params  closed-form Gaussian KL / step

Ranking: held-out log-likelihood (per obs) + AIC + BIC.
"""
import os, glob, numpy as np
from scipy import stats, optimize

np.random.seed(0)
ANN = "/home/sogang/jaehoon/VBPM/dataset_store/beat_this_annotations"
DATASETS = ["ballroom", "asap", "gtzan", "rwc", "hainsworth", "beatles", "hjdb", "smc"]

def load_increments(ds):
    out = []
    for f in sorted(glob.glob(os.path.join(ANN, ds, "annotations", "beats", "*.beats"))):
        t = []
        for line in open(f):
            line = line.strip()
            if line:
                t.append(float(line.split()[0]))
        t = np.asarray(t)
        if t.size < 4:
            continue
        ibi = np.diff(t); ibi = ibi[ibi > 1e-3]
        if ibi.size < 3:
            continue
        out.append(np.diff(np.log(ibi)))
    return np.concatenate(out) if out else np.array([])

data = {ds: load_increments(ds) for ds in DATASETS}
pooled = np.concatenate([data[ds] for ds in DATASETS if data[ds].size])
CLIP = np.log(4.0)   # >4x instantaneous IBI jump = missing/extra beat, not a tempo change

def describe(x):
    return (f"n={x.size:6d}  mean={x.mean():+.4f}  std={x.std():.4f}  "
            f"kurtosis(excess)={stats.kurtosis(x):6.2f}")

print("=" * 80)
print("REAL log-IBI increments  (delta = log IBI_{i+1} - log IBI_i)")
print("=" * 80)
for ds in DATASETS:
    x = data[ds]
    if x.size:
        xf = x[np.abs(x) <= CLIP]
        print(f"  {ds:11s} {describe(x)}   [clip kurt={stats.kurtosis(xf):5.2f}]")
print(f"  {'POOLED':11s} {describe(pooled)}")
pooled_c = pooled[np.abs(pooled) <= CLIP]
print(f"  {'POOLED-clip':11s} {describe(pooled_c)}")

def gauss_fit(x):    return (x.mean(), max(x.std(ddof=0), 1e-9))
def gauss_ll(x, th): m, s = th; return stats.norm.logpdf(x, m, s)
def laplace_fit(x):
    loc = np.median(x); return (loc, max(np.mean(np.abs(x - loc)), 1e-9))
def laplace_ll(x, th): return stats.laplace.logpdf(x, *th)
def studentt_fit(x): return stats.t.fit(x)
def studentt_ll(x, th): return stats.t.logpdf(x, *th)
def nig_fit(x):      return stats.norminvgauss.fit(x)
def nig_ll(x, th):   return stats.norminvgauss.logpdf(x, *th)
def mix_fit(x, iters=300):
    m0, s0 = x.mean(), x.std()
    mu = np.array([m0, m0]); sg = np.array([s0 * .5, s0 * 2.]); w = np.array([.5, .5])
    for _ in range(iters):
        r = np.vstack([w[k] * stats.norm.pdf(x, mu[k], max(sg[k], 1e-9)) for k in range(2)])
        r /= r.sum(0) + 1e-300
        Nk = r.sum(1) + 1e-9
        w = Nk / Nk.sum(); mu = (r * x).sum(1) / Nk
        sg = np.maximum(np.sqrt((r * (x - mu[:, None]) ** 2).sum(1) / Nk), 1e-6)
    return (w, mu, sg)
def mix_ll(x, th):
    w, mu, sg = th
    return np.log(sum(w[k] * stats.norm.pdf(x, mu[k], sg[k]) for k in range(2)) + 1e-300)
def arch_fit(x):
    x = np.asarray(x); x0, x1 = x[:-1], x[1:]
    def nll(p):
        mu, lw, la = p; w = np.exp(lw); a = np.exp(la)
        var = w + a * (x0 - mu) ** 2
        return 0.5 * np.sum(np.log(2 * np.pi * var) + (x1 - mu) ** 2 / var)
    r = optimize.minimize(nll, [x.mean(), np.log(x.var() + 1e-9), np.log(0.1)],
                          method="Nelder-Mead",
                          options=dict(maxiter=8000, xatol=1e-7, fatol=1e-7))
    mu, lw, la = r.x; return (mu, np.exp(lw), np.exp(la))
def arch_ll(x, th):
    mu, w, a = th; x0, x1 = x[:-1], x[1:]
    var = w + a * (x0 - mu) ** 2
    return -0.5 * (np.log(2 * np.pi * var) + (x1 - mu) ** 2 / var)

MODELS = {
    "Gaussian(2)":   (2, gauss_fit,   gauss_ll),
    "Laplace(2)":    (2, laplace_fit, laplace_ll),
    "Student-t(3)":  (3, studentt_fit, studentt_ll),
    "NIG(4)":        (4, nig_fit,     nig_ll),
    "GaussMix2(5)":  (5, mix_fit,     mix_ll),
    "ARCH-Gauss(3)": (3, arch_fit,    arch_ll),
}

def evaluate(x, clip=True, label=""):
    if clip: x = x[np.abs(x) <= CLIP]
    x = np.asarray(x); n = x.size
    rng = np.random.default_rng(1); perm = rng.permutation(n); cut = int(0.8 * n)
    tr_s, te_s = perm[:cut], perm[cut:]
    idx = np.arange(n); tr_o, te_o = idx[:cut], idx[cut:]
    print(f"\n--- {label}  (n={n}, clip={clip}) ---")
    print(f"    {'model':14s}  {'AIC':>11s}  {'BIC':>11s}  {'heldout LL/obs':>14s}")
    rows = []
    for name, (k, fit, ll) in MODELS.items():
        try:
            if name.startswith("ARCH"):
                th = fit(x); llf = ll(x, th).sum(); neff = n - 1
                ho = ll(x[te_o], fit(x[tr_o])).mean()
            else:
                th = fit(x); llf = ll(x, th).sum(); neff = n
                ho = ll(x[te_s], fit(x[tr_s])).mean()
            aic = 2 * k - 2 * llf; bic = k * np.log(neff) - 2 * llf
            rows.append((name, aic, bic, ho))
            print(f"    {name:14s}  {aic:11.1f}  {bic:11.1f}  {ho:14.4f}")
        except Exception as e:
            print(f"    {name:14s}  FAILED: {e}")
    print(f"    >> best BIC: {min(rows, key=lambda r: r[2])[0]}   "
          f"best held-out LL/obs: {max(rows, key=lambda r: r[3])[0]}")
    return rows

print("\n" + "=" * 80)
print("MODEL SELECTION  (lower AIC/BIC better; higher held-out LL/obs better)")
print("=" * 80)
evaluate(pooled, clip=True,  label="POOLED (clipped)")
evaluate(pooled, clip=False, label="POOLED (raw, incl. octave jumps)")
for ds in ["ballroom", "asap", "gtzan", "rwc"]:
    evaluate(data[ds], clip=True, label=ds)

print("\n" + "=" * 80)
print("FITTED PARAMS on POOLED (clipped)")
print("=" * 80)
xc = pooled_c
print("  Gaussian   mu=%+.4f sigma=%.4f" % gauss_fit(xc))
print("  Laplace    loc=%+.4f b=%.4f  (kurtosis 6 vs Gaussian 3)" % laplace_fit(xc))
df, loc, sc = studentt_fit(xc)
print("  Student-t  dof=%.3f loc=%+.4f scale=%.4f  (low dof = heavy tail)" % (df, loc, sc))
print("  NIG        a=%.3f b=%+.3f loc=%+.4f scale=%.4f" % nig_fit(xc))
w, mu, sg = mix_fit(xc)
print("  GaussMix2  w=[%.3f,%.3f] mu=[%+.4f,%+.4f] sigma=[%.4f,%.4f]"
      % (w[0], w[1], mu[0], mu[1], sg[0], sg[1]))
print("  ARCH-Gauss mu=%+.4f omega=%.5f alpha=%.4f  (alpha>0 = volatility clustering)" % arch_fit(xc))
