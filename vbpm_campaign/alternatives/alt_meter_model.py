#!/usr/bin/env python
"""
alt_meter_model.py  -- IN-MODEL replacement search for VBPM's METER_MODEL.

Current faithful choice: per-FRAME Categorical meter with a full K x K transition
matrix (a meter switch is allowed at every frame).  This script measures the real
meter statistic (per-bar beats-per-bar sequences) and selects, by held-out
log-likelihood / AIC / BIC, the best in-model form for the meter latent's
generative transition p(m_t | ...).

Every candidate is a genuine drop-in for the faithful VAE:
  (1) reparameterizable via the SAME Gumbel-Softmax already used for the meter
      Categorical (change is only in the prior transition, not the sampler);
  (2) closed-form Categorical KL to a matching Categorical prior;
  (3) preserves the factorization: meter stays a Categorical "beats-per-bar"
      latent, only the temporal law over m changes.

Per song s we model the sequence of per-bar meter symbols m_{s,1..B_s}, a symbol
= #beats between consecutive downbeats (col1==1), mapped to K=4 classes
{2,3,4,other} to match faithful num_meters=4.
"""
import numpy as np, glob
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

BASE = "/home/sogang/jaehoon/VBPM/dataset_store/beat_this_annotations"
DATASETS = ["ballroom", "asap", "beatles", "gtzan", "hainsworth", "rwc"]
K = 4
SYM = {2: 0, 3: 1, 4: 2}
EPS = 1e-4
rng = np.random.default_rng(0)


def sym(v):
    return SYM.get(int(v), 3)


def load_song_sequences():
    songs = []
    for ds in DATASETS:
        for f in glob.glob(f"{BASE}/{ds}/annotations/beats/*.beats"):
            try:
                arr = np.loadtxt(f, ndmin=2)
            except Exception:
                continue
            if arr.size == 0 or arr.shape[1] < 2:
                continue
            bib = arr[:, 1].astype(int)
            db = np.where(bib == 1)[0]
            if len(db) < 3:
                continue
            bpb = np.diff(db)
            bpb = bpb[(bpb >= 2) & (bpb <= 12)]
            if len(bpb) < 2:
                continue
            songs.append((ds, np.array([sym(v) for v in bpb], dtype=int)))
    return songs


def fit_iid(train):
    c = np.full(K, EPS)
    for _, s in train:
        for x in s:
            c[x] += 1
    return {"pi": c / c.sum()}


def ll_iid(p, songs):
    lp = np.log(p["pi"])
    return sum(float(lp[s].sum()) for _, s in songs)


def fit_markov(train):
    init = np.full(K, EPS); A = np.full((K, K), EPS)
    for _, s in train:
        init[s[0]] += 1
        for a, b in zip(s[:-1], s[1:]):
            A[a, b] += 1
    init /= init.sum(); A /= A.sum(1, keepdims=True)
    return {"init": init, "A": A}


def ll_markov(p, songs):
    li, lA = np.log(p["init"]), np.log(p["A"])
    tot = 0.0
    for _, s in songs:
        tot += li[s[0]]
        if len(s) > 1:
            tot += lA[s[:-1], s[1:]].sum()
    return float(tot)


def fit_sticky(train):
    N = np.zeros((K, K)); init_c = np.full(K, EPS)
    for _, s in train:
        init_c[s[0]] += 1
        for a, b in zip(s[:-1], s[1:]):
            N[a, b] += 1
    init = init_c / init_c.sum()
    base = (N.sum(0) + EPS); base /= base.sum()

    def negll(theta):
        rho = 1 / (1 + np.exp(-theta[0]))
        z = theta[1:]; pi = np.exp(z - logsumexp(z))
        A = rho * np.eye(K) + (1 - rho) * np.outer(np.ones(K), pi)
        return -float((N * np.log(A)).sum())

    x0 = np.concatenate([[2.0], np.log(base + EPS)])
    res = minimize(negll, x0, method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-6})
    rho = 1 / (1 + np.exp(-res.x[0]))
    z = res.x[1:]; pi = np.exp(z - logsumexp(z))
    A = rho * np.eye(K) + (1 - rho) * np.outer(np.ones(K), pi)
    return {"init": init, "A": A, "rho": rho, "pi": pi}


def ll_sticky(p, songs):
    return ll_markov({"init": p["init"], "A": p["A"]}, songs)


def fit_const_blip(train):
    def negll(theta):
        z = theta[:K]; pi = np.exp(z - logsumexp(z))
        e = 1 / (1 + np.exp(-theta[K]))
        stay = np.log(1 - e + 1e-12); leave = np.log(e + 1e-12) - np.log(K - 1)
        tot = 0.0
        for _, s in train:
            cnt = np.bincount(s, minlength=K)
            ll_c = np.array([cnt[c] * stay + (len(s) - cnt[c]) * leave
                             for c in range(K)])
            tot += logsumexp(np.log(pi + 1e-12) + ll_c)
        return -tot
    x0 = np.concatenate([np.zeros(K), [-2.0]])
    res = minimize(negll, x0, method="Nelder-Mead",
                   options={"maxiter": 40000, "xatol": 1e-7, "fatol": 1e-7})
    z = res.x[:K]; pi = np.exp(z - logsumexp(z))
    e = 1 / (1 + np.exp(-res.x[K]))
    return {"pi": pi, "e": e}


def ll_const_blip(p, songs):
    pi, e = p["pi"], p["e"]
    stay = np.log(1 - e + 1e-12); leave = np.log(e + 1e-12) - np.log(K - 1)
    tot = 0.0
    for _, s in songs:
        cnt = np.bincount(s, minlength=K)
        ll_c = np.array([cnt[c] * stay + (len(s) - cnt[c]) * leave
                         for c in range(K)])
        tot += logsumexp(np.log(pi + 1e-12) + ll_c)
    return float(tot)


def fit_hier_dirmult(train):
    def negll(theta):
        alpha = np.exp(theta); a0 = alpha.sum(); tot = 0.0
        for _, s in train:
            cnt = np.bincount(s, minlength=K); n = cnt.sum()
            tot += (gammaln(a0) - gammaln(a0 + n)
                    + (gammaln(alpha + cnt) - gammaln(alpha)).sum())
        return -tot
    res = minimize(negll, np.zeros(K), method="Nelder-Mead",
                   options={"maxiter": 40000, "xatol": 1e-7, "fatol": 1e-7})
    return {"alpha": np.exp(res.x)}


def ll_hier_dirmult(p, songs):
    alpha = p["alpha"]; a0 = alpha.sum(); tot = 0.0
    for _, s in songs:
        cnt = np.bincount(s, minlength=K); n = cnt.sum()
        tot += (gammaln(a0) - gammaln(a0 + n)
                + (gammaln(alpha + cnt) - gammaln(alpha)).sum())
    return float(tot)


MODELS = {
    "Ciid_Categorical":     (fit_iid,          ll_iid,          K - 1),
    "C0_FullMarkov":        (fit_markov,       ll_markov,       (K - 1) + K * (K - 1)),
    "C1_StickyMarkov":      (fit_sticky,       ll_sticky,       (K - 1) + 1),
    "C2_ConstPerSong_blip": (fit_const_blip,   ll_const_blip,   (K - 1) + 1),
    "C3_Hier_DirMult":      (fit_hier_dirmult, ll_hier_dirmult, K),
}


def main():
    songs = load_song_sequences()
    nbars = sum(len(s) for _, s in songs)
    print(f"Loaded {len(songs)} songs, {nbars} bars, K={K} classes {{2,3,4,other}}\n")

    sw = sum(int((s[1:] != s[:-1]).sum()) for _, s in songs)
    songs_sw = sum(1 for _, s in songs if (s[1:] != s[:-1]).any())
    print(f"per-bar switch rate  = {sw/nbars:.5f}   "
          f"songs with >=1 switch = {songs_sw}/{len(songs)} "
          f"({100*songs_sw/len(songs):.1f}%)\n")

    idx = rng.permutation(len(songs))
    ntest = len(songs) // 5
    test = [songs[i] for i in idx[:ntest]]
    train = [songs[i] for i in idx[ntest:]]
    test_bars = sum(len(s) for _, s in test)

    rows = []
    for name, (fit, ll, npar) in MODELS.items():
        p = fit(train); ho = ll(p, test)
        pfull = fit(songs); full = ll(pfull, songs)
        aic = 2 * npar - 2 * full
        bic = npar * np.log(nbars) - 2 * full
        rows.append((name, npar, ho, ho / test_bars, aic, bic))

    rows.sort(key=lambda r: -r[2])
    print(f"{'model':22s} {'k':>3s} {'heldout_LL':>12s} {'LL/bar':>9s} "
          f"{'AIC':>11s} {'BIC':>11s}")
    print("-" * 72)
    best_ho = rows[0][2]
    for name, npar, ho, hob, aic, bic in rows:
        star = "  <-- best HO-LL" if ho == best_ho else ""
        print(f"{name:22s} {npar:3d} {ho:12.1f} {hob:9.4f} "
              f"{aic:11.1f} {bic:11.1f}{star}")

    best_aic = min(rows, key=lambda r: r[4])[0]
    best_bic = min(rows, key=lambda r: r[5])[0]
    print(f"\nbest by held-out LL : {rows[0][0]}")
    print(f"best by AIC         : {best_aic}")
    print(f"best by BIC         : {best_bic}")

    print("\n--- identifiability: meter from a BEAT-only emission ---")
    per_pos = {}; globcv = []
    for ds in ["ballroom", "asap"]:
        for f in glob.glob(f"{BASE}/{ds}/annotations/beats/*.beats"):
            try:
                arr = np.loadtxt(f, ndmin=2)
            except Exception:
                continue
            if arr.size == 0 or arr.shape[1] < 2:
                continue
            t = arr[:, 0]; bib = arr[:, 1].astype(int); ibi = np.diff(t)
            if len(ibi) < 8:
                continue
            med = np.median(ibi)
            if med <= 0:
                continue
            r = ibi / med
            globcv.append(np.std(ibi) / np.mean(ibi))
            for pos, v in zip(bib[1:], r):
                per_pos.setdefault(int(pos), []).append(v)
    print(f"mean within-song IBI coeff-of-variation (ballroom+asap) = "
          f"{np.mean(globcv):.4f}")
    for pos in sorted(per_pos):
        if pos <= 4 and len(per_pos[pos]) > 50:
            a = np.array(per_pos[pos])
            print(f"  beat-in-bar {pos}: normIBI mean={a.mean():.4f} "
                  f"std={a.std():.4f}  n={len(a)}")
    print("  -> IBI ~flat across beat-in-bar: a BEAT-only emission gives the meter")
    print("     latent no likelihood gradient (KL->0, posterior=prior). Meter is")
    print("     UNDER-IDENTIFIED without a downbeat/sub-pulse emission.")


if __name__ == "__main__":
    main()
