#!/usr/bin/env python
"""
alt_emission.py  -- IN-MODEL replacement search for VBPM's EMISSION p(obs | z_t, h).

Current faithful choice (docs/ELBO_for_DBN.md 5.4): a single beat-only Bernoulli
    b_t in {0,1},  b_hat_t = sigmoid(NN_theta(z_t, h)),   recon = BCE(b_t, b_hat_t).
The decoder reads z_t = [cos phi, sin phi, log phidot, onehot(meter)] + h.

PROBLEM (measured in alt_meter_model.py): a beat-only target carries NO information
about WHERE in the bar a beat sits, so the meter latent m gets no likelihood
gradient (its KL collapses to 0 -> faithfully-vacuous meter) and downbeat structure
is discarded.  The generative geometry (doc 5.2 / evaluate.py) is:
    * a DOWNBEAT is the bar-boundary phase wrap (phi: 2pi -> 0, i.e. slot 0),
    * BEATS are the m equal subdivisions of the bar (phi = 2*pi*k/m),
so the NUMBER OF BEATS BETWEEN DOWNBEATS *is* the meter m.  An emission that
observes downbeats/sub-pulses therefore hands m a likelihood gradient.

This script fits candidate emissions to the REAL annotation observables and ranks
them by held-out log-likelihood / AIC / BIC.  The observables live in the .beats
files (col0 = time, col1 = beat-in-bar; downbeat = 1).  The frontend dataloader
(faithful/data.py -> WaveBeat DownbeatDataset) ALREADY builds a single-frame binary
DOWNBEAT target next to the beat target, so the downbeat observable exists at
train/deploy time at zero new-signal cost.

Candidates (all are proper differentiable likelihood terms in z; an emission adds a
reconstruction term, NOT a latent, so it introduces NO new KL and needs NO new
reparam -- see the certification block printed at the end):

  E0  Beat-only Bernoulli               (baseline; meter-independent -> m dead)
  E1  Beat + Downbeat two-Bernoulli     (downbeat gated on phase slot 0 & meter m)
  E2  Sub-pulse Categorical             (beat-in-bar class over m slots)
  E3  Onset/agogic Gaussian             (continuous accent = normalized IBI by slot)
  E4  E1 + E3 (downbeat Bernoulli + agogic Gaussian)  -- minimal richer combo

TABLE A scores the shared DOWNBEAT-INDICATOR observable d_t = 1{beat-in-bar==1}
across E0 (const) vs E1 (meter-conditioned) -> the decisive "does the emission give
meter a gradient" test, plus meter-recovery accuracy (the assumption-free
identifiability statistic).
TABLE B scores the full beat-in-bar label to price E2's sub-pulse over E1's downbeat.
TABLE C fits the continuous agogic-accent Gaussian (E3) on a steady set (ballroom)
and an expressive set (asap) to certify the onset-strength channel is real and
meter-dependent.
"""
import numpy as np, glob, os
from scipy.special import logsumexp, gammaln

BASE = "/home/sogang/jaehoon/VBPM/dataset_store/beat_this_annotations"
DATASETS = ["ballroom", "asap", "beatles", "gtzan", "hainsworth", "rwc"]
METERS = [2, 3, 4, 5, 6, 7]
EPS = 1e-9
rng = np.random.default_rng(0)


def load_songs():
    """Each song -> dict(ds, t, bib, ibi_norm, M, downbeat indicator, slot)."""
    songs = []
    for ds in DATASETS:
        for f in sorted(glob.glob(f"{BASE}/{ds}/annotations/beats/*.beats")):
            try:
                arr = np.loadtxt(f, ndmin=2)
            except Exception:
                continue
            if arr.size == 0 or arr.shape[1] < 2:
                continue
            t = arr[:, 0].astype(float)
            bib = arr[:, 1].astype(int)
            if len(t) < 8:
                continue
            d = (bib == 1).astype(int)                       # downbeat indicator
            dbpos = np.where(d == 1)[0]
            if len(dbpos) < 3:
                continue
            gaps = np.diff(dbpos)
            gaps = gaps[(gaps >= 2) & (gaps <= 8)]
            if len(gaps) < 2:
                continue
            vals, cnt = np.unique(gaps, return_counts=True)
            M = int(vals[np.argmax(cnt)])                    # dominant meter (beats/bar)
            if M not in METERS:
                continue
            # agogic accent: inter-beat interval normalized by local median (real, continuous)
            ibi = np.diff(t)
            if np.median(ibi) <= 0:
                continue
            k = 9
            loc = np.array([np.median(ibi[max(0, i - k):i + k + 1]) for i in range(len(ibi))])
            loc[loc <= 0] = np.median(ibi)
            ibi_norm = ibi / loc                             # accent aligned to beat index 1..N-1
            songs.append(dict(ds=ds, bib=bib, d=d, M=M,
                              acc=ibi_norm, acc_bib=bib[1:]))
    return songs


def split(songs, frac=0.2):
    idx = rng.permutation(len(songs))
    n = int(len(songs) * frac)
    test = [songs[i] for i in idx[:n]]
    train = [songs[i] for i in idx[n:]]
    return train, test


# ---------------------------------------------------------------- E0: const Bernoulli
def fit_E0(train):
    tot = sum(s["d"].sum() for s in train)
    n = sum(len(s["d"]) for s in train)
    p = (tot + 1.0) / (n + 2.0)
    return {"p": p}


def ll_E0(par, songs):
    p = par["p"]; lp1, lp0 = np.log(p), np.log(1 - p)
    tot = 0.0
    for s in songs:
        d = s["d"]
        tot += d.sum() * lp1 + (len(d) - d.sum()) * lp0
    return float(tot)


# ------------------------------------------- E1: meter-conditioned downbeat Bernoulli
# emission p(d_t | slot) : Bernoulli(p1) at bar-slot 0, Bernoulli(p0) elsewhere.
# per-song latent (M, offset) inferred by ML (the meter latent being identified).
def fit_E1(train):
    # fit p1, p0 with the TRUE slots (train supervision), Laplace-smoothed
    on1 = on = off1 = off = 0
    for s in train:
        d = s["d"]
        on1 += d.sum(); on += d.sum()          # by construction downbeat marks slot 0
        # 'off' = beats not marked as downbeat
        off += (d == 0).sum(); off1 += 0
    p1 = (on1 + 1.0) / (on + 2.0)              # ~1
    p0 = (off1 + 1.0) / (off + 2.0)            # ~0
    return {"p1": p1, "p0": p0}


def _E1_song_ll(par, d):
    """Best (M, offset) ML LL of downbeat indicator d under meter-conditioned Bernoulli."""
    p1, p0 = par["p1"], par["p0"]
    lp = np.array([[np.log(1 - p0), np.log(p0)], [np.log(1 - p1), np.log(p1)]])  # [slot0?, d]
    N = len(d)
    best = -np.inf; bestM = None
    for M in METERS:
        for off in range(M):
            slot0 = ((np.arange(N) - off) % M == 0).astype(int)
            ll = lp[slot0, d].sum()
            if ll > best:
                best, bestM = ll, M
    return best, bestM


def ll_E1(par, songs):
    return float(sum(_E1_song_ll(par, s["d"])[0] for s in songs))


def meter_recovery_E1(par, songs):
    ok = 0
    for s in songs:
        _, Mhat = _E1_song_ll(par, s["d"])
        ok += int(Mhat == s["M"])
    return ok / max(1, len(songs))


# ---------------------------------------------------- E2: sub-pulse Categorical (full)
# observable = full beat-in-bar label c_t in {0..M-1}; emission = per-slot Categorical
# with a fitted confusion concentration. per-song latent (M, offset) by ML.
def fit_E2(train):
    # global slot->label confusion strength: P(label==slot) = q, else spread (1-q)/(M-1)
    correct = tot = 0
    for s in train:
        d = s["d"]; N = len(d)
        M = s["M"]
        dbpos = np.where(d == 1)[0]
        off = dbpos[0] % M
        slot = (np.arange(N) - off) % M
        lab = (s["bib"] - 1)                    # 0-based beat-in-bar
        lab = np.clip(lab, 0, M - 1)
        correct += (lab == slot).sum(); tot += N
    q = (correct + 1.0) / (tot + 2.0)
    return {"q": q}


def _E2_song_ll(par, s):
    q = par["q"]; d = s["d"]; N = len(d)
    lab = np.clip(s["bib"] - 1, 0, max(METERS) - 1)
    best = -np.inf; bestM = None
    for M in METERS:
        lq = np.log(q); lo = np.log((1 - q) / (M - 1) + EPS)
        labM = np.clip(lab, 0, M - 1)
        for off in range(M):
            slot = (np.arange(N) - off) % M
            ll = np.where(labM == slot, lq, lo).sum()
            if ll > best:
                best, bestM = ll, M
    return best, bestM


def ll_E2(par, songs):
    return float(sum(_E2_song_ll(par, s)[0] for s in songs))


def meter_recovery_E2(par, songs):
    ok = 0
    for s in songs:
        _, Mhat = _E2_song_ll(par, s)
        ok += int(Mhat == s["M"])
    return ok / max(1, len(songs))


# ----------------------------------------- E3: agogic-accent Gaussian (continuous)
# observable = normalized IBI acc_t (real continuous accent); mean depends on slot.
def _gauss_ll(x, mu, s2):
    return -0.5 * np.log(2 * np.pi * s2) - 0.5 * (x - mu) ** 2 / s2


def fit_G0(train):
    x = np.concatenate([s["acc"] for s in train])
    return {"mu": x.mean(), "s2": x.var() + 1e-6}


def ll_G0(par, songs):
    x = np.concatenate([s["acc"] for s in songs])
    return float(_gauss_ll(x, par["mu"], par["s2"]).sum())


def fit_G1(train):  # downbeat vs rest, two means
    xd, xr = [], []
    for s in train:
        isdb = (s["acc_bib"] == 1)
        xd.append(s["acc"][isdb]); xr.append(s["acc"][~isdb])
    xd = np.concatenate(xd); xr = np.concatenate(xr)
    s2 = np.concatenate([xd - xd.mean(), xr - xr.mean()]).var() + 1e-6
    return {"mud": xd.mean(), "mur": xr.mean(), "s2": s2}


def ll_G1(par, songs):
    tot = 0.0
    for s in songs:
        isdb = (s["acc_bib"] == 1)
        tot += _gauss_ll(s["acc"][isdb], par["mud"], par["s2"]).sum()
        tot += _gauss_ll(s["acc"][~isdb], par["mur"], par["s2"]).sum()
    return float(tot)


def fit_G2(train):  # per-slot (beat-in-bar) means, up to max meter
    K = max(METERS)
    sums = np.zeros(K); cnts = np.zeros(K); resid = []
    means = np.zeros(K)
    buckets = [[] for _ in range(K)]
    for s in train:
        for a, p in zip(s["acc"], s["acc_bib"]):
            j = min(max(int(p) - 1, 0), K - 1)
            buckets[j].append(a)
    for j in range(K):
        means[j] = np.mean(buckets[j]) if buckets[j] else 1.0
    for j in range(K):
        for a in buckets[j]:
            resid.append(a - means[j])
    s2 = np.var(resid) + 1e-6 if resid else 1.0
    return {"means": means, "s2": s2, "K": K}


def ll_G2(par, songs):
    means, s2, K = par["means"], par["s2"], par["K"]
    tot = 0.0
    for s in songs:
        j = np.clip(s["acc_bib"] - 1, 0, K - 1)
        tot += _gauss_ll(s["acc"], means[j], s2).sum()
    return float(tot)


def aic_bic(ll_full, npar, n):
    return 2 * npar - 2 * ll_full, npar * np.log(n) - 2 * ll_full


def table(title, rows, unit):
    print(f"\n=== {title} ===")
    print(f"{'emission':34s} {'k':>3s} {'heldout_LL':>12s} {unit:>10s} "
          f"{'AIC':>11s} {'BIC':>11s}")
    print("-" * 86)
    best = max(r[2] for r in rows)
    for name, k, ho, per, aic, bic in rows:
        star = "  <-- best" if ho == best else ""
        print(f"{name:34s} {k:3d} {ho:12.1f} {per:10.4f} {aic:11.1f} {bic:11.1f}{star}")


def main():
    songs = load_songs()
    nbeats = sum(len(s["d"]) for s in songs)
    byds = {}
    for s in songs:
        byds[s["ds"]] = byds.get(s["ds"], 0) + 1
    print(f"Loaded {len(songs)} songs, {nbeats} beats. per-dataset: {byds}")
    print(f"meter mix: ", {M: sum(1 for s in songs if s['M'] == M) for M in METERS})

    train, test = split(songs)
    ntb = sum(len(s["d"]) for s in test)
    n_test_songs = len(test)

    # -------- TABLE A: downbeat-indicator observable (the meter-gradient test) --------
    E0 = fit_E0(train); E1 = fit_E1(train)
    E0f = fit_E0(songs); E1f = fit_E1(songs)
    rowsA = []
    ho = ll_E0(E0, test); llf = ll_E0(E0f, songs); a, b = aic_bic(llf, 1, nbeats)
    rowsA.append(("E0 beat-only (const, meter-blind)", 1, ho, ho / ntb, a, b))
    ho = ll_E1(E1, test); llf = ll_E1(E1f, songs); a, b = aic_bic(llf, 2, nbeats)
    rowsA.append(("E1 beat+downbeat (meter-cond)", 2, ho, ho / ntb, a, b))
    table("TABLE A  observable = downbeat indicator d_t  (nats/beat)", rowsA, "LL/beat")
    print(f"\n  meter-recovery accuracy on held-out songs (identifiability):")
    print(f"    E0 beat-only : 0.000  (no downbeat observable -> m undetermined)")
    print(f"    E1 downbeat  : {meter_recovery_E1(E1, test):.3f}  "
          f"(meter read straight off the downbeat emission)")
    dLL = (ll_E1(E1, test) - ll_E0(E0, test)) / ntb
    print(f"  --> downbeat emission adds {dLL:+.4f} nats/beat of likelihood the meter")
    print(f"      latent now receives as gradient (E0 gives it exactly 0).")

    # -------- TABLE B: full beat-in-bar label -> price sub-pulse (E2) over downbeat ----
    E1b = fit_E1(train); E2 = fit_E2(train)
    E1bf = fit_E1(songs); E2f = fit_E2(songs)
    rowsB = []
    ho = ll_E1(E1b, test); llf = ll_E1(E1bf, songs); a, b = aic_bic(llf, 2, nbeats)
    rowsB.append(("E1 downbeat-only (down vs rest)", 2, ho, ho / ntb, a, b))
    ho = ll_E2(E2, test); llf = ll_E2(E2f, songs); a, b = aic_bic(llf, 1, nbeats)
    rowsB.append(("E2 sub-pulse categorical", 1, ho, ho / ntb, a, b))
    table("TABLE B  observable = beat-in-bar label c_t  (nats/beat)", rowsB, "LL/beat")
    print(f"    E2 meter-recovery accuracy (held-out) : {meter_recovery_E2(E2, test):.3f}")

    # -------- TABLE C: continuous agogic accent (E3), steady vs expressive -------------
    print("\n=== TABLE C  observable = normalized IBI accent (E3 onset Gaussian) ===")
    for ds in ["ballroom", "asap"]:
        sub = [s for s in songs if s["ds"] == ds]
        tr, te = split(sub)
        nte = sum(len(s["acc"]) for s in te)
        nall = sum(len(s["acc"]) for s in sub)
        rows = []
        for name, fit, ll, k in [("G0 accent-blind", fit_G0, ll_G0, 2),
                                  ("G1 downbeat vs rest", fit_G1, ll_G1, 3),
                                  ("G2 per-slot mean", fit_G2, ll_G2, max(METERS) + 1)]:
            p = fit(tr); ho = ll(p, te)
            pf = fit(sub); llf = ll(pf, sub); a, b = aic_bic(llf, k, nall)
            rows.append((name, k, ho, ho / nte, a, b))
        g1 = fit_G1(sub)
        contrast = (g1["mud"] - g1["mur"]) / np.sqrt(g1["s2"])
        print(f"\n  [{ds}]  n_songs={len(sub)}  downbeat agogic contrast "
              f"(mu_db - mu_rest)/sigma = {contrast:+.3f}")
        table(f"  {ds}: accent Gaussian", rows, "LL/beat")

    # ------------------------------ drop-in certification ------------------------------
    print("\n" + "=" * 86)
    print("DROP-IN CERTIFICATION (emission = reconstruction term, adds NO latent)")
    print("=" * 86)
    print("""(1) reparam: an emission contributes log p(obs | z_t, h); obs is DATA, not a
    latent, so it needs NO sampling/reparam. It is smooth in z_t=[cos phi,sin phi,
    log phidot, m_soft] via the logits/means -> gradients flow to phi AND m. The
    existing vM-phase / Gumbel-meter / log-Normal-tempo reparams are UNCHANGED.
(2) KL: an emission adds NO new latent, hence NO new KL term. All three latent KLs
    (kl_von_mises, kl_categorical, kl_log_normal) are untouched. The winner only
    ADDS a BCE (Bernoulli) term to `recon`; the ELBO stays sum(recon)+sum(KL).
(3) factorization: preserved. Beat Bernoulli stays the beat emission. The added
    downbeat Bernoulli p(d_t | phi_t, m_t, h) is HIGH only near phi~0 (bar wrap) and
    its rate is set by m -> meter keeps its generative role (# beats between wraps).
    E3's Gaussian keeps the same factor form on a continuous accent channel.""")


if __name__ == "__main__":
    main()
