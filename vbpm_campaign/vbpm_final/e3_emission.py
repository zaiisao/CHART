"""EXPERIMENT 2 -- supervised (madmom-style) observation model p(activation | bar phase, meter).

Fitted DIRECTLY on the TRAIN fold from labels; never through the ELBO.
Nothing in vbpm/, vbpm_fix/, vbpm_arms/ is modified.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms")

from audit_common import load_split, ideal_barphase, ideal_beatlinear_barphase, FPS  # noqa: E402
from vbpm.evaluate import _estimate_meter, f_measure  # noqa: E402

TWO_PI = 2.0 * math.pi
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
METERS = (2, 3, 4)
KMET = 4          # model's num_meters; meter m occupies index m-1


def load_act(split):
    d = np.load(f"{ARMS}/act_{split}.npz", allow_pickle=True)
    out = {}
    for k in d.files:
        if k.endswith("|act"):
            out[k[:-4]] = np.clip(np.asarray(d[k], np.float32), 1e-4, 1.0 - 1e-4)
    return out


def song_phase(s, mode="downbeat"):
    """True bar phase: 0 at each downbeat, linear to 2pi at the next (task spec).
    mode='beat' = piecewise linear between BEATS (madmom's true pointer semantics)."""
    if mode == "beat":
        return ideal_beatlinear_barphase(s["beats"], s["downs"], s["T"], FPS)
    return ideal_barphase(s["downs"], s["T"], FPS, mode="extrap")


def inside_mask(s, T):
    """Frames within the annotated span (avoid extrapolated head/tail)."""
    t = (np.arange(T) + 0.5) / FPS
    if len(s["downs"]) < 2:
        return np.zeros(T, bool)
    return (t >= s["downs"][0]) & (t < s["downs"][-1])


class PhaseEmission:
    """p(o_t | bar phase, meter): per-(meter, phase-bin) table over the 2-channel activation.

    likelihood='bern'  : soft-Bernoulli / cross-entropy on the [0,1] activation
                         log p = sum_c [ a_c log mu_c + (1-a_c) log(1-mu_c) ]
                         (identical functional form to the arm's head_bern emission)
    likelihood='gauss' : Gaussian on logit(a) with per-(bin,channel) mean and sd.

    Bins are per BEAT so time resolution is meter-independent: meter m -> m*bins_per_beat bins.
    """

    def __init__(self, bins_per_beat=12, likelihood="bern", smooth=1.0, prior_w=20.0):
        self.bpb = int(bins_per_beat)
        self.likelihood = likelihood
        self.smooth = smooth
        self.prior_w = prior_w
        self.nb = {m: m * self.bpb for m in METERS}
        self.Bmax = max(self.nb.values())

    # ---------------------------------------------------------------- fitting
    def fit(self, songs, acts, phase_mode="downbeat", shuffle_phase=False, seed=0):
        rngs = np.random.default_rng(seed)
        s1 = {m: np.zeros((self.nb[m], 2)) for m in METERS}
        s2 = {m: np.zeros((self.nb[m], 2)) for m in METERS}
        cnt = {m: np.zeros(self.nb[m]) for m in METERS}
        gm = np.zeros(2); gs = np.zeros(2); gn = 0.0
        n_used = {m: 0 for m in METERS}
        for s in songs:
            a = acts.get(s["stem"])
            if a is None or len(s["downs"]) < 3:
                continue
            T = min(len(a), s["T"])
            m = _estimate_meter(s["beats"], s["downs"])
            if m not in METERS:
                continue
            ph = song_phase(s, phase_mode)
            if ph is None:
                continue
            if shuffle_phase:      # NEGATIVE CONTROL: random per-song phase rotation
                ph = (ph + rngs.random() * TWO_PI) % TWO_PI
            msk = inside_mask(s, T)
            if msk.sum() < 50:
                continue
            x = a[:T][msk]
            v = x if self.likelihood == "bern" else np.log(x / (1.0 - x))
            b = (ph[:T][msk] / TWO_PI * self.nb[m]).astype(int) % self.nb[m]
            np.add.at(s1[m], b, v)
            np.add.at(s2[m], b, v ** 2)
            np.add.at(cnt[m], b, 1.0)
            gm = gm + v.sum(0); gs = gs + (v ** 2).sum(0); gn += len(v)
            n_used[m] += 1
        gmu = gm / max(gn, 1); gvar = gs / max(gn, 1) - gmu ** 2
        self.n_used = n_used
        self.counts = cnt
        self.mu = {}
        self.sd = {}
        for m in METERS:
            c = cnt[m][:, None]
            mu = (s1[m] + self.prior_w * gmu) / (c + self.prior_w)
            var = (s2[m] + self.prior_w * (gvar + gmu ** 2)) / (c + self.prior_w) - mu ** 2
            if self.smooth > 0:                      # circular smoothing over phase bins
                mu = self._circ_smooth(mu)
                var = self._circ_smooth(var)
            self.mu[m] = np.clip(mu, 1e-3, 1 - 1e-3) if self.likelihood == "bern" else mu
            self.sd[m] = np.sqrt(np.maximum(var, 1e-3))
        self._build_coefs()
        return self

    def _circ_smooth(self, x):
        out = np.zeros_like(x)
        for i, w in zip((-1, 0, 1), (0.25, 0.5, 0.25)):
            out = out + w * np.roll(x, i, axis=0)
        return out

    def _build_coefs(self):
        """log p(o|bin,m) is affine (bern) / quadratic (gauss) in the activation ->
        precompute coefficients so a whole song becomes one matmul."""
        self.C = {}
        for m in METERS:
            if self.likelihood == "bern":
                mu = self.mu[m]
                w = np.log(mu / (1 - mu))            # [nb,2]
                c0 = np.log(1 - mu).sum(1)           # [nb]
                self.C[m] = (w, c0)
            else:
                mu, sd = self.mu[m], self.sd[m]
                w1 = mu / sd ** 2
                w2 = -0.5 / sd ** 2
                c0 = (-0.5 * (mu / sd) ** 2 - np.log(sd) - 0.5 * math.log(TWO_PI)).sum(1)
                self.C[m] = (w1, w2, c0)

    # ------------------------------------------------------------- likelihood
    def loglik_table(self, act):
        """[T,2] activation -> dict m -> [T, nb[m]] log p(o_t | bin, m)."""
        x = np.asarray(act, np.float64)
        v = x if self.likelihood == "bern" else np.log(x / (1 - x))
        out = {}
        for m in METERS:
            if self.likelihood == "bern":
                w, c0 = self.C[m]
                out[m] = v @ w.T + c0[None, :]
            else:
                w1, w2, c0 = self.C[m]
                out[m] = v @ w1.T + (v ** 2) @ w2.T + c0[None, :]
        return out

    def loglik_at(self, act, phase, m):
        """log p(o_t | phi_t, m) for a given phase trajectory. -> [T]"""
        tab = self.loglik_table(act)[m]
        b = (np.asarray(phase) / TWO_PI * self.nb[m]).astype(int) % self.nb[m]
        return tab[np.arange(len(b)), b]

    def padded_table(self, act):
        """[T, KMET, Bmax] padded log-lik table for the particle filter (index by meter-1)."""
        tab = self.loglik_table(act)
        T = act.shape[0]
        out = np.full((T, KMET, self.Bmax), -1e9, np.float32)
        for m in METERS:
            out[:, m - 1, :self.nb[m]] = tab[m]
        return out


# ----------------------------------------------------------------- contrast
def obs_contrast(emis, songs, acts, n_off=12, within_beat=False, phase_mode="downbeat"):
    """Geometric-mean per-frame likelihood ratio, TRUE bar phase vs 11 wrong offsets.
    within_beat=True shifts by fractions of a BEAT instead of a BAR."""
    vals = []
    for s in songs:
        a = acts.get(s["stem"])
        if a is None or len(s["downs"]) < 3:
            continue
        T = min(len(a), s["T"])
        m = _estimate_meter(s["beats"], s["downs"])
        if m not in METERS:
            continue
        ph = song_phase(s, phase_mode)
        if ph is None:
            continue
        ph = ph[:T]
        tab = emis.loglik_table(a[:T])[m]
        nb = emis.nb[m]
        idx = np.arange(T)
        b0 = (ph / TWO_PI * nb).astype(int) % nb
        ll_true = tab[idx, b0].mean()
        span = TWO_PI / m if within_beat else TWO_PI
        offs = []
        for k in range(1, n_off):
            pk = (ph + span * k / n_off) % TWO_PI
            bk = (pk / TWO_PI * nb).astype(int) % nb
            offs.append(tab[idx, bk].mean())
        vals.append(math.exp(min(ll_true - float(np.mean(offs)), 60.0)))
    return float(np.mean(vals)), vals
