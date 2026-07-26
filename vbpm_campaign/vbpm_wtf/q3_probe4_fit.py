"""PROBE 4 (a)+(b): fit p(activation | bar phase, meter) SUPERVISED on the train fold,
then measure the phase CONTRAST with exactly the arm's metric on the held-out eval fold.
"""
import json, math, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms")
from audit_common import load_split, ideal_barphase, ideal_beatlinear_barphase, FPS
from vbpm.evaluate import _estimate_meter

ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
OUT = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf"
TWO_PI = 2 * math.pi
KM = 4                       # meter classes, index = m-1


def song_phase(s, T, kind):
    if kind == "bar":
        return ideal_barphase(s["downs"], T, FPS, mode="extrap")
    return ideal_beatlinear_barphase(s["beats"], s["downs"], T, FPS)


def fit_table(songs, acts, nbins, kind, prior_strength=5.0):
    """Per (meter, phase-bin, channel) Bernoulli mean of the activation. madmom's obs model."""
    num = np.zeros((KM, nbins, 2)); den = np.zeros((KM, nbins, 1))
    gmean = np.zeros(2); gn = 0
    for s in songs:
        T = s["T"]; a = acts[s["stem"]]
        ref = s["beats"]; dref = s["downs"]
        if len(dref) < 3 or len(ref) < 4: continue
        ph = song_phase(s, T, kind)
        if ph is None: continue
        m = _estimate_meter(ref, dref)
        ok = ph > -1
        if kind == "beatlin":                       # beat-linear phase is 0 outside the span
            lo = int(dref[0] * FPS); hi = int(dref[-1] * FPS)
            ok = np.zeros(T, bool); ok[lo:hi] = True
        b = np.floor(ph[ok] / TWO_PI * nbins).astype(int) % nbins
        aa = a[ok]
        np.add.at(num[m - 1], b, aa); np.add.at(den[m - 1], b, 1.0)
        gmean += aa.sum(0); gn += len(aa)
    gmean = gmean / max(gn, 1)
    mu = (num + prior_strength * gmean[None, None, :]) / (den + prior_strength)
    return np.clip(mu, 1e-4, 1 - 1e-4), gmean, den


class BinEmission(torch.nn.Module):
    """p(o|phi,m) from the fitted table. Same Bernoulli likelihood the VAE emission uses."""
    def __init__(self, mu, dev="cpu", temper=1.0):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32, device=dev))
        self.nbins = mu.shape[1]; self.K = KM; self.temper = temper

    def obs_logp(self, z_feat, o_t):
        phi = torch.atan2(z_feat[:, 1], z_feat[:, 0]) % TWO_PI
        b = torch.clamp((phi / TWO_PI * self.nbins).long(), 0, self.nbins - 1)
        m = z_feat[:, 3:3 + self.K].argmax(-1)
        mu = self.mu[m, b]                                   # [N,2]
        ll = o_t * torch.log(mu) + (1 - o_t) * torch.log(1 - mu)
        return self.temper * ll.sum(-1)

    def z_features(self, meter_soft, phi, log_tempo):
        return torch.cat([torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1),
                          log_tempo.unsqueeze(-1) * 0, meter_soft], -1)


def contrast(em, songs, acts, kind, n_off=12, dev="cpu"):
    """EXACTLY the arm's obs_contrast_song, with this emission substituted."""
    out = []
    for s in songs:
        T = s["T"]; dref = s["downs"]
        if len(dref) < 3: continue
        ph = song_phase(s, T, kind)
        if ph is None: continue
        m = _estimate_meter(s["beats"], dref)
        o = torch.tensor(acts[s["stem"]], dtype=torch.float32, device=dev)
        mt = F.one_hot(torch.full((T,), m - 1, device=dev), KM).float()
        lt = torch.zeros(T, device=dev)
        p = torch.tensor(ph, dtype=torch.float32, device=dev)
        lt_true = float(em.obs_logp(em.z_features(mt, p, lt), o).mean())
        offs = [float(em.obs_logp(em.z_features(mt, (p + TWO_PI * k / n_off) % TWO_PI, lt), o).mean())
                for k in range(1, n_off)]
        out.append(math.exp(min(lt_true - float(np.mean(offs)), 60.0)))
    return float(np.mean(out)), out


def main():
    tr = load_split("train"); ev = load_split("eval")
    dtr = np.load(f"{ARMS}/act_train.npz", allow_pickle=True)
    dev_ = np.load(f"{ARMS}/act_eval.npz", allow_pickle=True)
    atr = {s["stem"]: np.clip(np.asarray(dtr[s["stem"] + "|act"], np.float32), 1e-4, 1 - 1e-4) for s in tr}
    aev = {s["stem"]: np.clip(np.asarray(dev_[s["stem"] + "|act"], np.float32), 1e-4, 1 - 1e-4) for s in ev}
    print(f"train {len(tr)}  eval {len(ev)}")
    res = {}
    for kind in ("bar", "beatlin"):
        for nb in (12, 36, 72, 144):
            mu, gm, den = fit_table(tr, atr, nb, kind)
            ctr_tr, _ = contrast(BinEmission(mu), tr, atr, kind)
            ctr_ev, per = contrast(BinEmission(mu), ev, aev, kind)
            m4 = mu[3]
            print(f"[{kind:7s} nbins={nb:3d}] global act mean {gm.round(4)}  "
                  f"m=4 beat-ch mu range [{m4[:,0].min():.4f},{m4[:,0].max():.4f}] "
                  f"db-ch [{m4[:,1].min():.4f},{m4[:,1].max():.4f}]  "
                  f"CONTRAST train={ctr_tr:.3f}  EVAL={ctr_ev:.3f}")
            res[f"{kind}_{nb}"] = dict(contrast_train=ctr_tr, contrast_eval=ctr_ev,
                                       per_song=per, mu_shape=list(mu.shape))
            if nb == 36:
                np.save(f"{OUT}/emis_{kind}_{nb}.npy", mu)
                print("     m=4 beat-channel mu by bin: " +
                      " ".join(f"{v:.3f}" for v in m4[:, 0]))
                print("     m=4 db  -channel mu by bin: " +
                      " ".join(f"{v:.3f}" for v in m4[:, 1]))
                print("     bin counts per meter: " + str(den.sum(1).astype(int).ravel().tolist()))
    json.dump(res, open(f"{OUT}/probe4_fit.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
