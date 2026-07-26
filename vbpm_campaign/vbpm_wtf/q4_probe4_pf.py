"""PROBE 4 (c): plug the SUPERVISED observation model into VB.particle_filter.

Cells (all on eval fold 0, full length, mandatory density-matched blind controls):
  vae|trained     VAE-learned emission + trained prior   (reproduce the 0.385 baseline)
  sup|trained     SUPERVISED emission  + trained prior
  sup|tight       SUPERVISED emission  + madmom-like tight bar-pointer transition
  oracle|trained  von Mises oracle     + trained prior   (reproduce the ~0.96 ceiling)
  oracle|tight    von Mises oracle     + tight transition
  flat|tight      NO evidence (constant emission) + tight transition  (control)
"""
import argparse, json, math, sys, time, types
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms")
import variant_b as VB
from audit_common import load_split, ideal_barphase, FPS
from common import smooth_phase
from vbpm.evaluate import beats_from_barphase, downbeats_from_barphase, f_measure, _estimate_meter, metronome
from arm_i import blind_grid_controls, phase_diag, M
from q3_probe4_fit import BinEmission, KM

DEV = "cuda:0"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
OUT = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf"
TWO_PI = 2 * math.pi


# --------------------------------------------------------------- tight (madmom-like) prior
class TightPrior(VB.BarPointerVAE_B):
    """Same class, but the audio-driven prior heads are replaced by CONSTANTS:
    a near-deterministic bar-pointer advance and a slow tempo random walk."""
    def __init__(self, rho=0.99, s_lv=0.005, s_dv=1e-4, stick=0.999, **kw):
        super().__init__(h_dim=2, hidden=128, num_meters=4, obs_dim=2, obs_type="bern", **kw)
        self.rho_c, self.s_lv_c, self.s_dv_c, self.stick = rho, s_lv, s_dv, stick
        with torch.no_grad():                       # meter init: uniform over m in {2,3,4}
            self.prior_init_head[2].weight.zero_()
            self.prior_init_head[2].bias.zero_()
            self.prior_init_head[2].bias[:4] = torch.tensor([-20.0, 0.0, 0.0, 0.0])

    def encode_prior(self, h):
        return h.new_zeros(h.shape[0], h.shape[1], self.hidden)

    def _c(self, ctx, v):
        return torch.full((ctx.shape[0],), v, device=ctx.device, dtype=ctx.dtype)

    def prior_phase_conc(self, ctx): return self._c(ctx, self.rho_c)
    def prior_level_scale(self, ctx): return self._c(ctx, self.s_lv_c)
    def prior_dev_scale(self, ctx): return self._c(ctx, self.s_dv_c)
    def prior_dev_coef(self, ctx): return self._c(ctx, 0.0)
    def level_ar(self): return torch.tensor(1.0, device=self.z0.device)
    def tempo_dof(self): return torch.tensor(30.0, device=self.z0.device)

    def meter_prior_logp(self, meter_prev, phi_t, phi_prev, ctx):
        n = meter_prev.shape[0]
        off = math.log((1 - self.stick) / 2.0)
        P = torch.full((n, self.K), off, device=meter_prev.device)
        P[:, 0] = -30.0                              # m=1 disallowed
        return P + (meter_prev * (math.log(self.stick) - off))


# --------------------------------------------------------------- emissions
def make_sup_logp(mu, dev, temper=1.0):
    em = BinEmission(mu, dev=dev, temper=temper).to(dev)
    def f(self, z_feat, o_t):
        return em.obs_logp(z_feat, o_t)
    return f


def make_oracle_logp(kappa):
    def f(self, z_feat, o_t):        # o_t = [cos phi_true, sin phi_true]
        return kappa * (z_feat[:, 0] * o_t[:, 0] + z_feat[:, 1] * o_t[:, 1])
    return f


def flat_logp(self, z_feat, o_t):
    return z_feat[:, 0] * 0.0


# --------------------------------------------------------------- eval
@torch.no_grad()
def run_cell(model, songs, obs_fn, K, alpha, tag, smooth=5, seed=1234):
    rows = []
    for i, s in enumerate(songs):
        T = s["T"]
        ref = s["beats"][s["beats"] < T / FPS]; dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 3 or len(dref) < 3:
            continue
        obs = obs_fn(s, T)
        if obs is None:
            continue
        h = obs[:, :, :2] if obs.shape[-1] >= 2 else obs
        torch.manual_seed(seed + i)
        out = VB.particle_filter(model, h, obs, K=K, alpha=alpha)
        m = _estimate_meter(ref, dref)
        row = dict(stem=s["stem"], dataset=s["dataset"], T=T, meter=int(m),
                   n_true=len(ref), n_true_db=len(dref), ess=float(out["ess"]),
                   metronome_F=f_measure(ref, metronome(T, FPS)))
        for rd, ph in (("mean", out["phase_mean"].numpy()),
                       ("map", out["phase_map"].numpy()),
                       ("smooth", smooth_phase(out["phase_mean"].numpy(), smooth))):
            est = beats_from_barphase(ph, m, FPS); dest = downbeats_from_barphase(ph, FPS)
            b0, bb = blind_grid_controls(ref, T, len(est))
            d0, db = blind_grid_controls(dref, T, len(dest))
            pd = phase_diag(ph)
            row.update({f"{rd}|beat_F": f_measure(ref, est),
                        f"{rd}|db_F": f_measure(dref, dest),
                        f"{rd}|n_est": len(est), f"{rd}|n_est_db": len(dest),
                        f"{rd}|blind0": b0, f"{rd}|blind_best": bb,
                        f"{rd}|blind_db0": d0, f"{rd}|blind_db_best": db,
                        f"{rd}|frac_neg": pd["frac_neg"], f"{rd}|mean_adv": pd["mean_adv"],
                        f"{rd}|jit_adv": pd["jitter_over_adv"]})
        rows.append(row)
    return rows


def summarize(rows, rd, tag):
    ne = sum(r[f"{rd}|n_est"] for r in rows); nt = sum(r["n_true"] for r in rows)
    ned = sum(r[f"{rd}|n_est_db"] for r in rows); ntd = sum(r["n_true_db"] for r in rows)
    bf, bb = M(rows, f"{rd}|beat_F"), M(rows, f"{rd}|blind_best")
    dfm, dbb = M(rows, f"{rd}|db_F"), M(rows, f"{rd}|blind_db_best")
    return dict(cell=tag, readout=rd, beat_F=bf, downbeat_F=dfm,
                n_ratio=ne / max(nt, 1), n_ratio_db=ned / max(ntd, 1),
                blind0=M(rows, f"{rd}|blind0"), blind_best=bb, margin=bf - bb,
                blind_db_best=dbb, margin_db=dfm - dbb,
                frac_neg=M(rows, f"{rd}|frac_neg"), mean_adv=M(rows, f"{rd}|mean_adv"),
                jit_adv=M(rows, f"{rd}|jit_adv"), ess=M(rows, "ess"),
                metronome=M(rows, "metronome_F"), n_songs=len(rows))


def pr(d):
    print(f"    [{d['cell']:16s} {d['readout']:6s}] beat_F={d['beat_F']:.4f} db_F={d['downbeat_F']:.4f} "
          f"n_ratio={d['n_ratio']:.3f} blindbest={d['blind_best']:.4f} MARGIN={d['margin']:+.4f} | "
          f"db_blind={d['blind_db_best']:.4f} MARGIN_db={d['margin_db']:+.4f} | "
          f"frac_neg={d['frac_neg']:.3f} jit/adv={d['jit_adv']:.1f} ESS={d['ess']:.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--nbins", type=int, default=72)
    ap.add_argument("--rho", type=float, default=0.99)
    ap.add_argument("--s_lv", type=float, default=0.005)
    ap.add_argument("--kappa", type=float, default=8.0)
    ap.add_argument("--cells", nargs="+", default=["vae|trained", "sup|trained", "sup|tight",
                                                   "oracle|trained", "oracle|tight", "flat|tight"])
    ap.add_argument("--out", default="probe4_pf.json")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--s_dv", type=float, default=1e-4)
    ap.add_argument("--skip", type=int, default=0)
    a = ap.parse_args()

    ev = load_split(a.split, cap=(a.cap or None))
    ev = ev[a.skip:]
    d = np.load(f"{ARMS}/act_{a.split}.npz", allow_pickle=True)
    acts = {s["stem"]: np.clip(np.asarray(d[s["stem"] + "|act"], np.float32), 1e-4, 1 - 1e-4) for s in ev}
    print(f"eval songs {len(ev)}  total frames {sum(s['T'] for s in ev)}", flush=True)

    # supervised emission table (fit on TRAIN fold by q3)
    import q3_probe4_fit as Q3
    tr = load_split("train")
    dtr = np.load(f"{ARMS}/act_train.npz", allow_pickle=True)
    atr = {s["stem"]: np.clip(np.asarray(dtr[s["stem"] + "|act"], np.float32), 1e-4, 1 - 1e-4) for s in tr}
    mu, gm, _ = Q3.fit_table(tr, atr, a.nbins, "bar")
    print(f"supervised emission: nbins={a.nbins}, fitted on {len(tr)} TRAIN songs", flush=True)

    def obs_act(s, T):
        return torch.from_numpy(acts[s["stem"]][:T]).unsqueeze(0).to(DEV)

    def obs_oracle(s, T):
        ph = ideal_barphase(s["downs"], T, FPS, mode="extrap")
        if ph is None: return None
        o = np.stack([np.cos(ph), np.sin(ph)], 1).astype(np.float32)
        return torch.from_numpy(o).unsqueeze(0).to(DEV)

    # trained prior model = ARM (ii) checkpoint (h = the activation, so no MERT needed)
    ck = torch.load(f"{ARMS}/arm_i_ii_bern.pt", map_location=DEV)
    trained = VB.BarPointerVAE_B(h_dim=2, hidden=ck["config"]["hidden"], num_meters=4,
                                 obs_dim=2, obs_type="bern").to(DEV)
    trained.load_state_dict(ck["model"]); trained.eval()
    tight = TightPrior(rho=a.rho, s_lv=a.s_lv, s_dv=a.s_dv).to(DEV); tight.eval()

    res = {"config": vars(a), "cells": {}}
    for cell in a.cells:
        emis, prior = cell.split("|")
        model = trained if prior == "trained" else tight
        import copy
        model = copy.deepcopy(model)
        if emis == "sup":
            model.obs_logp = types.MethodType(make_sup_logp(mu, DEV), model)
            ofn = obs_act
        elif emis == "oracle":
            model.obs_logp = types.MethodType(make_oracle_logp(a.kappa), model)
            ofn = obs_oracle
        elif emis == "flat":
            model.obs_logp = types.MethodType(flat_logp, model)
            ofn = obs_act
        else:
            ofn = obs_act
        t0 = time.time()
        rows = run_cell(model, ev, ofn, a.K, a.alpha, cell)
        print(f"  {cell}  ({time.time()-t0:.0f}s, {len(rows)} songs)", flush=True)
        for rd in ("mean", "map", "smooth"):
            s = summarize(rows, rd, cell); pr(s)
            res["cells"][f"{cell}|{rd}"] = s
    json.dump(res, open(f"{OUT}/{a.out}", "w"), indent=1, default=float)
    print("WROTE", f"{OUT}/{a.out}")


if __name__ == "__main__":
    main()
