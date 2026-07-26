"""PROBE 3 -- is the learned latent bar phase aligned with the TRUE bar phase?

Replays the trained posterior recursion (exactly the elbo_b loop) on train/eval crops,
extracts phi trajectories, compares to the label-derived bar phase, and ablates the
individual z_feat channels to find which one actually carries the beat information.
Nothing under vbpm/, vbpm_fix/, vbpm_arms/ is modified.
"""
from __future__ import annotations
import argparse, json, math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms")

import variant_b as VB
from vbpm.distributions import (TWO_PI, gumbel_softmax, sample_wrapped_cauchy,
                                sample_student_t, kl_categorical, kl_wrapped_cauchy,
                                kl_log_normal, kl_student_t_mc)
from audit_common import load_split, ideal_barphase, banner, FPS
from common import targets

DEV = "cuda:0"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"


class LayerMerge(nn.Module):
    def __init__(self, n_layers=13):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, feats):
        return torch.einsum("l,bltf->btf", torch.softmax(self.layer_logits, 0), feats)


def circ_corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ab = math.atan2(np.sin(a).mean(), np.cos(a).mean())
    bb = math.atan2(np.sin(b).mean(), np.cos(b).mean())
    num = (np.sin(a - ab) * np.sin(b - bb)).sum()
    den = math.sqrt((np.sin(a - ab) ** 2).sum() * (np.sin(b - bb) ** 2).sum())
    return float(num / max(den, 1e-12))


def lock_R(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(abs(np.exp(1j * d).mean()))


def best_offset_mae(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    off = math.atan2(np.sin(d).mean(), np.cos(d).mean())
    e = (d - off + math.pi) % TWO_PI - math.pi
    return float(np.abs(e).mean()), float(off)


def raw_mae(a, b):
    e = (np.asarray(a, float) - np.asarray(b, float) + math.pi) % TWO_PI - math.pi
    return float(np.abs(e).mean())


def _stationary_dev_sigma(sigma, a):
    return sigma / torch.sqrt((1.0 - a ** 2).clamp(min=1e-3))


@torch.no_grad()
def posterior_replay(model, h, b, db, temperature=0.3):
    B, T, _ = h.shape
    post_ctx = model.encode_posterior(h, b)
    prior_ctx = model.encode_prior(h)
    dof = model.tempo_dof()
    kl_m = h.new_zeros(B); kl_p = h.new_zeros(B)
    kl_lv = h.new_zeros(B); kl_dv = h.new_zeros(B)
    z_feats, phis, mus, rhos, lts = [], [], [], [], []
    n_cross = h.new_zeros(B)

    z0 = model.z0.unsqueeze(0).expand(B, -1)
    q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
        model.post_head(torch.cat([post_ctx[:, 0], z0], dim=-1)))
    p_m, p_ph_mu, p_ph_rho, p_lv_mu, p_lv_s, _a, _c = model.unpack(
        model.prior_init_head(prior_ctx.mean(1)))
    a0 = model.prior_dev_coef(prior_ctx[:, 0]); sd0 = model.prior_dev_scale(prior_ctx[:, 0])
    p_dv_mu = torch.zeros_like(q_dv_mu); p_dv_s = _stationary_dev_sigma(sd0, a0)

    meter = gumbel_softmax(q_m, temperature)
    phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
    level = sample_student_t(dof, q_lv_mu, q_lv_s)
    dev = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
    log_tempo = level + dev

    kl_m = kl_m + kl_categorical(torch.log_softmax(q_m, -1), torch.log_softmax(p_m, -1))
    kl_p = kl_p + kl_wrapped_cauchy(q_ph_mu, q_ph_rho, p_ph_mu, p_ph_rho)
    kl_lv = kl_lv + kl_student_t_mc(dof, q_lv_mu, q_lv_s, p_lv_mu, p_lv_s, level)
    kl_dv = kl_dv + kl_log_normal(q_dv_mu, q_dv_s, p_dv_mu, p_dv_s)
    n_cross = n_cross + 1.0

    z_feats.append(model.z_features(meter, phi, log_tempo))
    phis.append(phi); mus.append(q_ph_mu); rhos.append(q_ph_rho); lts.append(log_tempo)
    level_anchor = level; a_lv = model.level_ar()
    meter_prev, phi_prev = meter, phi
    level_prev, dev_prev, log_tempo_prev = level, dev, log_tempo

    for t in range(1, T):
        z_prev_feat = model.z_features(meter_prev, phi_prev, log_tempo_prev)
        q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
            model.post_head(torch.cat([post_ctx[:, t], z_prev_feat], dim=-1)))
        tempo_prev = torch.exp(log_tempo_prev.clamp(-12.0, 6.0))
        advance = phi_prev + tempo_prev
        cross = (advance >= TWO_PI).to(h.dtype)
        p_ph_mu = advance % TWO_PI
        p_ph_rho = model.prior_phase_conc(prior_ctx[:, t])
        a = model.prior_dev_coef(prior_ctx[:, t])
        p_lv_mu = level_anchor + a_lv * (level_prev - level_anchor)
        p_lv_s = model.prior_level_scale(prior_ctx[:, t])
        p_dv_mu = a * dev_prev; p_dv_s = model.prior_dev_scale(prior_ctx[:, t])

        phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
        level = sample_student_t(dof, q_lv_mu, q_lv_s)
        dev = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
        log_tempo = level + dev
        q_meter_draw = gumbel_softmax(q_m, temperature)
        meter = torch.where(cross.unsqueeze(-1) > 0.5, q_meter_draw, meter_prev)
        log_pi_p = model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[:, t])

        kl_p = kl_p + kl_wrapped_cauchy(q_ph_mu, q_ph_rho, p_ph_mu, p_ph_rho)
        kl_lv = kl_lv + kl_student_t_mc(dof, q_lv_mu, q_lv_s, p_lv_mu, p_lv_s, level)
        kl_dv = kl_dv + kl_log_normal(q_dv_mu, q_dv_s, p_dv_mu, p_dv_s)
        kl_m = kl_m + cross * kl_categorical(torch.log_softmax(q_m, -1), log_pi_p)
        n_cross = n_cross + cross

        z_feats.append(model.z_features(meter, phi, log_tempo))
        phis.append(phi); mus.append(q_ph_mu); rhos.append(q_ph_rho); lts.append(log_tempo)
        meter_prev, phi_prev = meter, phi
        level_prev, dev_prev, log_tempo_prev = level, dev, log_tempo

    return dict(Z=torch.stack(z_feats, 1), phi=torch.stack(phis, 1),
                q_mu=torch.stack(mus, 1), q_rho=torch.stack(rhos, 1),
                log_tempo=torch.stack(lts, 1),
                kl_phase=kl_p, kl_level=kl_lv, kl_dev=kl_dv, kl_meter=kl_m, n_cross=n_cross)


def make_crops(songs, obs_cache, rng, n, frames, need_downs=3):
    out, tries = [], 0
    while len(out) < n and tries < n * 60:
        tries += 1
        s = songs[rng.integers(len(songs))]
        T = s["feats"].shape[1]
        if T <= frames + 4:
            continue
        st = int(rng.integers(0, T - frames))
        t0, t1 = st / FPS, (st + frames) / FPS
        if len(s["downs"][(s["downs"] >= t0) & (s["downs"] < t1)]) < need_downs:
            continue
        ph_full = ideal_barphase(s["downs"], T, FPS, mode="extrap")
        if ph_full is None:
            continue
        b, d = targets(s["beats"], s["downs"], st, frames)
        out.append(dict(stem=s["stem"], st=st,
                        feats=np.asarray(s["feats"][:, st:st + frames, :], np.float32),
                        b=b, d=d, obs=obs_cache[s["stem"]][st:st + frames],
                        true_phi=ph_full[st:st + frames],
                        n_beats=int(b.sum()), n_downs=int(d.sum())))
    return out


def build_obs_cache(songs, npz):
    dd = np.load(npz, allow_pickle=True)
    return {s["stem"]: np.clip(np.asarray(dd[s["stem"] + "|act"], np.float32),
                               1e-4, 1 - 1e-4) for s in songs}


def run_split(model, merge, arm, crops, tag, rng, results):
    banner(f"[{tag}]  n_crops={len(crops)}")
    rows = []
    for i in range(0, len(crops), 8):
        chunk = crops[i:i + 8]
        f = torch.from_numpy(np.stack([c["feats"] for c in chunk])).to(DEV)
        b = torch.from_numpy(np.stack([c["b"] for c in chunk])).to(DEV)
        d = torch.from_numpy(np.stack([c["d"] for c in chunk])).to(DEV)
        o = torch.from_numpy(np.stack([c["obs"] for c in chunk])).to(DEV)
        h = merge(f) if arm == "i" else o
        r = posterior_replay(model, h, b, d)
        Z = r["Z"]; B, T, _ = Z.shape

        def bce(Zx):
            lg = model.decoder(Zx)
            return (F.binary_cross_entropy_with_logits(lg[..., 0], b, reduction="none").sum(1),
                    F.binary_cross_entropy_with_logits(lg[..., 1], d, reduction="none").sum(1))

        def obsll(Zx):
            return model.obs_logp(Zx.reshape(B * T, -1), o.reshape(B * T, -1)).reshape(B, T).sum(1)

        rb0, rd0 = bce(Z); ob0 = obsll(Z)
        Zp = Z.clone(); rp = torch.rand(B, T, device=DEV) * TWO_PI
        Zp[..., 0] = torch.cos(rp); Zp[..., 1] = torch.sin(rp)
        rb_p, rd_p = bce(Zp); ob_p = obsll(Zp)
        Zl = Z.clone(); Zl[..., 2] = Z[..., 2].mean(1, keepdim=True)
        rb_l, rd_l = bce(Zl); ob_l = obsll(Zl)
        Zm = Z.clone()
        Zm[..., 3:] = F.one_hot(torch.randint(0, 4, (B, T), device=DEV), 4).float()
        rb_m, rd_m = bce(Zm); ob_m = obsll(Zm)
        tp = torch.from_numpy(np.stack([c["true_phi"] for c in chunk]).astype(np.float32)).to(DEV)
        Zt = Z.clone(); Zt[..., 0] = torch.cos(tp); Zt[..., 1] = torch.sin(tp)
        rb_t, rd_t = bce(Zt); ob_t = obsll(Zt)

        for j, c in enumerate(chunk):
            phi = r["phi"][j].cpu().numpy(); mu = r["q_mu"][j].cpu().numpy()
            rho = r["q_rho"][j].cpu().numpy(); lt = r["log_tempo"][j].cpu().numpy()
            tphi = c["true_phi"]; rnd = rng.random(len(tphi)) * TWO_PI
            rows.append(dict(
                stem=c["stem"], st=c["st"], n_beats=c["n_beats"], n_downs=c["n_downs"],
                cc_sample=circ_corr(phi, tphi), cc_mu=circ_corr(mu, tphi),
                cc_rand=circ_corr(rnd, tphi),
                R_sample=lock_R(phi, tphi), R_mu=lock_R(mu, tphi), R_rand=lock_R(rnd, tphi),
                mae_raw=raw_mae(mu, tphi), mae_off=best_offset_mae(mu, tphi)[0],
                mae_rand=best_offset_mae(rnd, tphi)[0],
                rho_mean=float(rho.mean()), rho_med=float(np.median(rho)),
                lt_mean=float(lt.mean()), lt_std=float(lt.std()),
                dphi_mean=float(np.mean((np.diff(mu) + math.pi) % TWO_PI - math.pi)),
                frac_neg=float(np.mean(((np.diff(mu) + math.pi) % TWO_PI - math.pi) < 0)),
                true_dphi=float(np.mean((np.diff(tphi) + math.pi) % TWO_PI - math.pi)),
                kl_phase=float(r["kl_phase"][j]), kl_level=float(r["kl_level"][j]),
                kl_dev=float(r["kl_dev"][j]), kl_meter=float(r["kl_meter"][j]),
                n_cross=float(r["n_cross"][j]),
                rec_b=float(rb0[j]), rec_d=float(rd0[j]), obs_ll=float(ob0[j]),
                rec_b_randphase=float(rb_p[j]), rec_d_randphase=float(rd_p[j]),
                obs_randphase=float(ob_p[j]),
                rec_b_flatlt=float(rb_l[j]), rec_d_flatlt=float(rd_l[j]),
                obs_flatlt=float(ob_l[j]),
                rec_b_randmeter=float(rb_m[j]), rec_d_randmeter=float(rd_m[j]),
                obs_randmeter=float(ob_m[j]),
                rec_b_truephase=float(rb_t[j]), rec_d_truephase=float(rd_t[j]),
                obs_truephase=float(ob_t[j]),
                lt_beat_mean=float(lt[c["b"] > 0.5].mean()) if c["n_beats"] else float("nan"),
                lt_nonbeat_mean=float(lt[c["b"] < 0.5].mean()),
                lt_beat_d=float((lt[c["b"] > 0.5].mean() - lt[c["b"] < 0.5].mean())
                                / max(lt.std(), 1e-9)) if c["n_beats"] else float("nan")))
        del f, b, d, o, h, r, Z

    A = lambda k: float(np.nanmean([x[k] for x in rows]))
    S = lambda k: float(np.nanstd([x[k] for x in rows]))

    def base(nk):
        v = []
        for x in rows:
            p = max(x[nk], 1) / 256.0
            v.append(-(x[nk] * math.log(p) + (256 - x[nk]) * math.log(1 - p)))
        return float(np.mean(v))

    base_b, base_d = base("n_beats"), base("n_downs")
    print(f"  --- PHASE ALIGNMENT (n={len(rows)} crops of 256 frames) ---")
    print(f"  circ_corr(q_mu , true) = {A('cc_mu'):+.4f} +- {S('cc_mu'):.4f}")
    print(f"  circ_corr(sample,true) = {A('cc_sample'):+.4f} +- {S('cc_sample'):.4f}")
    print(f"  circ_corr(RANDOM,true) = {A('cc_rand'):+.4f} +- {S('cc_rand'):.4f}   <-- chance")
    print(f"  lock_R  (q_mu,true)    = {A('R_mu'):.4f}   sample {A('R_sample'):.4f}   "
          f"RANDOM {A('R_rand'):.4f}")
    print(f"  MAE raw   (q_mu,true)  = {A('mae_raw'):.4f} rad  (uniform chance pi/2 = 1.5708)")
    print(f"  MAE best-offset        = {A('mae_off'):.4f} rad   RANDOM {A('mae_rand'):.4f}")
    print("  --- POSTERIOR CONCENTRATION / KL ---")
    print(f"  rho (wrapped-Cauchy resultant length) mean={A('rho_mean'):.4f} "
          f"median={A('rho_med'):.4f}")
    print(f"  kl_phase/crop={A('kl_phase'):.3f}  kl_level={A('kl_level'):.2f}  "
          f"kl_dev={A('kl_dev'):.2f}  kl_meter={A('kl_meter'):.2f}  n_cross={A('n_cross'):.1f}")
    print(f"  posterior d(mu_phi)/frame={A('dphi_mean'):+.5f} (TRUE {A('true_dphi'):+.5f}) "
          f"frac_neg={A('frac_neg'):.3f}")
    print("  --- WHERE DOES THE BEAT RECONSTRUCTION COME FROM? (nats / 256-frame crop) ---")
    print(f"  base-rate BCE            beat={base_b:.2f}  downbeat={base_d:.2f}")
    print(f"  model (posterior z)      beat={A('rec_b'):.2f}  downbeat={A('rec_d'):.2f}  "
          f"obs_ll={A('obs_ll'):.2f}")
    for lab, k in (("randomize phase   ", "randphase"), ("flatten log_tempo ", "flatlt"),
                   ("randomize meter   ", "randmeter"), ("phase:=TRUE bar ph", "truephase")):
        print(f"  {lab}       beat={A('rec_b_' + k):.2f} ({A('rec_b_' + k) - A('rec_b'):+.2f})"
              f"  downbeat={A('rec_d_' + k):.2f} ({A('rec_d_' + k) - A('rec_d'):+.2f})"
              f"  obs_ll={A('obs_' + k):.2f} ({A('obs_' + k) - A('obs_ll'):+.2f})")
    print("  --- MORSE-CODE TEST: log_tempo on beat vs non-beat frames ---")
    print(f"  mean log_tempo beat={A('lt_beat_mean'):+.4f} nonbeat={A('lt_nonbeat_mean'):+.4f}"
          f"  standardized gap={A('lt_beat_d'):+.4f} sd   (crop lt_std={A('lt_std'):.4f})")
    results[tag] = dict(agg={k: A(k) for k in rows[0] if isinstance(rows[0][k], float)},
                        base_b=base_b, base_d=base_d, n=len(rows), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="i", choices=["i", "ii"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--frames", type=int, default=256)
    a = ap.parse_args()

    ck = f"{ARMS}/arm_i_{'i_bern' if a.arm == 'i' else 'ii_bern'}.pt"
    sd = torch.load(ck, map_location="cpu")
    h_dim = 768 if a.arm == "i" else 2
    merge = LayerMerge().to(DEV); merge.load_state_dict(sd["merge"]); merge.eval()
    model = VB.BarPointerVAE_B(h_dim=h_dim, hidden=128, num_meters=4,
                               obs_dim=2, obs_type="bern").to(DEV)
    model.load_state_dict(sd["model"]); model.eval()
    banner(f"PROBE 3  arm={a.arm}  ckpt={ck}")

    train = load_split("train", with_feats=True)
    ev = load_split("eval", with_feats=True)
    print(f"  train {len(train)} eval {len(ev)}", flush=True)
    otr = build_obs_cache(train, f"{ARMS}/act_train.npz")
    oev = build_obs_cache(ev, f"{ARMS}/act_eval.npz")

    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    ctr = make_crops(train, otr, rng, a.n, a.frames)
    cev = make_crops(ev, oev, rng, a.n, a.frames)
    results = {}
    run_split(model, merge, a.arm, ctr, "TRAIN-fold", rng, results)
    run_split(model, merge, a.arm, cev, "EVAL-fold", rng, results)
    json.dump(results, open("/home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf/"
                            f"p3align_agentP3_arm{a.arm}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
