"""EXPERIMENT 2 driver: supervised emission -> contrast -> particle filter -> controlled eval."""
from __future__ import annotations

import argparse, json, math, sys, time
import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")
from emission import (PhaseEmission, load_act, load_split, obs_contrast, song_phase,  # noqa
                      METERS, TWO_PI, FPS, _estimate_meter)
from pf import particle_filter
from vbpm.evaluate import (beats_from_barphase, downbeats_from_barphase,   # noqa
                           beats_from_activation, metronome, f_measure)
from common import smooth_phase                                            # noqa


# ---- MANDATORY density-matched blind control (verbatim from vbpm_arms/arm_ii.py) ----
def blind_grid_controls(ref, T, n_est, n_off=12):
    dur = T / FPS
    if n_est < 2 or len(ref) < 2:
        return float("nan"), float("nan")
    per = dur / n_est
    base = np.arange(n_est) * per
    f0 = f_measure(ref, base)
    best = max(f_measure(ref, base + k * per / n_off) for k in range(n_off))
    return float(f0), float(max(best, f0))


def phase_diag(ph):
    d = (np.diff(np.asarray(ph, float)) + math.pi) % TWO_PI - math.pi
    if len(d) == 0:
        return dict(frac_neg=float("nan"), mean_adv=float("nan"), jitter=float("nan"),
                    jitter_over_adv=float("nan"))
    adv = float(d.mean())
    return dict(frac_neg=float(np.mean(d < 0)), mean_adv=adv, jitter=float(d.std()),
                jitter_over_adv=float(d.std() / max(abs(adv), 1e-9)))


def score_traj(ph, m, ref, dref, T):
    est = beats_from_barphase(ph, m, FPS)
    dest = downbeats_from_barphase(ph, FPS)
    b0, bb = blind_grid_controls(ref, T, len(est))
    d0, db = blind_grid_controls(dref, T, len(dest))
    pd = phase_diag(ph)
    return dict(beat_F=f_measure(ref, est),
                db_F=f_measure(dref, dest) if len(dref) >= 2 else float("nan"),
                n_est=len(est), n_est_db=len(dest), blind0=b0, blind_best=bb,
                blind_db0=d0, blind_db_best=db, **pd)


def score_events(est, dest, ref, dref, T):
    b0, bb = blind_grid_controls(ref, T, len(est))
    d0, db = blind_grid_controls(dref, T, len(dest))
    return dict(beat_F=f_measure(ref, est),
                db_F=f_measure(dref, dest) if len(dref) >= 2 else float("nan"),
                n_est=len(est), n_est_db=len(dest), blind0=b0, blind_best=bb,
                blind_db0=d0, blind_db_best=db, frac_neg=float("nan"),
                mean_adv=float("nan"), jitter=float("nan"), jitter_over_adv=float("nan"))


def summarize(rows, name):
    def M(k):
        v = [r[k] for r in rows if isinstance(r.get(k), float) and not math.isnan(r[k])]
        return float(np.mean(v)) if v else float("nan")
    ne = sum(r["n_est"] for r in rows); nt = sum(r["n_true"] for r in rows)
    ned = sum(r["n_est_db"] for r in rows); ntd = sum(r["n_true_db"] for r in rows)
    bf, bb = M("beat_F"), M("blind_best")
    dfm, dbb = M("db_F"), M("blind_db_best")
    return dict(name=name, beat_F=bf, downbeat_F=dfm,
                n_ratio=ne / max(nt, 1), n_ratio_db=ned / max(ntd, 1),
                blind_same_density=M("blind0"), blind_best_offset=bb,
                margin_over_blind=bf - bb, blind_db_best=dbb,
                margin_db_over_blind=dfm - dbb, frac_neg=M("frac_neg"),
                jitter_over_adv=M("jitter_over_adv"), obs_contrast=M("obs_contrast"),
                ess=M("ess"), meter_acc=M("meter_ok"), n_songs=len(rows))


def pr(d):
    print(f"  [{d['name']:34s}] beat_F={d['beat_F']:.4f} db_F={d['downbeat_F']:.4f} "
          f"n_ratio={d['n_ratio']:.3f} blind0={d['blind_same_density']:.4f} "
          f"blindbest={d['blind_best_offset']:.4f} MARGIN={d['margin_over_blind']:+.4f} | "
          f"db_blind={d['blind_db_best']:.4f} MARGIN_db={d['margin_db_over_blind']:+.4f} "
          f"n_ratio_db={d['n_ratio_db']:.2f} | frac_neg={d['frac_neg']:.3f} "
          f"jit/adv={d['jitter_over_adv']:.2f} ESS={d['ess']:.0f} "
          f"contrast={d['obs_contrast']:.3g} meter_acc={d['meter_acc']:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lik", default="gauss", choices=["bern", "gauss"])
    ap.add_argument("--bpb", type=int, default=24)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--phase_mode", default="downbeat", choices=["downbeat", "beat"])
    ap.add_argument("--K", type=int, default=600)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.005])
    ap.add_argument("--sphis", type=float, nargs="+", default=[0.0])
    ap.add_argument("--shuffle_phase", action="store_true")
    ap.add_argument("--tempo_prior", default="none", choices=["none", "init", "ou"])
    ap.add_argument("--tp_rho", type=float, default=0.999)
    ap.add_argument("--p_switch", type=float, default=0.005)
    ap.add_argument("--noise", default="gauss", choices=["gauss", "laplace"])
    ap.add_argument("--n_eval", type=int, default=0)
    ap.add_argument("--eval_split", default="eval", choices=["eval", "train"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--tag", default="exp2")
    a = ap.parse_args()

    tr = load_split("train"); at = load_act("train")
    if a.eval_split == "train":     # fold-honest HYPERPARAMETER SELECTION: score on train
        ev, ae = tr, at
    else:
        ev, ae = load_split("eval"), load_act("eval")
    if a.n_eval:
        ev = ev[:a.n_eval]
    print(f"train {len(tr)}  eval {len(ev)}", flush=True)

    t0 = time.time()
    emis = PhaseEmission(bins_per_beat=a.bpb, likelihood=a.lik,
                         smooth=a.smooth).fit(tr, at, phase_mode=a.phase_mode, shuffle_phase=a.shuffle_phase)
    c_ev, per_c = obs_contrast(emis, ev, ae, phase_mode=a.phase_mode)
    c_tr, _ = obs_contrast(emis, tr, at, phase_mode=a.phase_mode)
    print(f"EMISSION lik={a.lik} bpb={a.bpb} smooth={a.smooth} phase={a.phase_mode} "
          f"songs/meter={emis.n_used}  obs_contrast eval={c_ev:.4f} train={c_tr:.4f} "
          f"({time.time()-t0:.1f}s)", flush=True)

    prior = np.zeros(5)
    lts = {m: [] for m in METERS}
    for s in tr:
        m = _estimate_meter(s["beats"], s["downs"])
        if m in METERS:
            prior[m] += 1
            if len(s["beats"]) > 3:
                ibi = float(np.median(np.diff(s["beats"])))
                lts[m].append(math.log(TWO_PI / max(m * ibi * FPS, 1e-6)))
    tp = None
    if a.tempo_prior != "none":     # train-fitted log bar-advance prior, per meter
        tp = {m: (float(np.mean(lts[m])), float(np.std(lts[m]) + 1e-3)) for m in METERS}
        print("tempo prior (log bar-advance) per meter:",
              {m: (round(v[0], 3), round(v[1], 3)) for m, v in tp.items()}, flush=True)

    res = {"config": vars(a), "emission": {"contrast_eval": c_ev, "contrast_train": c_tr,
                                           "n_used": {str(k): v for k, v in emis.n_used.items()}}}
    base_rows = {}
    for cfg in [(al, sg, sp) for al in a.alphas for sg in a.sigmas for sp in a.sphis]:
        alpha, sigma, sphi = cfg
        rows = {k: [] for k in ("mean", "map", "path", "smooth_mean", "pf_meter_path")}
        t1 = time.time()
        for i, s in enumerate(ev):
            act = ae.get(s["stem"])
            if act is None: continue
            T = min(len(act), s["T"])
            ref = s["beats"][s["beats"] < T / FPS]
            dref = s["downs"][s["downs"] < T / FPS]
            if len(ref) < 3: continue
            m_gt = _estimate_meter(s["beats"], s["downs"])
            LL = emis.padded_table(act[:T])
            out = particle_filter(LL, emis.nb, K=a.K, alpha=alpha, sigma_lt=sigma,
                                  sigma_phi=sphi, p_switch=a.p_switch, meter_prior=prior, fps=FPS,
                                  tempo_prior=tp, tp_mode=a.tempo_prior, tp_rho=a.tp_rho,
                                  seed=a.seed + i, noise=a.noise)
            base = dict(stem=s["stem"], dataset=s["dataset"], T=T, n_true=len(ref),
                        n_true_db=len(dref), ess=out["ess"], obs_contrast=per_c[i] if i < len(per_c) else float("nan"),
                        meter_ok=float(int(np.bincount(out["meter_path"]).argmax()) == m_gt))
            trajs = {"mean": out["phase_mean"], "map": out["phase_map"],
                     "path": out["phase_path"],
                     "smooth_mean": smooth_phase(out["phase_mean"], 5)}
            for k, ph in trajs.items():
                rows[k].append({**base, **score_traj(ph, m_gt, ref, dref, T)})
            # fully-blind variant: read out with the PF's OWN inferred meter
            m_pf = int(np.bincount(out["meter_path"]).argmax())
            rows["pf_meter_path"].append({**base, **score_traj(out["phase_path"], m_pf,
                                                               ref, dref, T)})
            if a.baselines and not base_rows:
                pass
        print(f"PF alpha={alpha} sigma={sigma} sphi={sphi} K={a.K} noise={a.noise} "
              f"({time.time()-t1:.0f}s)", flush=True)
        for k in rows:
            if rows[k]:
                d = summarize(rows[k], f"PF a={alpha} s={sigma} sp={sphi} {k}")
                pr(d)
                res.setdefault("pf", {})[f"a{alpha}_s{sigma}_sp{sphi}_{k}"] = d
        res.setdefault("rows", {})[f"a{alpha}_s{sigma}_sp{sphi}"] = rows["path"]

    if a.baselines:
        print("\nBASELINES (same eval fold, same controls)", flush=True)
        pk, mt, orc = [], [], []
        for s in ev:
            act = ae.get(s["stem"]);  T = min(len(act), s["T"])
            ref = s["beats"][s["beats"] < T / FPS]; dref = s["downs"][s["downs"] < T / FPS]
            if len(ref) < 3: continue
            m_gt = _estimate_meter(s["beats"], s["downs"])
            base = dict(stem=s["stem"], n_true=len(ref), n_true_db=len(dref),
                        ess=float("nan"), obs_contrast=float("nan"), meter_ok=float("nan"))
            e_b = beats_from_activation(act[:T, 0], FPS)
            e_d = beats_from_activation(act[:T, 1], FPS, min_dist_sec=0.30)
            pk.append({**base, **score_events(e_b, e_d, ref, dref, T)})
            mt.append({**base, **score_events(metronome(T, FPS), np.array([]), ref, dref, T)})
            ph = song_phase(s, a.phase_mode)
            if ph is not None:
                orc.append({**base, **score_traj(ph[:T], m_gt, ref, dref, T)})
        for nm, rr in (("act-head peak-pick", pk), ("metronome-120", mt),
                       ("oracle true bar phase", orc)):
            d = summarize(rr, nm); pr(d); res.setdefault("baselines", {})[nm] = d

    json.dump(res, open(f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_final/{a.tag}.json", "w"),
              indent=1, default=float)
    print("WROTE", a.tag + ".json", flush=True)


if __name__ == "__main__":
    main()
