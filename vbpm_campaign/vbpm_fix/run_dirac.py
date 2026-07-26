"""DIRAC ceiling experiment for VARIANT B.

Trains (a) the UNFIXED VBPM baseline and (b) Variant B (observation decoder p(h|z)) on the
same Dirac input, then compares deploy paths:
    open-loop free_run   vs   bootstrap particle filter
plus the mandatory SHIFT TEST (+25 frames on the Dirac impulses, same seed).

DIRAC IS AN ORACLE INPUT (h literally contains the answer). A good score here proves the
MECHANISM is repaired; it does NOT prove beat tracking. MERT is the honest test (run_mert.py).
"""
import sys, json, time, math, argparse
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
import numpy as np, torch

from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run
from vbpm_fix.variant_b import BarPointerVAE_B, dirac_obs, elbo_b, particle_filter
from vbpm_fix.common import (load_split, dirac_h, targets, score_phase, shift_stats,
                             agg, ratio, FPS, H_DIM_DIRAC)

DEV = "cuda:0"
SMOOTH = 5


def make_batch(train, rng, bs, frames):
    hs, bb, dd = [], [], []
    for _ in range(bs):
        s = train[rng.integers(len(train))]
        if s["T"] <= frames: continue
        st = int(rng.integers(0, s["T"] - frames))
        hs.append(torch.from_numpy(dirac_h(s["beats"], s["downs"], st, frames, rng=rng)))
        b, d = targets(s["beats"], s["downs"], st, frames)
        bb.append(torch.from_numpy(b)); dd.append(torch.from_numpy(d))
    return (torch.stack(hs).to(DEV), torch.stack(bb).to(DEV), torch.stack(dd).to(DEV))


def _h(s, T, seed, shift=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(dirac_h(s["beats"], s["downs"], 0, T, shift_frames=shift,
                                    rng=rng)).unsqueeze(0).to(DEV)


@torch.no_grad()
def eval_openloop(model, songs, max_frames, seed=1234):
    rows = []
    for i, s in enumerate(songs):
        T = min(s["T"], max_frames)
        torch.manual_seed(seed + i)
        out = free_run(model, _h(s, T, seed + i))
        rows.append(score_phase(out["phase_mu"][0, :T].cpu().numpy(), s, T))
    return rows


@torch.no_grad()
def eval_filter(model, songs, max_frames, K, alpha=1.0, diffuse=True, seed=1234):
    rm, rp, rs, ess = [], [], [], []
    for i, s in enumerate(songs):
        T = min(s["T"], max_frames)
        torch.manual_seed(seed + i)
        h = _h(s, T, seed + i)
        out = particle_filter(model, h, dirac_obs(h), K=K, alpha=alpha, diffuse_init=diffuse)
        rm.append(score_phase(out["phase_mean"].numpy(), s, T))
        rp.append(score_phase(out["phase_map"].numpy(), s, T))
        rs.append(score_phase(out["phase_mean"].numpy(), s, T, smooth=SMOOTH))
        ess.append(out["ess"])
    return rm, rp, rs, float(np.mean(ess))


def summarize(tag, rows):
    return (f"{tag}: beat_F={agg(rows,'beat_F'):.3f} db_F={agg(rows,'db_F'):.3f} "
            f"n_est/n_true={ratio(rows):.3f} metronome={agg(rows,'metronome'):.3f}")


def rec(rows):
    return {"beat_F": agg(rows, "beat_F"), "db_F": agg(rows, "db_F"), "n_ratio": ratio(rows),
            "metronome": agg(rows, "metronome")}


def shift_test(model, songs, max_frames, mode, K=400, shift=25, seed=99):
    """SAME seed, Dirac impulses delayed by +25 frames. A tracking deploy path must
    (i) move a lot and (ii) move by the RIGHT amount: best-align lag ~= +25 frames."""
    out = []
    for s in songs:
        T = min(s["T"], max_frames)
        trajs = []
        for sh in (0, shift):
            torch.manual_seed(seed)                       # SAME seed for both runs
            h = _h(s, T, seed, shift=sh)                  # same Dirac noise too
            if mode == "free":
                trajs.append(free_run(model, h)["phase_mu"][0, :T].cpu().numpy())
            else:
                trajs.append(particle_filter(model, h, dirac_obs(h), K=K)["phase_mean"].numpy())
        out.append(shift_stats(trajs[0], trajs[1]))
    return {k: (float(np.mean([o[k] for o in out])) if k != "lag"
                else [o["lag"] for o in out]) for k in out[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--K", type=int, default=400)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--max_frames", type=int, default=1600)
    ap.add_argument("--n_shift", type=int, default=8)
    ap.add_argument("--only", choices=["base","varB","both"], default="both")
    ap.add_argument("--out", default="/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/results_dirac.json")
    a = ap.parse_args()

    train = load_split("train"); ev = load_split("eval", a.n_eval)
    print(f"DIRAC: train {len(train)} eval {len(ev)}", flush=True)
    res = {"config": vars(a)}

    # ---------------- (a) UNFIXED baseline ----------------
    if a.only in ("base", "both"):
     torch.manual_seed(0); rng = np.random.default_rng(0)
     base = BarPointerVAE(h_dim=H_DIM_DIRAC, hidden=a.hidden, num_meters=4).to(DEV)
     opt = torch.optim.AdamW(base.parameters(), lr=a.lr); t0 = time.time()
     for step in range(1, a.steps + 1):
        beta = min(1.0, step / a.warmup); temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        h, b, d = make_batch(train, rng, a.bs, a.frames)
        opt.zero_grad(); loss, info = strict_elbo(base, h, b, d, temperature=temp, beta=beta)
        if not torch.isfinite(loss): print("NaN@", step, flush=True); break
        loss.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(), 5.0); opt.step()
        if step % 200 == 0:
            print(f"  [base] s{step} rec_b={info['recon_beat']:.1f} kl_phi={info['kl_phase']:.2f} "
                  f"{step/(time.time()-t0):.2f} it/s", flush=True)
     base.eval()
     r = eval_openloop(base, ev, a.max_frames)
     print("BASELINE " + summarize("free_run(open-loop)", r), flush=True)
     res["baseline_freerun"] = rec(r)
     st = shift_test(base, ev[:a.n_shift], a.max_frames, "free")
     print(f"BASELINE SHIFT (free_run,+25f): max|dphi|={st['max']:.4f} mean={st['mean']:.4f} lags={st['lag']}", flush=True)
     res["baseline_shift"] = st

    # ---------------- (b) VARIANT B ----------------
    if a.only == "base":
        json.dump(res, open(a.out, "w"), indent=2); print("WROTE " + a.out, flush=True); return
    torch.manual_seed(0); rng = np.random.default_rng(0)
    model = BarPointerVAE_B(h_dim=H_DIM_DIRAC, hidden=a.hidden, num_meters=4,
                            obs_dim=2, obs_type="bern").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr); t0 = time.time()
    for step in range(1, a.steps + 1):
        beta = min(1.0, step / a.warmup); temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        h, b, d = make_batch(train, rng, a.bs, a.frames)
        opt.zero_grad(); loss, info = elbo_b(model, h, b, d, dirac_obs(h), temperature=temp, beta=beta)
        if not torch.isfinite(loss): print("NaN@", step, flush=True); break
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 200 == 0:
            print(f"  [varB] s{step} rec_b={info['recon_beat']:.1f} rec_obs={info['recon_obs']:.1f} "
                  f"kl_phi={info['kl_phase']:.2f} {step/(time.time()-t0):.2f} it/s", flush=True)
    model.eval()
    torch.save(model.state_dict(), "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/varB_dirac.pt")

    r_free = eval_openloop(model, ev, a.max_frames)
    print("VARIANT-B " + summarize("free_run(open-loop, SAME trained model)", r_free), flush=True)
    res["varB_freerun"] = rec(r_free); res["metronome"] = agg(r_free, "metronome")

    rm, rp, rs, ess = eval_filter(model, ev, a.max_frames, a.K)
    print("VARIANT-B " + summarize(f"PF(K={a.K}) circ-wmean      ", rm) + f" ESS={ess:.1f}", flush=True)
    print("VARIANT-B " + summarize(f"PF(K={a.K}) MAP-particle    ", rp), flush=True)
    print("VARIANT-B " + summarize(f"PF(K={a.K}) wmean+smooth{SMOOTH}  ", rs), flush=True)
    res["varB_pf_mean"] = dict(rec(rm), ess=ess); res["varB_pf_map"] = rec(rp)
    res["varB_pf_mean_smooth"] = rec(rs)

    rm2, _, rs2, _ = eval_filter(model, ev, a.max_frames, a.K, diffuse=False)
    print("VARIANT-B " + summarize(f"PF(K={a.K}) model-init wmean", rm2), flush=True)
    res["varB_pf_modelinit"] = rec(rm2); res["varB_pf_modelinit_smooth"] = rec(rs2)

    stf = shift_test(model, ev[:a.n_shift], a.max_frames, "free")
    stp = shift_test(model, ev[:a.n_shift], a.max_frames, "pf", K=a.K)
    print(f"VARIANT-B SHIFT free_run: max|dphi|={stf['max']:.4f} mean={stf['mean']:.4f} lags={stf['lag']}", flush=True)
    print(f"VARIANT-B SHIFT PF      : max|dphi|={stp['max']:.4f} mean={stp['mean']:.4f} lags={stp['lag']} "
          f"(expect ~+25) cost min/max={stp['lag_cost_min']:.3f}/{stp['lag_cost_max']:.3f}", flush=True)
    res["varB_shift_freerun"] = stf; res["varB_shift_pf"] = stp

    json.dump(res, open(a.out, "w"), indent=2)
    print("WROTE " + a.out, flush=True)


if __name__ == "__main__":
    main()
