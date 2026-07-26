"""VARIANT B on the DIRAC (oracle) input.

DIRAC IS A CEILING, NOT BEAT TRACKING: h[t,0]=1 exactly at the true beat frames.  A good
score here proves the deploy MECHANISM is repaired; it proves nothing about perception.

Runs, all with the same budget and the same eval protocol:
  BASE   : unmodified vbpm.BarPointerVAE + vbpm.strict_elbo, deploy = vbpm.free_run
  VARB   : BarPointerVAE_B + elbo_b, deploy = (a) vbpm.free_run [open loop, same weights]
           and (b) the bootstrap particle filter.
Plus the MANDATORY controls: 120-BPM metronome, a DENSITY-MATCHED phase-blind grid, and
the +25-frame shift test on both deploy paths.
"""
import argparse, json, math, sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run
from vbpm.evaluate import metronome, f_measure, _estimate_meter
from vbpm_fix.varB_pf.vb import BarPointerVAE_B, elbo_b, particle_filter
from vbpm_fix.varB_pf import common as C

FPS = C.FPS
DEV = "cuda:0"


def batch(train, rng, bs, fr):
    hs, bb, dd = [], [], []
    for _ in range(bs):
        s = train[rng.integers(len(train))]
        if s["T"] <= fr:
            continue
        st = int(rng.integers(0, s["T"] - fr))
        hs.append(torch.from_numpy(C.dirac_h(s["beats"], s["downs"], st, fr, rng)))
        b, d = C.targets(s["beats"], s["downs"], st, fr)
        bb.append(torch.from_numpy(b)); dd.append(torch.from_numpy(d))
    return (torch.stack(hs).to(DEV), torch.stack(bb).to(DEV), torch.stack(dd).to(DEV))


def train_model(kind, train, steps, warm, bs, fr, lr, hidden, lam_h, n_harm, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    if kind == "base":
        model = BarPointerVAE(h_dim=C.H_DIM_DIRAC, hidden=hidden, num_meters=4).to(DEV)
    else:
        model = BarPointerVAE_B(h_dim=C.H_DIM_DIRAC, hidden=hidden, num_meters=4,
                                obs_mode="bern", obs_dim=2, n_harm=n_harm).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        beta = min(1.0, step / max(warm, 1))
        temp = 1.0 + (0.3 - 1.0) * min(step / steps, 1.0)
        h, b, d = batch(train, rng, bs, fr)
        opt.zero_grad()
        if kind == "base":
            loss, info = strict_elbo(model, h, b, d, temperature=temp, beta=beta)
        else:
            loss, info = elbo_b(model, h, b, d, C.dirac_obs(h), temperature=temp,
                                beta=beta, lam_h=lam_h)
        if not torch.isfinite(loss):
            print(f"  !! NaN at {step}", flush=True); break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 100 == 0:
            extra = f" obs_lp={info.get('obs_lp', float('nan')):8.1f}" if kind != "base" else ""
            print(f"  [{kind}] s{step:4d} b={beta:.2f} rec_b={info['recon_beat']:7.2f}"
                  f"{extra} klphi={info['kl_phase']:.1f} kllv={info['kl_level']:.1f}"
                  f" | {step/(time.time()-t0):.1f} it/s", flush=True)
    return model


@torch.no_grad()
def evaluate(model, songs, cap, K, alpha, kind, do_pf, seed=0):
    model.eval()
    res = {k: [] for k in ["fr", "pf_circ", "pf_map", "pf_anc",
                           "pf_circ_mono", "pf_anc_mono"]}
    met, bpm_err = [], []
    ess = []
    for i, s in enumerate(songs):
        T = min(s["T"], cap)
        rng = np.random.default_rng(1000 + i)
        h = torch.from_numpy(C.dirac_h(s["beats"], s["downs"], 0, T, rng)).unsqueeze(0).to(DEV)
        ref = s["beats"][s["beats"] < T / FPS]; dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 2:
            continue
        m = _estimate_meter(ref, dref)
        torch.manual_seed(seed)
        out = free_run(model, h)
        pm = out["phase_mu"][0, :T].cpu().numpy()
        res["fr"].append(C.score_phase(pm, ref, dref, m, T, tag_seed=i))
        met.append(f_measure(ref, metronome(T, FPS)))
        if do_pf:
            torch.manual_seed(seed)
            pf = particle_filter(model, h, C.dirac_obs(h), K=K, alpha=alpha)
            for nm, key in [("pf_circ", "circ"), ("pf_map", "map"), ("pf_anc", "anc")]:
                res[nm].append(C.score_phase(pf[key], ref, dref, m, T, tag_seed=i))
            for nm, key in [("pf_circ_mono", "circ"), ("pf_anc_mono", "anc")]:
                res[nm].append(C.score_phase(C.monotonise(pf[key]), ref, dref, m, T, tag_seed=i))
            ess.append(pf["mean_ess"])
            true_bpm = 60.0 / np.median(np.diff(ref)) if len(ref) > 2 else np.nan
            pf_bpm = 60.0 * FPS * m * math.exp(float(np.median(pf["anc_log_tempo"]))) / (2 * math.pi)
            bpm_err.append(pf_bpm / true_bpm)
    model.train()
    o = {k: C.summarize(v) for k, v in res.items() if v}
    o["metronome"] = float(np.mean(met))
    if ess:
        o["mean_ess"] = float(np.mean(ess)); o["pf_bpm_ratio"] = float(np.median(bpm_err))
    return o


@torch.no_grad()
def shift_test(model, songs, cap, K, alpha, shift, seed=0):
    """MANDATORY MECHANISM CHECK: rebuild the Dirac input with impulses moved +shift
    frames (identical noise), run the SAME deploy path with the SAME seed, and measure
    the max circular difference of the phase trajectory.  Audio-blind => ~0."""
    model.eval()
    out = {"fr": [], "pf": [], "null_fr": [], "null_pf": []}
    for i, s in enumerate(songs):
        T = min(s["T"], cap)
        h0 = torch.from_numpy(C.dirac_h(s["beats"], s["downs"], 0, T,
                                        np.random.default_rng(1000 + i))).unsqueeze(0).to(DEV)
        h1 = torch.from_numpy(C.dirac_h(s["beats"], s["downs"], 0, T,
                                        np.random.default_rng(1000 + i), shift=shift)).unsqueeze(0).to(DEV)
        h0b = torch.from_numpy(C.dirac_h(s["beats"], s["downs"], 0, T,
                                         np.random.default_rng(1000 + i))).unsqueeze(0).to(DEV)
        for tag, ha, hb in [("fr", h0, h1), ("null_fr", h0, h0b)]:
            torch.manual_seed(seed); a = free_run(model, ha)["phase_mu"][0, :T].cpu().numpy()
            torch.manual_seed(seed); b = free_run(model, hb)["phase_mu"][0, :T].cpu().numpy()
            out[tag].append(C.circ_maxdiff(a, b)[0])
        for tag, ha, hb in [("pf", h0, h1), ("null_pf", h0, h0b)]:
            torch.manual_seed(seed)
            a = particle_filter(model, ha, C.dirac_obs(ha), K=K, alpha=alpha)["anc"]
            torch.manual_seed(seed)
            b = particle_filter(model, hb, C.dirac_obs(hb), K=K, alpha=alpha)["anc"]
            out[tag].append(C.circ_maxdiff(a, b)[0])
    model.train()
    return {k: {"max": float(np.max(v)), "mean": float(np.mean(v))} for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_harm", type=int, default=64)
    ap.add_argument("--lam_h", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--cap", type=int, default=1600)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--n_shift", type=int, default=8)
    ap.add_argument("--shift", type=int, default=25)
    ap.add_argument("--skip_base", type=int, default=0)
    ap.add_argument("--out", default="/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/varB_pf/res_dirac.json")
    a = ap.parse_args()

    train = C.load("train"); ev = C.load("eval", a.n_eval)
    print(f"DIRAC (ORACLE CEILING).  train {len(train)}  eval {len(ev)}  "
          f"K={a.K} alpha={a.alpha} n_harm={a.n_harm}", flush=True)
    R = {"config": vars(a)}

    if not a.skip_base:
        print("\n== BASELINE: unmodified VBPM ==", flush=True)
        mb = train_model("base", train, a.steps, a.warmup, a.bs, a.frames, a.lr,
                         a.hidden, 0.0, a.n_harm)
        R["base_freerun"] = evaluate(mb, ev, a.cap, a.K, a.alpha, "base", do_pf=False)
        print("  BASE free_run:", json.dumps(R["base_freerun"]), flush=True)

    print("\n== VARIANT B ==", flush=True)
    mv = train_model("varB", train, a.steps, a.warmup, a.bs, a.frames, a.lr,
                     a.hidden, a.lam_h, a.n_harm)
    torch.save(mv.state_dict(), a.out.replace(".json", "_model.pt"))

    # emission sharpness diagnostic: p(beat impulse | phi) profile from h_dec
    with torch.no_grad():
        ph = torch.linspace(0, 2 * math.pi, 721, device=DEV)
        mo = torch.zeros(721, 4, device=DEV); mo[:, 3] = 1.0
        zf = mv.z_features(mo, ph, torch.full_like(ph, -2.66))
        pr = torch.sigmoid(mv.h_dec(mv.harm(zf)))[:, 0].cpu().numpy()
    R["obs_profile"] = {"max": float(pr.max()), "min": float(pr.min()),
                        "frac_above_half_max": float((pr > 0.5 * pr.max()).mean()),
                        "n_local_peaks": int(((pr[1:-1] > pr[:-2]) & (pr[1:-1] >= pr[2:])
                                              & (pr[1:-1] > 0.5 * pr.max())).sum())}
    print("  obs sharpness:", json.dumps(R["obs_profile"]), flush=True)

    R["varB"] = evaluate(mv, ev, a.cap, a.K, a.alpha, "varB", do_pf=True)
    print("  VARB:", json.dumps(R["varB"], indent=1), flush=True)

    R["shift"] = shift_test(mv, ev[:a.n_shift], a.cap, a.K, a.alpha, a.shift)
    print("  SHIFT:", json.dumps(R["shift"]), flush=True)

    json.dump(R, open(a.out, "w"), indent=2)
    print("\nwrote", a.out, flush=True)


if __name__ == "__main__":
    main()
