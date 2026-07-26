"""VARIANT B on FROZEN MERT features -- THE HONEST TEST (real audio, no labels in h).

h = learnable-softmax-weighted sum over the 13 MERT layers -> [T,768], then a fixed
non-affine LayerNorm.  The observation likelihood p_theta(h_t|z_t) is a 768-d Gaussian
with a learned per-dim scale, so dims predictable from metrical position get a small
sigma and dominate the particle weights while unpredictable dims cancel in the weight
normalisation.

Reported against: the same model's OPEN-LOOP free_run (isolates what FILTERING buys),
the 120-BPM metronome floor, and a DENSITY-MATCHED phase-blind grid floor.
`alpha` (the observation tempering exponent) is tuned on held-out TRAIN songs, never eval.
"""
import argparse, json, math, sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run
from vbpm.evaluate import metronome, f_measure, _estimate_meter
from vbpm_fix.varB_pf.vb import BarPointerVAE_B, MertFront, elbo_b, particle_filter
from vbpm_fix.varB_pf import common as C

FPS = C.FPS
DEV = "cuda:0"


def batch(train, rng, bs, fr):
    fe, bb, dd = [], [], []
    for _ in range(bs):
        s = train[rng.integers(len(train))]
        T = s["feats"].shape[1]
        if T <= fr:
            continue
        st = int(rng.integers(0, T - fr))
        fe.append(torch.from_numpy(s["feats"][:, st:st + fr, :].astype(np.float32)))
        b, d = C.targets(s["beats"], s["downs"], st, fr)
        bb.append(torch.from_numpy(b)); dd.append(torch.from_numpy(d))
    return (torch.stack(fe).to(DEV), torch.stack(bb).to(DEV), torch.stack(dd).to(DEV))


def train_model(kind, train, a, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    front = MertFront().to(DEV)
    if kind == "base":
        model = BarPointerVAE(h_dim=768, hidden=a.hidden, num_meters=4).to(DEV)
    else:
        model = BarPointerVAE_B(h_dim=768, hidden=a.hidden, num_meters=4,
                                obs_mode="gauss", obs_dim=768, n_harm=a.n_harm).to(DEV)
    opt = torch.optim.AdamW(list(front.parameters()) + list(model.parameters()), lr=a.lr)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        beta = min(1.0, step / max(a.warmup, 1))
        temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        feats, b, d = batch(train, rng, a.bs, a.frames)
        h = front(feats)
        opt.zero_grad()
        if kind == "base":
            loss, info = strict_elbo(model, h, b, d, temperature=temp, beta=beta)
        else:
            loss, info = elbo_b(model, h, b, d, h.detach(), temperature=temp,
                                beta=beta, lam_h=a.lam_h)
        if not torch.isfinite(loss):
            print(f"  !! NaN at {step}", flush=True); break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(front.parameters()) + list(model.parameters()), 5.0)
        opt.step()
        if step % 100 == 0:
            extra = f" obs_lp={info.get('obs_lp', float('nan')):10.0f}" if kind != "base" else ""
            print(f"  [{kind}] s{step:5d} b={beta:.2f} rec_b={info['recon_beat']:7.2f}"
                  f"{extra} klphi={info['kl_phase']:.1f} kllv={info['kl_level']:.1f}"
                  f" | {step/(time.time()-t0):.2f} it/s", flush=True)
    return front, model


@torch.no_grad()
def evaluate(front, model, songs, a, do_pf, alpha=None, seed=0):
    front.eval(); model.eval()
    alpha = a.alpha if alpha is None else alpha
    res = {k: [] for k in ["fr", "pf_circ", "pf_map", "pf_anc_mono",
                           "pf_circ_mono", "pf_anc_mono_mono"]}
    met, bpm_r, ess = [], [], []
    for i, s in enumerate(songs):
        T = min(s["feats"].shape[1], a.cap)
        ref = s["beats"][s["beats"] < T / FPS]; dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 2:
            continue
        m = _estimate_meter(ref, dref)
        feats = torch.from_numpy(s["feats"][:, :T, :].astype(np.float32)).unsqueeze(0).to(DEV)
        h = front(feats)
        torch.manual_seed(seed)
        pm = free_run(model, h)["phase_mu"][0, :T].cpu().numpy()
        res["fr"].append(C.score_phase(pm, ref, dref, m, T, tag_seed=i))
        met.append(f_measure(ref, metronome(T, FPS)))
        if do_pf:
            torch.manual_seed(seed)
            pf = particle_filter(model, h, h, K=a.K, alpha=alpha)
            for nm, key in [("pf_circ", "circ"), ("pf_map", "map"), ("pf_anc_mono", "anc")]:
                res[nm].append(C.score_phase(pf[key], ref, dref, m, T, tag_seed=i))
            for nm, key in [("pf_circ_mono", "circ"), ("pf_anc_mono_mono", "anc")]:
                res[nm].append(C.score_phase(C.monotonise(pf[key]), ref, dref, m, T, tag_seed=i))
            ess.append(pf["mean_ess"])
            tb = 60.0 / np.median(np.diff(ref)) if len(ref) > 2 else np.nan
            pb = 60.0 * FPS * m * math.exp(float(np.median(pf["anc_log_tempo"]))) / (2 * math.pi)
            bpm_r.append(pb / tb)
    front.train(); model.train()
    o = {k: C.summarize(v) for k, v in res.items() if v}
    o["metronome"] = float(np.mean(met))
    if ess:
        o["mean_ess"] = float(np.mean(ess)); o["pf_bpm_ratio"] = float(np.median(bpm_r))
        o["alpha"] = alpha
    return o


@torch.no_grad()
def shift_test(front, model, songs, a, shift, seed=0):
    """MANDATORY MECHANISM CHECK on real MERT features: roll the feature sequence by
    `shift` frames (labels untouched) and measure how much the deploy trajectory moves.
    Audio-blind => ~0 regardless of the input."""
    front.eval(); model.eval()
    out = {"fr": [], "pf": []}
    for i, s in enumerate(songs):
        T = min(s["feats"].shape[1], a.cap)
        f0 = torch.from_numpy(s["feats"][:, :T, :].astype(np.float32)).unsqueeze(0).to(DEV)
        f1 = torch.roll(f0, shifts=shift, dims=2)
        h0, h1 = front(f0), front(f1)
        torch.manual_seed(seed); p0 = free_run(model, h0)["phase_mu"][0].cpu().numpy()
        torch.manual_seed(seed); p1 = free_run(model, h1)["phase_mu"][0].cpu().numpy()
        out["fr"].append(C.circ_maxdiff(p0, p1)[0])
        torch.manual_seed(seed); q0 = particle_filter(model, h0, h0, K=a.K, alpha=a.alpha)["anc"]
        torch.manual_seed(seed); q1 = particle_filter(model, h1, h1, K=a.K, alpha=a.alpha)["anc"]
        out["pf"].append(C.circ_maxdiff(q0, q1)[0])
    front.train(); model.train()
    return {k: {"max": float(np.max(v)), "mean": float(np.mean(v))} for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--frames", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_harm", type=int, default=64)
    ap.add_argument("--lam_h", type=float, default=1e-3)
    ap.add_argument("--K", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--alpha_sweep", default="1.0,0.3,0.1,0.03,0.01")
    ap.add_argument("--cap", type=int, default=1600)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--n_tune", type=int, default=10)
    ap.add_argument("--n_shift", type=int, default=6)
    ap.add_argument("--shift", type=int, default=25)
    ap.add_argument("--skip_base", type=int, default=0)
    ap.add_argument("--tag", default="mert")
    a = ap.parse_args()
    out = f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/varB_pf/res_{a.tag}.json"

    train = C.load("train", with_feats=True)
    ev = C.load("eval", a.n_eval, with_feats=True)
    tune = train[:a.n_tune]
    print(f"MERT (HONEST TEST).  train {len(train)} eval {len(ev)} lam_h={a.lam_h} "
          f"n_harm={a.n_harm} K={a.K}", flush=True)
    R = {"config": vars(a)}

    if not a.skip_base:
        print("\n== BASELINE: unmodified VBPM ==", flush=True)
        fb, mb = train_model("base", train, a)
        R["base_freerun"] = evaluate(fb, mb, ev, a, do_pf=False)
        print("  BASE free_run:", json.dumps(R["base_freerun"]), flush=True)
        del fb, mb; torch.cuda.empty_cache()

    print("\n== VARIANT B ==", flush=True)
    fv, mv = train_model("varB", train, a)
    torch.save({"front": fv.state_dict(), "model": mv.state_dict()},
               out.replace(".json", "_model.pt"))
    R["layer_w"] = fv.weights().tolist()

    print("\n-- alpha tuning on HELD-OUT TRAIN songs (never eval) --", flush=True)
    sweep = {}
    for al in [float(x) for x in a.alpha_sweep.split(",")]:
        r = evaluate(fv, mv, tune, a, do_pf=True, alpha=al)
        sweep[al] = r
        print(f"   alpha={al:<6} pf_anc_mono beat_F={r['pf_anc_mono']['beat_F']:.3f} "
              f"n_ratio={r['pf_anc_mono']['n_ratio']:.2f} blind={r['pf_anc_mono']['blind_floor']:.3f} "
              f"ess={r['mean_ess']:.0f}", flush=True)
    best = max(sweep, key=lambda k: sweep[k]["pf_anc_mono"]["beat_F"] - sweep[k]["pf_anc_mono"]["blind_floor"])
    a.alpha = best
    R["alpha_sweep"] = {str(k): v for k, v in sweep.items()}
    R["alpha_chosen"] = best
    print(f"   -> chosen alpha={best} (by beat_F minus density-matched blind floor)", flush=True)

    R["varB"] = evaluate(fv, mv, ev, a, do_pf=True)
    print("  VARB:", json.dumps(R["varB"], indent=1), flush=True)
    R["shift"] = shift_test(fv, mv, ev[:a.n_shift], a, a.shift)
    print("  SHIFT:", json.dumps(R["shift"]), flush=True)

    json.dump(R, open(out, "w"), indent=2)
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()
