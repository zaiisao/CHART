"""Runner for VARIANT A+B (audio-conditioned prior mean + observation decoder + PF deploy).

  python run.py --regime dirac --variant ab
  python run.py --regime mert  --variant baseline
"""
import sys, os, json, time, math, argparse
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")

import numpy as np
import torch

from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run
from model_ab import BarPointerVAE_AB
from elbo_ab import elbo_ab, free_run_ab, particle_filter
import common as C

DEV = "cuda:0"


# ---------------------------------------------------------------- deploy wrappers
@torch.no_grad()
def deploy_phase(kind, model, h, K, temper, seed):
    torch.manual_seed(seed)
    if kind == "baseline_freerun":
        return free_run(model, h)["phase_mu"][0].float().cpu().numpy()
    if kind == "A_freerun":
        return free_run_ab(model, h, use_corr=True)["phase_mu"][0].float().cpu().numpy()
    if kind == "noA_freerun":
        return free_run_ab(model, h, use_corr=False)["phase_mu"][0].float().cpu().numpy()
    if kind in ("B_filter", "AB_filter"):
        r = particle_filter(model, h, K=K, use_corr=(kind == "AB_filter"), temper=temper)
        return r["phase_path"][0].float().cpu().numpy(), r["phase_mean"][0].float().cpu().numpy()
    raise ValueError(kind)


def build_h(regime, song, T, merge, shift=0, rng=None):
    if regime == "dirac":
        h = C.dirac_h(song["beats"], song["downs"], 0, T, shift_frames=shift, rng=rng)
        return torch.from_numpy(h).unsqueeze(0).to(DEV)
    f = torch.from_numpy(song["feats"][:, :T, :].astype(np.float32)).unsqueeze(0).to(DEV)
    return merge(f)


@torch.no_grad()
def evaluate(regime, kinds, model, merge, songs, K, temper, max_frames, seed=1234, tag=""):
    model.eval()
    if merge is not None: merge.eval()
    rows = {k: [] for k in kinds}
    rows_mean = {k: [] for k in kinds if "filter" in k}
    for s in songs:
        T = min(s["T"], max_frames)
        rng = np.random.default_rng(0)
        h = build_h(regime, s, T, merge, rng=rng)
        for k in kinds:
            p = deploy_phase(k, model, h, K, temper, seed)
            if isinstance(p, tuple):
                rows[k].append(C.score_phase(p[0], s, T))
                rows_mean[k].append(C.score_phase(p[1], s, T))
            else:
                rows[k].append(C.score_phase(p, s, T))
    model.train()
    if merge is not None: merge.train()
    out = {}
    for k in kinds:
        out[k] = C.aggregate(rows[k])
        if k in rows_mean:
            out[k + "|circmean"] = C.aggregate(rows_mean[k])
    if tag:
        print(f"  [{tag}] " + json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()}
                                          for k, v in out.items()}), flush=True)
    return out


@torch.no_grad()
def shift_test(kinds, model, songs, K, temper, max_frames, shift=25, seed=1234):
    """MANDATORY mechanism check: shift the Dirac impulses by +25 frames, same seed."""
    model.eval()
    res = {k: [] for k in kinds}
    for s in songs:
        T = min(s["T"], max_frames)
        h0 = torch.from_numpy(C.dirac_h(s["beats"], s["downs"], 0, T, 0,
                                        np.random.default_rng(0))).unsqueeze(0).to(DEV)
        h1 = torch.from_numpy(C.dirac_h(s["beats"], s["downs"], 0, T, shift,
                                        np.random.default_rng(0))).unsqueeze(0).to(DEV)
        for k in kinds:
            a = deploy_phase(k, model, h0, K, temper, seed)
            b = deploy_phase(k, model, h1, K, temper, seed)
            if isinstance(a, tuple): a, b = a[0], b[0]
            res[k].append(float(C.circ_absdiff(a, b).max()))
    model.train()
    return {k: dict(max_over_songs=float(np.max(v)), mean_of_per_song_max=float(np.mean(v)))
            for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["dirac", "mert"], required=True)
    ap.add_argument("--variant", choices=["baseline", "ab"], required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--temper", type=float, default=1.0)
    ap.add_argument("--obs_weight", type=float, default=1.0)
    ap.add_argument("--max_phase_corr", type=float, default=0.30)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--max_frames", type=int, default=1600)
    ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.steps is None:  a.steps = 800 if a.regime == "dirac" else 1200
    if a.warmup is None: a.warmup = 400 if a.regime == "dirac" else 600
    if a.frames is None: a.frames = 256 if a.regime == "dirac" else 512
    tag = a.out or f"{a.regime}_{a.variant}"
    outdir = f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/runs/{tag}"
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(0); rng = np.random.default_rng(0)
    with_feats = (a.regime == "mert")
    train = C.load_split("train", with_feats=with_feats)
    ev = C.load_split("eval", cap=a.n_eval, with_feats=with_feats)
    print(f"[{tag}] train {len(train)} eval {len(ev)} | steps {a.steps} warm {a.warmup} "
          f"frames {a.frames} bs {a.bs} K {a.K} temper {a.temper} obs_w {a.obs_weight}", flush=True)

    h_dim = C.H_DIM_DIRAC if a.regime == "dirac" else 768
    merge = C.LayerMerge().to(DEV) if a.regime == "mert" else None
    if a.variant == "baseline":
        model = BarPointerVAE(h_dim=h_dim, hidden=a.hidden, num_meters=4).to(DEV)
        kinds = ["baseline_freerun"]
    else:
        model = BarPointerVAE_AB(h_dim=h_dim, hidden=a.hidden, num_meters=4,
                                 max_phase_corr=a.max_phase_corr).to(DEV)
        kinds = ["A_freerun", "noA_freerun", "B_filter", "AB_filter"]
    params = list(model.parameters()) + (list(merge.parameters()) if merge else [])
    opt = torch.optim.AdamW(params, lr=a.lr)

    log = open(f"{outdir}/metrics.jsonl", "w")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        beta = min(1.0, step / max(a.warmup, 1))
        temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        hs, bs_, ds = [], [], []
        for _ in range(a.bs):
            s = train[rng.integers(len(train))]
            if s["T"] <= a.frames: continue
            st = int(rng.integers(0, s["T"] - a.frames))
            if a.regime == "dirac":
                hs.append(torch.from_numpy(C.dirac_h(s["beats"], s["downs"], st, a.frames, rng=rng)))
            else:
                hs.append(torch.from_numpy(s["feats"][:, st:st + a.frames, :].astype(np.float32)))
            b, d = C.targets(s["beats"], s["downs"], st, a.frames)
            bs_.append(torch.from_numpy(b)); ds.append(torch.from_numpy(d))
        H = torch.stack(hs).to(DEV)
        b = torch.stack(bs_).to(DEV); d = torch.stack(ds).to(DEV)
        h = merge(H) if merge is not None else H

        opt.zero_grad()
        if a.variant == "baseline":
            loss, info = strict_elbo(model, h, b, d, temperature=temp, beta=beta)
        else:
            loss, info = elbo_ab(model, h, b, d, temperature=temp, beta=beta,
                                 obs_weight=a.obs_weight, use_corr=True)
        if not torch.isfinite(loss):
            print("NaN@", step, flush=True); break
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()

        if step % 100 == 0 or step == 1:
            print(f"s{step:5d} b={beta:.2f} rec_b={info['recon_beat']:7.2f} "
                  f"rec_db={info['recon_db']:7.2f} rec_o={info.get('recon_obs', 0):9.1f} "
                  f"kl(phi={info['kl_phase']:.2f} lv={info['kl_level']:.2f} dv={info['kl_dev']:.2f}) "
                  f"| {step/(time.time()-t0):.2f} it/s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            fast = [k for k in kinds if "filter" not in k]
            r = evaluate(a.regime, fast, model, merge, ev[:15], a.K, a.temper,
                         a.max_frames, tag=f"MID s{step}")
            log.write(json.dumps({"step": step, **{k: v["beat_F"] for k, v in r.items()}}) + "\n")
            log.flush()

    torch.save({"model": model.state_dict(),
                "merge": merge.state_dict() if merge else None}, f"{outdir}/final.pt")

    # ---------------- FINAL EVAL (all deploy paths) ----------------
    print(f"\n=== FINAL [{tag}] ===", flush=True)
    final = evaluate(a.regime, kinds, model, merge, ev, a.K, a.temper, a.max_frames,
                     tag="FINAL")
    res = {"tag": tag, "args": vars(a), "final": final}

    # temper sweep for the filter variants
    if a.variant == "ab":
        sweep = {}
        for tp in (0.25, 0.5, 1.0):
            r = evaluate(a.regime, ["AB_filter"], model, merge, ev[:15], a.K, tp,
                         a.max_frames, tag=f"temper={tp}")
            sweep[str(tp)] = r
        res["temper_sweep"] = sweep

    # ---------------- MANDATORY SHIFT TEST (dirac only) ----------------
    if a.regime == "dirac":
        st = shift_test(kinds, model, ev[:5], a.K, a.temper, a.max_frames)
        print("SHIFT TEST (+25 frames):", json.dumps(st, indent=1), flush=True)
        res["shift_test"] = st

    json.dump(res, open(f"{outdir}/result.json", "w"), indent=1)
    print("WROTE", f"{outdir}/result.json", flush=True)


if __name__ == "__main__":
    main()
