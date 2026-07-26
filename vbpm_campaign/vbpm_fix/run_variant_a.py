"""Runner for VARIANT A (audio-conditioned prior mean) on DIRAC (ceiling) and MERT (honest).

Protocol is copied from vbpm/probe_dirac.py + vbpm/train_mert.py so numbers are comparable
with the established baselines (unfixed Dirac free-run ~0.04-0.26, unfixed MERT ~0.31,
120-BPM metronome ~0.295).

  --baseline 1   uses the UNMODIFIED vbpm.model.BarPointerVAE + vbpm.elbo (reference run)
  otherwise      uses vbpm_fix.variant_a.AudioCondPriorVAE + elbo_A / free_run_A
"""
import argparse, glob, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run
from vbpm.evaluate import (beats_from_barphase, downbeats_from_barphase, metronome,
                           f_measure, _estimate_meter)
from vbpm_fix.variant_a import AudioCondPriorVAE, elbo_A, free_run_A

CACHE = "/disk1/jaehoon/vbpm_mert_cache"
FPS = 50.0
H_DIM_DIRAC = 8


# ------------------------------------------------------------------ data
def load_split(split, cap=None, with_feats=False):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        rec = dict(stem=Path(f).stem,
                   beats=np.asarray(d["beats"], float), downs=np.asarray(d["downs"], float))
        if with_feats:
            rec["feats"] = d["feats"]
            rec["T"] = int(rec["feats"].shape[1])
        else:
            rec["T"] = int(d["feats"].shape[1])
        out.append(rec)
        if cap and len(out) >= cap:
            break
    return out


def dirac_h(beats, downs, start, n, rng, shift=0):
    """h[:,0]=beat impulses, h[:,1]=downbeat impulses (+tiny noise). `shift` in FRAMES."""
    h = rng.standard_normal((n, H_DIM_DIRAC)).astype(np.float32) * 0.01
    for t in beats:
        i = int(round(t * FPS)) - start + shift
        if 0 <= i < n:
            h[i, 0] += 1.0
    for t in downs:
        i = int(round(t * FPS)) - start + shift
        if 0 <= i < n:
            h[i, 1] += 1.0
    return h


def targets(beats, downs, start, n):
    b = np.zeros(n, np.float32); db = np.zeros(n, np.float32)
    for t in beats:
        i = int(round(t * FPS)) - start
        if 0 <= i < n:
            b[i] = 1.0
    for t in downs:
        i = int(round(t * FPS)) - start
        if 0 <= i < n:
            db[i] = 1.0
    return b, db


# ------------------------------------------------------------------ eval
@torch.no_grad()
def eval_freerun(model, songs, dev, fr_fn, h_of_song, max_frames=1600):
    model.eval()
    acc = {"beat_phase": [], "downbeat_phase": [], "metronome": [], "ratio": [], "lt": []}
    for s in songs:
        T = min(s["T"], max_frames)
        h = h_of_song(s, T).to(dev)
        out = fr_fn(model, h)
        pm = out["phase_mu"][0, :T].cpu().numpy()
        ref = s["beats"][s["beats"] < T / FPS]
        dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 2:
            continue
        m = _estimate_meter(ref, dref)
        est = beats_from_barphase(pm, m, FPS)
        acc["beat_phase"].append(f_measure(ref, est))
        acc["ratio"].append(len(est) / max(len(ref), 1))
        if len(dref) >= 2:
            acc["downbeat_phase"].append(
                f_measure(dref, downbeats_from_barphase(pm, FPS)))
        acc["metronome"].append(f_measure(ref, metronome(T, FPS)))
        acc["lt"].append(float(out["log_tempo"][0, :T].mean()))
    model.train()
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


# ------------------------------------------------------------------ shift test
@torch.no_grad()
def shift_test(model, songs, dev, fr_fn, shift=25, max_frames=1000, seed=1234):
    """MECHANISM CHECK: rebuild Dirac h with beats shifted +`shift` frames, run the deploy
    path with the SAME seed, report max |circular difference| of the phase trajectory."""
    model.eval()
    diffs, corr_rng = [], []
    for s in songs:
        T = min(s["T"], max_frames)
        rng = np.random.default_rng(0)
        h0 = torch.from_numpy(dirac_h(s["beats"], s["downs"], 0, T, rng, 0)).unsqueeze(0).to(dev)
        rng = np.random.default_rng(0)
        h1 = torch.from_numpy(dirac_h(s["beats"], s["downs"], 0, T, rng, shift)).unsqueeze(0).to(dev)
        torch.manual_seed(seed); o0 = fr_fn(model, h0)
        torch.manual_seed(seed); o1 = fr_fn(model, h1)
        p0 = o0["phase_mu"][0, :T].cpu().numpy(); p1 = o1["phase_mu"][0, :T].cpu().numpy()
        d = np.abs(np.angle(np.exp(1j * (p1 - p0))))
        diffs.append(float(d.max()))
        if "corr" in o0:
            c = o0["corr"][0, :T].cpu().numpy()
            corr_rng.append(float(c.max() - c.min()))
    model.train()
    return float(np.mean(diffs)), (float(np.mean(corr_rng)) if corr_rng else float("nan"))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dirac", "mert"], default="dirac")
    ap.add_argument("--baseline", type=int, default=0)
    ap.add_argument("--corr_scale", type=float, default=0.5)
    ap.add_argument("--tempo_corr_scale", type=float, default=0.0)
    ap.add_argument("--tempo_init", type=int, default=1)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="A")
    ap.add_argument("--out", default="/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/runs")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    log = open(outdir / f"{a.tag}.jsonl", "w")

    with_feats = (a.mode == "mert")
    train = load_split("train", with_feats=with_feats)
    ev = load_split("eval", cap=a.n_eval, with_feats=with_feats)
    print(f"[{a.tag}] mode={a.mode} baseline={a.baseline} corr={a.corr_scale} "
          f"tcorr={a.tempo_corr_scale} tinit={a.tempo_init} | train {len(train)} eval {len(ev)}",
          flush=True)

    if a.mode == "dirac":
        h_dim = H_DIM_DIRAC
        merge = None

        def h_of_song(s, T):
            return torch.from_numpy(
                dirac_h(s["beats"], s["downs"], 0, T, np.random.default_rng(0))).unsqueeze(0)

        def sample_batch():
            hs, bs_, ds = [], [], []
            for _ in range(a.bs):
                s = train[rng.integers(len(train))]
                if s["T"] <= a.frames:
                    continue
                st = int(rng.integers(0, s["T"] - a.frames))
                hs.append(torch.from_numpy(dirac_h(s["beats"], s["downs"], st, a.frames, rng)))
                b, d = targets(s["beats"], s["downs"], st, a.frames)
                bs_.append(torch.from_numpy(b)); ds.append(torch.from_numpy(d))
            return (torch.stack(hs).to(dev), torch.stack(bs_).to(dev), torch.stack(ds).to(dev))
    else:
        h_dim = 768

        class LayerMerge(nn.Module):
            def __init__(self, n=13):
                super().__init__(); self.layer_logits = nn.Parameter(torch.zeros(n))

            def forward(self, feats):
                return torch.einsum("l,bltf->btf", torch.softmax(self.layer_logits, 0), feats)

        merge = LayerMerge().to(dev)

        def h_of_song(s, T):
            f = torch.from_numpy(s["feats"][:, :T, :].astype(np.float32)).unsqueeze(0).to(dev)
            return merge(f)

        def sample_batch():
            fe, bb, dd = [], [], []
            for _ in range(a.bs):
                s = train[rng.integers(len(train))]
                if s["T"] <= a.frames:
                    continue
                st = int(rng.integers(0, s["T"] - a.frames))
                fe.append(torch.from_numpy(s["feats"][:, st:st + a.frames, :].astype(np.float32)))
                b, d = targets(s["beats"], s["downs"], st, a.frames)
                bb.append(torch.from_numpy(b)); dd.append(torch.from_numpy(d))
            return (merge(torch.stack(fe).to(dev)),
                    torch.stack(bb).to(dev), torch.stack(dd).to(dev))

    if a.baseline:
        model = BarPointerVAE(h_dim=h_dim, hidden=a.hidden, num_meters=4).to(dev)
        elbo_fn, fr_fn = strict_elbo, free_run
    else:
        model = AudioCondPriorVAE(h_dim=h_dim, hidden=a.hidden, num_meters=4,
                                  corr_scale=a.corr_scale,
                                  tempo_corr_scale=a.tempo_corr_scale,
                                  tempo_init=bool(a.tempo_init)).to(dev)
        elbo_fn, fr_fn = elbo_A, free_run_A

    params = list(model.parameters()) + (list(merge.parameters()) if merge else [])
    opt = torch.optim.AdamW(params, lr=a.lr)

    # ---- pre-training mechanism check (untrained model) ----
    if a.mode == "dirac":
        d0, c0 = shift_test(model, ev[:5], dev, fr_fn)
        print(f"[{a.tag}] SHIFT-TEST untrained: max|dphi| = {d0:.4f} rad", flush=True)

    best, t0, last_r = -1.0, time.time(), None
    for step in range(1, a.steps + 1):
        beta = min(1.0, step / max(a.warmup, 1))
        temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        h, b, d = sample_batch()
        opt.zero_grad()
        loss, info = elbo_fn(model, h, b, d, temperature=temp, beta=beta)
        if not torch.isfinite(loss):
            print(f"[{a.tag}] NaN@{step}", flush=True); break
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        if step % 100 == 0 or step == 1:
            print(f"[{a.tag}] s{step:5d} b={beta:.2f} rec_b={info['recon_beat']:7.2f} "
                  f"kl_phi={info['kl_phase']:.2f} kl_lv={info['kl_level']:.2f} "
                  f"|corr|={info.get('corr_mean_abs', 0.0):.4f} "
                  f"logT={info.get('log_tempo_mean', float('nan')):.2f} "
                  f"| {step/(time.time()-t0):.2f} it/s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            r = eval_freerun(model, ev, dev, fr_fn, h_of_song)
            print(f"[{a.tag}]   [EVAL s{step}] beat_F={r['beat_phase']:.3f} "
                  f"db_F={r['downbeat_phase']:.3f} metronome={r['metronome']:.3f} "
                  f"n_est/n_true={r['ratio']:.3f} mean_logT={r['lt']:.2f}", flush=True)
            log.write(json.dumps({"step": step, **r}) + "\n"); log.flush()
            if r["beat_phase"] == r["beat_phase"]:
                best = max(best, r["beat_phase"])
            last_r = r

    final = last_r
    res = {"tag": a.tag, "mode": a.mode, "baseline": a.baseline, "corr_scale": a.corr_scale,
           "tempo_corr_scale": a.tempo_corr_scale, "tempo_init": a.tempo_init,
           "steps": a.steps, "best_beat_F": best, "final": final}
    if a.mode == "dirac":
        d1, c1 = shift_test(model, ev[:5], dev, fr_fn)
        res["shift_rad_trained"] = d1
        res["shift_rad_untrained"] = d0
        res["corr_range"] = c1
        print(f"[{a.tag}] SHIFT-TEST trained: max|dphi| = {d1:.4f} rad "
              f"(corr range {c1:.4f})", flush=True)
    else:
        # analogue for MERT: roll the features by 25 frames, same seed
        model.eval(); ds = []
        with torch.no_grad():
            for s in ev[:10]:
                T = min(s["T"], 1600)
                f = torch.from_numpy(s["feats"][:, :T, :].astype(np.float32)).unsqueeze(0).to(dev)
                h0 = merge(f)
                h1 = torch.roll(h0, 25, dims=1)   # shift the observation by +25 frames
                torch.manual_seed(1234); o0 = fr_fn(model, h0)
                torch.manual_seed(1234); o1 = fr_fn(model, h1)
                p0 = o0["phase_mu"][0].cpu().numpy(); p1 = o1["phase_mu"][0].cpu().numpy()
                ds.append(float(np.abs(np.angle(np.exp(1j * (p1 - p0)))).max()))
        model.train()
        res["shift_rad_trained"] = float(np.mean(ds))
        print(f"[{a.tag}] SHIFT-TEST (MERT roll+25) trained: max|dphi| = {res['shift_rad_trained']:.4f} rad",
              flush=True)
    print("RESULT " + json.dumps(res), flush=True)
    (outdir / f"{a.tag}.result.json").write_text(json.dumps(res, indent=2))
    torch.save({"model": model.state_dict(),
                "merge": (merge.state_dict() if merge else None)}, outdir / f"{a.tag}.pt")


if __name__ == "__main__":
    main()
