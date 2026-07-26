"""HONEST test for VARIANT B: frozen MERT features (no labels in the input).

Baseline (unfixed VBPM, strict ELBO, open-loop free_run) vs Variant B
(observation decoder p_theta(h_t|z_t) in the ELBO + bootstrap particle filter at deploy).

Observation target for MERT: a FIXED, unsupervised PCA-32 basis of the uniform-layer-mean
MERT features, z-scored over time inside each crop.  Fixed on purpose -- a *learned*
observation target could be maximised by collapsing h to a constant, which would be a
representation-collapse artifact rather than a generative model of the audio.
"""
import sys, json, time, math, argparse
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
import numpy as np, torch, torch.nn as nn

from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run
from vbpm_fix.variant_b import BarPointerVAE_B, elbo_b, particle_filter
from vbpm_fix.common import (load_split, targets, score_phase, shift_stats, agg, ratio, FPS)

DEV = "cuda:0"
OBS_DIM = 32
SMOOTH = 5


class LayerMerge(nn.Module):
    def __init__(self, n_layers=13):
        super().__init__(); self.layer_logits = nn.Parameter(torch.zeros(n_layers))
    def forward(self, feats):                       # [B,13,T,768]
        return torch.einsum("l,bltf->btf", torch.softmax(self.layer_logits, 0), feats)
    def weights(self):
        return torch.softmax(self.layer_logits.detach(), 0).cpu().numpy()


class FixedProj(nn.Module):
    """Fixed PCA-32 projection of the UNIFORM-layer-mean MERT frame, z-scored over time."""
    def __init__(self, mean, comps):
        super().__init__()
        self.register_buffer("mean", mean); self.register_buffer("comps", comps)

    @torch.no_grad()
    def forward(self, feats):                       # [B,13,T,768] -> [B,T,OBS_DIM]
        x = feats.mean(1)
        o = (x - self.mean) @ self.comps
        return (o - o.mean(1, keepdim=True)) / o.std(1, keepdim=True).clamp(min=1e-4)


def fit_pca(train, rng, n_songs=48, n_frames=250):
    X = []
    for s in train[:n_songs]:
        f = s["feats"]; T = f.shape[1]
        idx = rng.choice(T, size=min(n_frames, T), replace=False)
        X.append(np.asarray(f[:, idx, :], np.float32).mean(0))
    X = torch.from_numpy(np.concatenate(X, 0))
    mu = X.mean(0)
    _, _, V = torch.pca_lowrank(X - mu, q=OBS_DIM, niter=4)
    return mu, V[:, :OBS_DIM].contiguous()


def sample_batch(train, rng, bs, frames):
    fe, bb, dd = [], [], []
    for _ in range(bs):
        s = train[rng.integers(len(train))]; T = s["feats"].shape[1]
        if T <= frames: continue
        st = int(rng.integers(0, T - frames))
        fe.append(torch.from_numpy(s["feats"][:, st:st + frames, :].astype(np.float32)))
        b, d = targets(s["beats"], s["downs"], st, frames)
        bb.append(torch.from_numpy(b)); dd.append(torch.from_numpy(d))
    return (torch.stack(fe).to(DEV), torch.stack(bb).to(DEV), torch.stack(dd).to(DEV))


def song_feats(s, T, roll=0):
    f = torch.from_numpy(s["feats"][:, :T, :].astype(np.float32)).unsqueeze(0)
    if roll:
        f = torch.roll(f, shifts=roll, dims=2)
    return f.to(DEV)


@torch.no_grad()
def eval_openloop(merge, model, songs, max_frames, seed=1234):
    rows = []
    for i, s in enumerate(songs):
        T = min(s["feats"].shape[1], max_frames)
        torch.manual_seed(seed + i)
        h = merge(song_feats(s, T))
        rows.append(score_phase(free_run(model, h)["phase_mu"][0, :T].cpu().numpy(), s, T))
    return rows


@torch.no_grad()
def eval_filter(merge, proj, model, songs, max_frames, K, alpha=1.0, diffuse=True, seed=1234):
    rm, rp, rs, ess = [], [], [], []
    for i, s in enumerate(songs):
        T = min(s["feats"].shape[1], max_frames)
        torch.manual_seed(seed + i)
        f = song_feats(s, T)
        out = particle_filter(model, merge(f), proj(f), K=K, alpha=alpha, diffuse_init=diffuse)
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


def shift_test(merge, proj, model, songs, max_frames, mode, K=400, roll=25, seed=99):
    out = []
    for s in songs:
        T = min(s["feats"].shape[1], max_frames)
        trajs = []
        for r in (0, roll):
            torch.manual_seed(seed)
            f = song_feats(s, T, roll=r)
            if mode == "free":
                trajs.append(free_run(model, merge(f))["phase_mu"][0, :T].cpu().numpy())
            else:
                trajs.append(particle_filter(model, merge(f), proj(f), K=K)["phase_mean"].numpy())
        # drop the wrap-around region introduced by torch.roll
        out.append(shift_stats(trajs[0][roll + 50:], trajs[1][roll + 50:]))
    return {k: (float(np.mean([o[k] for o in out])) if k != "lag"
                else [o["lag"] for o in out]) for k in out[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--frames", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--K", type=int, default=400)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--max_frames", type=int, default=1600)
    ap.add_argument("--n_shift", type=int, default=8)
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--only_baseline", action="store_true")
    ap.add_argument("--out", default="/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/results_mert.json")
    a = ap.parse_args()

    train = load_split("train", with_feats=True); ev = load_split("eval", a.n_eval, with_feats=True)
    print(f"MERT: train {len(train)} eval {len(ev)}", flush=True)
    rng0 = np.random.default_rng(0)
    torch.manual_seed(0)   # pca_lowrank is randomised -- pin it
    mu, comps = fit_pca(train, rng0)
    proj = FixedProj(mu, comps).to(DEV)
    res = {"config": vars(a)}

    # ---------------- (a) UNFIXED baseline ----------------
    if not a.skip_baseline:
        torch.manual_seed(0); rng = np.random.default_rng(0)
        merge0 = LayerMerge().to(DEV)
        base = BarPointerVAE(h_dim=768, hidden=a.hidden, num_meters=4).to(DEV)
        opt = torch.optim.AdamW(list(merge0.parameters()) + list(base.parameters()), lr=a.lr)
        t0 = time.time()
        for step in range(1, a.steps + 1):
            beta = min(1.0, step / a.warmup); temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
            f, b, d = sample_batch(train, rng, a.bs, a.frames)
            opt.zero_grad(); loss, info = strict_elbo(base, merge0(f), b, d, temperature=temp, beta=beta)
            if not torch.isfinite(loss): print("NaN@", step, flush=True); break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(merge0.parameters()) + list(base.parameters()), 5.0)
            opt.step()
            if step % 200 == 0:
                print(f"  [base] s{step} rec_b={info['recon_beat']:.1f} kl_phi={info['kl_phase']:.2f} "
                      f"{step/(time.time()-t0):.2f} it/s", flush=True)
        merge0.eval(); base.eval()
        r = eval_openloop(merge0, base, ev, a.max_frames)
        print("BASELINE " + summarize("free_run(open-loop)", r), flush=True)
        st = shift_test(merge0, proj, base, ev[:a.n_shift], a.max_frames, "free")
        print(f"BASELINE SHIFT (free_run, roll+25f): max|dphi|={st['max']:.4f} mean={st['mean']:.4f} lags={st['lag']}", flush=True)
        res["baseline_freerun"] = rec(r); res["baseline_shift"] = st
        del base, merge0, opt; torch.cuda.empty_cache()
        if a.only_baseline:
            json.dump(res, open(a.out, "w"), indent=2); print("WROTE " + a.out, flush=True); return

    # ---------------- (b) VARIANT B ----------------
    torch.manual_seed(0); rng = np.random.default_rng(0)
    merge = LayerMerge().to(DEV)
    model = BarPointerVAE_B(h_dim=768, hidden=a.hidden, num_meters=4,
                            obs_dim=OBS_DIM, obs_type="gauss").to(DEV)
    opt = torch.optim.AdamW(list(merge.parameters()) + list(model.parameters()), lr=a.lr)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        beta = min(1.0, step / a.warmup); temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        f, b, d = sample_batch(train, rng, a.bs, a.frames)
        opt.zero_grad(); loss, info = elbo_b(model, merge(f), b, d, proj(f), temperature=temp, beta=beta)
        if not torch.isfinite(loss): print("NaN@", step, flush=True); break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(merge.parameters()) + list(model.parameters()), 5.0)
        opt.step()
        if step % 200 == 0:
            print(f"  [varB] s{step} rec_b={info['recon_beat']:.1f} rec_obs={info['recon_obs']:.1f} "
                  f"kl_phi={info['kl_phase']:.2f} {step/(time.time()-t0):.2f} it/s", flush=True)
    merge.eval(); model.eval()
    torch.save({"merge": merge.state_dict(), "model": model.state_dict(),
                "pca_mean": mu.cpu(), "pca_comps": comps.cpu()},
               "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/varB_mert.pt")

    r_free = eval_openloop(merge, model, ev, a.max_frames)
    print("VARIANT-B " + summarize("free_run(open-loop, SAME trained model)", r_free), flush=True)
    res["varB_freerun"] = rec(r_free); res["metronome"] = agg(r_free, "metronome")

    for alpha in (1.0, 3.0):
        rm, rp, rs, ess = eval_filter(merge, proj, model, ev, a.max_frames, a.K, alpha=alpha)
        print("VARIANT-B " + summarize(f"PF(K={a.K},a={alpha}) wmean ", rm) + f" ESS={ess:.1f}", flush=True)
        print("VARIANT-B " + summarize(f"PF(K={a.K},a={alpha}) MAP   ", rp), flush=True)
        print("VARIANT-B " + summarize(f"PF(K={a.K},a={alpha}) smooth", rs), flush=True)
        res[f"varB_pf_mean_a{alpha}"] = dict(rec(rm), ess=ess)
        res[f"varB_pf_map_a{alpha}"] = rec(rp)
        res[f"varB_pf_smooth_a{alpha}"] = rec(rs)

    stf = shift_test(merge, proj, model, ev[:a.n_shift], a.max_frames, "free")
    stp = shift_test(merge, proj, model, ev[:a.n_shift], a.max_frames, "pf", K=a.K)
    print(f"VARIANT-B SHIFT free_run: max|dphi|={stf['max']:.4f} mean={stf['mean']:.4f} lags={stf['lag']}", flush=True)
    print(f"VARIANT-B SHIFT PF      : max|dphi|={stp['max']:.4f} mean={stp['mean']:.4f} lags={stp['lag']} "
          f"cost min/max={stp['lag_cost_min']:.3f}/{stp['lag_cost_max']:.3f}", flush=True)
    res["varB_shift_freerun"] = stf; res["varB_shift_pf"] = stp
    res["layer_w"] = merge.weights().tolist()

    json.dump(res, open(a.out, "w"), indent=2)
    print("WROTE " + a.out, flush=True)


if __name__ == "__main__":
    main()
