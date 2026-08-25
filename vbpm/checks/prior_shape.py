"""Which tempo-level prior shape, chosen on VAL and only then read on gtzan.

The earlier sweep compared arms on gtzan, which selects on the test set. Here
the shape is picked by val F; gtzan is scored once per shape and reported
beside it so the selection is visible rather than hidden.
"""
from __future__ import annotations

import argparse
import importlib
import math

import numpy as np
import torch

from ..config import load_config
from ..variants.base import load_model_state
from ..data.dataset import split_songs
from ..run import VAL_FOLD
from ..data.excerpts import ExcerptDataset, collate_excerpts
from ..scoring.evaluation import (f_measure, rule_g_times, scoring_records,
                                  trajectory_period)


def shapes(rates, mu, sigma, empirical):
    """Named log-priors over the rate grid, each normalised by the caller."""
    lr = torch.log(rates)
    z = (lr - mu) / sigma
    out = {
        "gaussian (trained)": -0.5 * z ** 2,
        "uniform in log-rate": torch.zeros_like(lr),
        "uniform in period": -lr,
        "student-t nu=3": -2.0 * torch.log1p(z ** 2 / 3.0),
        "student-t nu=1": -torch.log1p(z ** 2),
        "gaussian sigma x2": -0.5 * (z / 2) ** 2,
        "laplace": -z.abs(),
    }
    for r in (1.5, 2.0, 3.0):
        wide = torch.where(lr < mu, sigma * r, sigma)
        out[f"two-piece slow x{r}"] = -0.5 * ((lr - mu) / wide) ** 2
    for a in (-2.0, -4.0):
        out[f"skew-normal a={a}"] = (-0.5 * z ** 2
                                     + torch.log(torch.erfc(-a * z / math.sqrt(2)) + 1e-12))
    for tail in (0.5, 1.0):
        out[f"gauss+slow tail {tail}"] = torch.logaddexp(
            -0.5 * z ** 2, torch.log(torch.tensor(tail)) - 0.5 * ((lr - (mu - 0.69)) / sigma) ** 2)
    if empirical is not None:
        out["corpus empirical"] = empirical
    return out


def corpus_log_hist(songs, rates, smooth=1.0):
    """Smoothed log-histogram of TRAIN bar rates on the model's own grid."""
    lr = torch.log(rates)
    edges = torch.cat([lr[:1] - 1e3, (lr[1:] + lr[:-1]) / 2, lr[-1:] + 1e3])
    counts = torch.full_like(lr, smooth)
    for s in songs:
        d = np.diff(np.asarray(s.beats()[1]))
        if len(d) < 2:
            continue
        rate = 2 * math.pi / (float(np.median(d)) * 50.0)
        idx = int(torch.searchsorted(edges, torch.tensor(math.log(rate))) - 1)
        if 0 <= idx < len(counts):
            counts[idx] += 1.0
    return torch.log(counts)


def main():
    """Score every shape on val and gtzan; the selection rule is val."""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, hooks = load_config(blob["config_path"], list(blob.get("overrides", [])))
    frontend = importlib.import_module(f"vbpm.data.frontends.{cfg.frontend}").FRONTEND(
        checkpoint=cfg.frontend_checkpoint, device=device, output="features")
    frontend._audio2frames.model.load_state_dict(blob["frontend"])
    model = hooks.build_model(cfg, frontend.num_channels).to(device)
    load_model_state(model, blob["model"])
    model.eval()

    train_songs, val_songs, test_songs = split_songs(VAL_FOLD, None)
    tol = getattr(cfg, "target_tol_frames", 0)

    def cache_of(songs):
        ds = ExcerptDataset(songs, frontend, cfg.excerpt_seconds, deterministic=True,
                            target_tol_frames=tol)
        ld = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size,
                                         collate_fn=collate_excerpts)
        out = []
        with torch.no_grad():
            for raw in ld:
                recs = scoring_records(raw)
                keep = [i for i, c in enumerate(recs) if c is not None]
                if keep:
                    out.append((frontend.forward_features(raw["input"]).clone(),
                                raw["mask"].to(device), keep, [recs[i] for i in keep]))
        return out

    caches = {"val": cache_of(val_songs), "gtzan": cache_of(test_songs)}

    def score(cache):
        F, ref, rel = [], [], []
        with torch.no_grad():
            for h, mask, keep, crops in cache:
                mu_t = model.infer_phase(h, mask)[keep]
                times = rule_g_times(mu_t, mask[keep], crops)
                per = trajectory_period(mu_t, mask[keep], crops[0]["fps"])
                for i, c in enumerate(crops):
                    truth = np.asarray(c["downbeat_times"])
                    if len(truth) < 2:
                        continue
                    r = float(np.median(np.diff(truth)))
                    F.append(f_measure(times[i], truth)[0])
                    ref.append(r)
                    rel.append(float(per[i]) / r)
        F, ref, rel = np.array(F), np.array(ref), np.array(rel)
        slow = ref > 3.0
        hv = ((rel > 0.42) & (rel < 0.58) & slow).sum() / max(slow.sum(), 1)
        return F.mean(), F[slow].mean(), hv

    emp = corpus_log_hist(train_songs, model.rates.cpu()).to(device)
    table = shapes(model.rates, float(model.walk.tempo_mu),
                   float(model.walk.tempo_sigma), emp)

    print(f"val n={len(val_songs)}  gtzan n={len(test_songs)}\n")
    print(f"{'shape':>20} {'VAL F':>7} {'gtzan F':>8} {'gtzan slow':>11} {'halved':>7}")
    rows = []
    for name, lp in table.items():
        model.tempo_log_prior.copy_(lp - torch.logsumexp(lp, 0))
        v = score(caches["val"])[0]
        g, gs, hv = score(caches["gtzan"])
        rows.append((name, v, g, gs, hv))
        print(f"{name:>20} {v:7.4f} {g:8.4f} {gs:11.4f} {hv:7.1%}")
    best = max(rows, key=lambda r: r[1])
    print(f"\nval picks: {best[0]}  ->  gtzan {best[2]:.4f}")


if __name__ == "__main__":
    main()
