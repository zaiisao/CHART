"""Does the tempo-level prior halve the slow songs? Asked at decode time only.

Overwrites the initial tempo prior of a TRAINED checkpoint and re-decodes. No
training, so a change here is the prior steering inference and nothing else.
"""
from __future__ import annotations

import argparse
import importlib
import math

import numpy as np
import torch

from ..config import load_config
from ..variants.base import load_model_state
from ..data.dataset import load_catalog
from ..data.excerpts import ExcerptDataset, collate_excerpts
from ..scoring.evaluation import (f_measure, rule_g_times, scoring_records,
                                  trajectory_period)


def set_prior(model, mu, sigma):
    """Rebuild tempo_log_prior over the existing rate grid; None sigma = uniform."""
    rates = model.rates
    if sigma is None:
        lp = torch.zeros_like(rates)
    else:
        lp = -0.5 * ((torch.log(rates) - mu) / sigma) ** 2
    model.tempo_log_prior.copy_(lp - torch.logsumexp(lp, 0))


def main():
    """Sweep the prior and report F, split by how slow the true bar is."""
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

    songs = sorted(sum(load_catalog(["gtzan"]).values(), []), key=lambda s: s.song_id)
    data = ExcerptDataset(songs, frontend, cfg.excerpt_seconds, deterministic=True,
                          target_tol_frames=getattr(cfg, "target_tol_frames", 0))
    loader = torch.utils.data.DataLoader(data, batch_size=cfg.batch_size,
                                         collate_fn=collate_excerpts)

    mu0, s0 = float(model.walk.tempo_mu), float(model.walk.tempo_sigma)
    arms = [("as trained", mu0, s0)]
    arms += [(f"sigma x{k}", mu0, s0 * k) for k in (1.5, 2.0, 3.0)]
    arms += [(f"mu {d:+.2f}", mu0 + d, s0) for d in (-0.35, -0.69)]
    arms += [("uniform", mu0, None)]

    cache = []
    with torch.no_grad():
        for raw in loader:
            recs = scoring_records(raw)
            keep = [i for i, c in enumerate(recs) if c is not None]
            if keep:
                cache.append((frontend.forward_features(raw["input"]).clone(),
                              raw["mask"].to(device), keep, [recs[i] for i in keep]))

    print(f"{'arm':>12} {'F all':>7} {'F bar<2.5':>10} {'F bar>3.0':>10} "
          f"{'halved>3s':>10} {'est p90':>8}")
    for name, mu, sig in arms:
        set_prior(model, mu, sig)
        F, rel, ref = [], [], []
        with torch.no_grad():
            for h, mask, keep, crops in cache:
                mu_t = model.infer_phase(h, mask)[keep]
                times = rule_g_times(mu_t, mask[keep], crops)
                per = trajectory_period(mu_t, mask[keep], crops[0]["fps"])
                for i, c in enumerate(crops):
                    truth = np.asarray(c["downbeat_times"])
                    r = float(np.median(np.diff(truth))) if len(truth) > 1 else float("nan")
                    F.append(f_measure(times[i], truth)[0])
                    ref.append(r)
                    rel.append(float(per[i]) / r if r > 0 else float("nan"))
        F, rel, ref = np.array(F), np.array(rel), np.array(ref)
        slow, fast = ref > 3.0, ref < 2.5
        hv = ((rel > 0.42) & (rel < 0.58) & slow).sum() / max(slow.sum(), 1)
        print(f"{name:>12} {F.mean():7.4f} {F[fast].mean():10.4f} {F[slow].mean():10.4f} "
              f"{hv:9.1%} {np.nanpercentile(rel * ref, 90):8.2f}")


if __name__ == "__main__":
    main()
