"""How much of the discrete tempo grid's coarseness costs us, asked for free.

psi_head and the emission are indexed by PHASE only, so the tempo grid is pure
inference machinery: refining it needs no retraining. Sweeping it bounds what a
continuous-velocity posterior could buy before anyone writes a smoother.
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
from ..data.excerpts import ExcerptDataset, collate_excerpts
from ..run import VAL_FOLD
from ..scoring.evaluation import (f_measure, rule_g_times, scoring_records,
                                  trajectory_period)


def regrid(model, bins, lo, hi, band_bins_at_24):
    """Rebuild the tempo grid at a new resolution, holding the PHYSICAL width fixed."""
    dev = model.rates.device
    rates = torch.exp(torch.linspace(math.log(lo), math.log(hi), bins, device=dev))
    model.tempo_bins = bins
    model.rates = rates
    model.band = max(1, int(round(band_bins_at_24 * (bins - 1) / 23)))
    z = (torch.log(rates) - model.walk.tempo_mu) / model.walk.tempo_sigma
    lp = -0.5 * z ** 2
    model.tempo_log_prior = lp - torch.logsumexp(lp, 0)


def main():
    """Sweep tempo-grid resolution; val selects, gtzan is reported beside it."""
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

    _tr, val_songs, test_songs = split_songs(VAL_FOLD, None)
    tol = getattr(cfg, "target_tol_frames", 0)

    def cache_of(songs):
        ds = ExcerptDataset(songs, frontend, cfg.excerpt_seconds, deterministic=True,
                            target_tol_frames=tol)
        ld = torch.utils.data.DataLoader(ds, batch_size=1,
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
                mu = model.infer_phase(h, mask)[keep]
                times = rule_g_times(mu, mask[keep], crops)
                per = trajectory_period(mu, mask[keep], crops[0]["fps"])
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
        return F.mean(), hv, float(np.median(np.abs(rel - 1)))

    lo, hi, band0 = cfg.tempo_lo, cfg.tempo_hi, cfg.tempo_band
    print(f"trained grid: {cfg.tempo_bins} bins, band {band0}\n")
    print(f"{'bins':>6} {'spacing':>9} {'VAL F':>7} {'gtzan F':>8} {'halved':>7} "
          f"{'|rel-1|':>8}")
    for bins in (96, 192):
        regrid(model, bins, lo, hi, band0)
        sp = math.log(hi / lo) / (bins - 1)
        v = score(caches["val"])[0]
        g, hv, err = score(caches["gtzan"])
        tag = "  (trained)" if bins == cfg.tempo_bins else ""
        print(f"{bins:6d} {sp:9.4f} {v:7.4f} {g:8.4f} {hv:7.1%} {err:8.4f}{tag}")


if __name__ == "__main__":
    main()
