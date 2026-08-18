"""The tempo-level prior, the model's aggregated posterior, and the truth.

Three densities on one bar-period axis: what we assumed, what the trained model
actually believes once averaged over songs, and what the annotations say. The
prior is fixed, so any mismatch is ours by declaration rather than by fitting.
"""
from __future__ import annotations

import argparse
import importlib
import math

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import load_config  # noqa: E402
from ..data.dataset import split_songs  # noqa: E402
from ..data.excerpts import ExcerptDataset, collate_excerpts  # noqa: E402
from ..run import VAL_FOLD  # noqa: E402
from ..scoring.evaluation import scoring_records  # noqa: E402

FPS = 50.0


def true_periods(songs):
    """Median annotated bar period per song, in seconds."""
    out = []
    for s in songs:
        d = np.diff(np.asarray(s.beats()[1]))
        if len(d) >= 2:
            out.append(float(np.median(d)))
    return np.asarray(out)


def main():
    """Draw prior, aggregated posterior and truth on the bar-period axis."""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default="docs/tempo_shapes.png")
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, hooks = load_config(blob["config_path"], list(blob.get("overrides", [])))
    frontend = importlib.import_module(f"vbpm.data.frontends.{cfg.frontend}").FRONTEND(
        checkpoint=cfg.frontend_checkpoint, device=device, output="features")
    frontend._audio2frames.model.load_state_dict(blob["frontend"])
    model = hooks.build_model(cfg, frontend.num_channels).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    train_songs, _val, test_songs = split_songs(VAL_FOLD, None)
    periods = 2 * math.pi / (model.rates.cpu().numpy() * FPS)

    agg = torch.zeros(model.tempo_bins, device=device)
    ds = ExcerptDataset(test_songs, frontend, cfg.excerpt_seconds, deterministic=True,
                        target_tol_frames=getattr(cfg, "target_tol_frames", 0))
    ld = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size,
                                     collate_fn=collate_excerpts)
    with torch.no_grad():
        for raw in ld:
            recs = scoring_records(raw)
            keep = [i for i, c in enumerate(recs) if c is not None]
            if not keep:
                continue
            h = frontend.forward_features(raw["input"]).clone()
            mask = raw["mask"].to(device)
            _psi, log_g, _z = model.marginals(h, mask)
            g = log_g.exp().sum(-2)
            agg += (g * mask[..., None])[keep].sum((0, 1))
    agg = (agg / agg.sum()).cpu().numpy()

    prior = model.tempo_log_prior.exp().cpu().numpy()
    prior = prior / prior.sum()

    edges = np.concatenate([[periods[0] * 1.001],
                            np.sqrt(periods[1:] * periods[:-1]),
                            [periods[-1] * 0.999]])[::-1]

    def hist(vals):
        h, _ = np.histogram(vals, bins=edges)
        return h / max(h.sum(), 1)

    tr_train, tr_test = true_periods(train_songs), true_periods(test_songs)
    centres = periods[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    ax.plot(centres, hist(tr_test), lw=2, color="#1b1b1b", label="gtzan truth")
    ax.plot(centres, hist(tr_train), lw=2, color="#8a8a8a", ls=(0, (5, 2)),
            label="train-fold truth")
    ax.plot(centres, agg[::-1], lw=2, color="#2f6fb2",
            label="model aggregated posterior")
    ax.plot(centres, prior[::-1], lw=2, color="#c1553a", ls=(0, (2, 2)),
            label="tempo prior p(rate)")
    ax.set_xscale("log")
    ax.set_xticks([0.8, 1.0, 1.5, 2.0, 3.0, 4.0])
    ax.set_xticklabels(["0.8", "1.0", "1.5", "2.0", "3.0", "4.0"])
    ax.set_xlabel("bar period (s)")
    ax.set_ylabel("probability mass per grid bin")
    ax.set_title("What we assumed, what the model believes, what is true")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")

    def summ(name, w):
        m = float((w * np.log(centres)).sum() / w.sum())
        v = float((w * (np.log(centres) - m) ** 2).sum() / w.sum())
        print(f"  {name:>26}  median bar {math.exp(m):5.2f}s  sd(log) {math.sqrt(v):.3f}")
    print("\non the log-period axis:")
    summ("tempo prior", prior[::-1])
    summ("model aggregated posterior", agg[::-1])
    summ("gtzan truth", hist(tr_test))
    summ("train-fold truth", hist(tr_train))


if __name__ == "__main__":
    main()
