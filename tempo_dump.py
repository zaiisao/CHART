"""Freeze the frontend's beat+downbeat activations for val/gtzan windows, once.

Every bar-period estimator candidate reads exactly the same [N, T, 2] activations, the
same mask and the same annotated period, so comparing them costs no GPU and no frontend
pass. Deterministic (middle) windows -- the ones evaluate() scores on.
"""
from __future__ import annotations

import argparse
import importlib

import numpy as np
import torch

from phasevae.data.dataset import split_songs
from phasevae.data.excerpts import ExcerptDataset, collate_excerpts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--excerpt-seconds", type=float, default=45.0)
    ap.add_argument("--out", default="tempo_acts.npz")
    ap.add_argument("--splits", default="val,gtzan")
    ap.add_argument("--repeats", type=int, default=1,
                    help="passes over the split; >1 draws FRESH random windows per pass")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    frontend = importlib.import_module("phasevae.data.frontends.beat_this").FRONTEND(
        checkpoint="final0", device=device, output="features+activations")

    train_songs, val_songs, test_songs = split_songs()
    acts, masks, truths, splits, datasets, stems = [], [], [], [], [], []

    available = {"train": train_songs, "val": val_songs, "gtzan": test_songs}
    for split_name in args.splits.split(","):
        songs = available[split_name]
        data = ExcerptDataset(songs, frontend, args.excerpt_seconds,
                              deterministic=args.repeats == 1)
        loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4,
                                             collate_fn=collate_excerpts)
        for raw in (batch for _ in range(args.repeats) for batch in loader):
            with torch.no_grad():
                h = frontend.forward_features(raw["input"])
                acts.append(h[..., -2:].float().cpu().numpy().astype(np.float16))
            masks.append(raw["mask"].numpy().astype(np.uint8))
            truths.append(raw["bar_period"].numpy())
            datasets.extend(raw["dataset"])
            stems.extend(raw["stem"])
            splits.extend([split_name] * len(raw["bar_period"]))
        print(f"  {split_name}: {sum(len(a) for a in acts)} windows", flush=True)

    width = max(a.shape[1] for a in acts)

    def pad(arrays, fill):
        out = np.full((sum(len(a) for a in arrays), width) + arrays[0].shape[2:],
                      fill, dtype=arrays[0].dtype)
        row = 0
        for a in arrays:
            out[row:row + len(a), :a.shape[1]] = a
            row += len(a)
        return out

    np.savez_compressed(args.out, act=pad(acts, 0.0), mask=pad(masks, 0),
                        truth=np.concatenate(truths), split=np.array(splits),
                        dataset=np.array(datasets), stem=np.array(stems),
                        fps=frontend.FPS)
    print(f"wrote {args.out}: {len(np.concatenate(truths))} windows, T={width}")


if __name__ == "__main__":
    main()
