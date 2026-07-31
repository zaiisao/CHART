"""Sanity self-check: train Stage 0 on the synthetic bench and print the diagnostics.

Not an experiment and not part of the pytest suite (no test_ prefix): the bench is the
"simplest data imaginable", so balanced accuracy here should always be ~0.95+ — anything
less means the machinery is broken, not that the model is bad. Real training is train.py
at the repo root.

Usage:
    /disk4/anaconda3/envs/vbpm/bin/python tests/synth_bench.py \
        [--n-per-class 8] [--steps 500] [--lr 0.5] [--seed 0]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import reference as R  # sibling module: the synthetic bench + metrics, spec-side code

from vbpm.stage0 import Stage0


def main(argv=None):
    """Train Stage 0 on the synthetic bench and print the standard diagnostics."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-per-class", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)   # the spec's training defaults
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0, help="bench draw; the model is deterministic")
    args = ap.parse_args(argv)

    values = R.DEFAULT_VALUES
    train = R.make_synth_dataset(args.n_per_class, values=values, seed=args.seed)
    held = R.make_synth_dataset(args.n_per_class, values=values, seed=args.seed + 1000)

    model = Stage0(values)

    def mean_elbo(songs):
        with torch.no_grad():
            return float(np.mean([float(model.elbo(s["h"], s["y"])) for s in songs]))

    print(f"bench: {len(train)} train / {len(held)} held-out songs, values={values}")
    print(f"ELBO before fit: train {mean_elbo(train):+.4f}  held {mean_elbo(held):+.4f}")

    model.fit(train, steps=args.steps, lr=args.lr, seed=args.seed)

    print(f"ELBO after fit:  train {mean_elbo(train):+.4f}  held {mean_elbo(held):+.4f}")

    # score the DEPLOYABLE path (predict reads audio features only) on held-out data
    true = [s["m_true"] for s in held]
    pred = [model.to_value(int(model.predict(s["h"]).argmax())) for s in held]
    logits = np.stack([model.predict(s["h"]).detach().numpy() for s in held])

    print(f"held-out balanced accuracy: {R.balanced_accuracy(true, pred, values):.3f} "
          f"(chance {1 / len(values):.3f})")
    print(f"held-out NLL of true class: {R.nll_true(logits, true, values):.4f}")
    print(f"distinct predicted classes: {R.distinct_predicted(pred)} / {len(values)}")
    print(f"collapse ratio: {R.collapse_ratio(logits):.3f}")
    print(f"confusion (rows=true {values}):\n{R.confusion(true, pred, values)}")

    # every parameter must have moved off its initialisation — assert, don't assume
    fresh = Stage0(values)
    stuck = [n for n, p in model.named_params().items()
             if np.array_equal(p.detach().numpy(), fresh.named_params()[n].detach().numpy())]
    if stuck:
        print(f"WARNING: parameters bit-identical to init after fit: {stuck}")
    print(f"theta: alpha={float(model.alpha):+.3f} beta={float(model.beta):+.3f}")


if __name__ == "__main__":
    main()
