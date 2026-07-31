"""Train Stage 0 (§5) on the synthetic bench (§6.4) and report §8-style diagnostics.

Real corpora are §6 and deliberately not wired here — the spec's data path has never been
executed and this script must not silently become its first, untested consumer. The bench
generator is reused from tests/v2/reference.py, the spec's executable copy of §6.4.

Usage:
    python -m vbpm.train [--n-per-class 8] [--steps 500] [--lr 0.5] [--seed 0]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "tests" / "v2"))
import reference as R  # noqa: E402  (the §6.4 bench + §8 metrics, spec-side code)

from .stage0 import Stage0  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-per-class", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)   # §5 defaults
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0, help="bench draw, not model noise (§4.6)")
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

    # §8: score the DEPLOYABLE path (predict reads h only, C2) on held-out data
    true = [s["m_true"] for s in held]
    pred = [model.to_value(int(model.predict(s["h"]).argmax())) for s in held]
    logits = np.stack([model.predict(s["h"]).detach().numpy() for s in held])

    print(f"held-out balanced accuracy: {R.balanced_accuracy(true, pred, values):.3f} "
          f"(chance {1 / len(values):.3f})")
    print(f"held-out NLL of true class: {R.nll_true(logits, true, values):.4f}")
    print(f"distinct predicted classes: {R.distinct_predicted(pred)} / {len(values)}")
    print(f"collapse ratio: {R.collapse_ratio(logits):.3f}")
    print(f"confusion (rows=true {values}):\n{R.confusion(true, pred, values)}")

    # §4.7 / §10.2: every parameter must have moved off init — assert, don't assume
    fresh = Stage0(values)
    stuck = [n for n, p in model.named_params().items()
             if np.array_equal(p.detach().numpy(), fresh.named_params()[n].detach().numpy())]
    if stuck:
        print(f"WARNING: parameters bit-identical to init after fit: {stuck}")
    print(f"theta: alpha={float(model.alpha):+.3f} beta={float(model.beta):+.3f}")


if __name__ == "__main__":
    main()
