"""Is the tf512 gtzan collapse caused by the CHECKPOINT the features came from?

Standing suspicion: gtzan is the only split whose features come from `final0` rather than
a fold checkpoint. Beat This's 512-dim penultimate basis is not pinned across checkpoints
(nothing in training ties unit 37 of fold3 to unit 37 of final0), so a model that reads
raw features can be handed a systematically rotated input at test time. The autocorr head
is immune by construction (it mean-centres, variance-normalises, and reads periodicity,
which no change of basis destroys); the raw-feature transformer is not.

That is a hypothesis about the FEATURES, and the gtzan comparison cannot test it, because
gtzan differs from the CV folds in two ways at once (different checkpoint AND different
songs). This probe removes the second difference:

    same songs, same labels, same trained model, features recomputed through final0.

Each fold's held-out crops are scored twice -- once through their fold-honest checkpoint
(the baseline), once through final0 -- while the model, the training data and the ground
truth are held fixed. A drop between the two columns is caused by the checkpoint swap and
by nothing else. Read alongside the arms: tf512 is predicted to drop and autocorr not.

NOTE the final0 column is deliberately NOT fold-honest (final0 trained on these songs), so
it is a diagnostic about feature geometry and is never a fold-honest score. If anything it
is biased in the swap's FAVOUR -- final0 has seen these songs -- which makes a drop
stronger evidence, not weaker.

Run (after `python -m vbpm.warm_cache --gpus 0 1 3 --output features+activations` and
`--override final0`, which is what --warm below does):
    CUDA_VISIBLE_DEVICES=1 python experiments/stage0_checkpoint_swap.py
"""
import argparse

import numpy as np

from experiments.stage0_transformer_prior import fit, make_entry, predict
from vbpm.data import VALUES, iter_frontend_features, load_crops, slice_h
from vbpm.fitting import score
from vbpm.metrics import balanced_accuracy

OUTPUT = "features+activations"
SWAP_CHECKPOINT = "final0"
PROBE_ARMS = ["tf512", "autocorr", "tf512n", "linear"]


def swapped_features(crops_by_song, device="cuda"):
    """The same crops, features recomputed through ONE checkpoint for every song."""
    swapped = {}
    for song, h in iter_frontend_features(output=OUTPUT, device=device,
                                          override_checkpoint=SWAP_CHECKPOINT,
                                          verbose=False):
        if song.stem not in crops_by_song:
            continue
        for crop, entry in crops_by_song[song.stem]:
            h_crop, t0 = slice_h(h, crop["beats"])
            swapped[id(entry)] = make_entry(song, crop, h_crop, t0)
    return swapped


def main():
    """Baseline vs checkpoint-swapped scores for each arm, on identical crops."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-fold", type=int, default=None)
    ap.add_argument("--arms", nargs="*", default=PROBE_ARMS)
    args = ap.parse_args()
    device = "cuda"

    # crops keyed by song so the swapped pass can rebuild exactly the same ones
    crops_by_song: dict = {}

    def keyed_entry(song, crop, h_crop, t0):
        entry = make_entry(song, crop, h_crop, t0)
        crops_by_song.setdefault(song.stem, []).append((crop, entry))
        return entry

    crops, report = load_crops(limit_per_fold=args.limit_per_fold, device=device,
                               output=OUTPUT, make_entry=keyed_entry, verbose=False)
    print(f"crops: {len(crops)}  rejects: {report['rejects']}", flush=True)

    cv = [c for c in crops if c["fold"] is not None]
    swapped = swapped_features(crops_by_song, device)
    print(f"swapped features for {len(swapped)} crops through {SWAP_CHECKPOINT}",
          flush=True)

    for arm in args.arms:
        base_true, base_pred, swap_pred = [], [], []
        for fold in sorted({c["fold"] for c in cv}):
            train = [c for c in cv if c["fold"] != fold]
            held = [c for c in cv if c["fold"] == fold]
            fitted = fit(arm, train, device)
            # the standardizer is part of the fitted model and is NOT refitted: the swap
            # must be the only thing that changes between the two columns
            base_pred += predict(arm, fitted, held, device)
            swap_pred += predict(arm, fitted, [swapped[id(c)] for c in held], device)
            base_true += [c["m_true"] for c in held]
            print(f"  [{arm}] fold {fold} done", flush=True)

        base = balanced_accuracy(base_true, base_pred, VALUES)
        swap = balanced_accuracy(base_true, swap_pred, VALUES)
        agree = float(np.mean(np.asarray(base_pred) == np.asarray(swap_pred)))
        print(f"\n######## arm: {arm} ########")
        print(f"  fold-honest checkpoint : balanced={base:.3f}")
        print(f"  {SWAP_CHECKPOINT} features       : balanced={swap:.3f}   "
              f"(delta {swap - base:+.3f}, predictions agree on {agree:.3f})")
        subset = [{"m_true": t} for t in base_true]
        score(f"{arm}/baseline", subset, base_pred, VALUES)
        score(f"{arm}/{SWAP_CHECKPOINT}", subset, swap_pred, VALUES)


if __name__ == "__main__":
    main()
