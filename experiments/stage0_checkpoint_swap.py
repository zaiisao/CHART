"""Is the tf512 gtzan collapse caused by the CHECKPOINT the features came from?

Standing suspicion: gtzan is the only split whose features come from `final0` rather than
a fold checkpoint. Beat This's 512-dim penultimate basis is not pinned across checkpoints
(nothing in training ties unit 37 of fold3 to unit 37 of final0), so a model that reads
raw features can be handed a systematically rotated input at test time, while the 2-dim
activation channels keep their meaning across checkpoints by construction.

That is a hypothesis about the FEATURES, and the gtzan comparison cannot test it, because
gtzan differs from the CV folds in two ways at once (different checkpoint AND different
songs). This probe removes the second difference:

    same songs, same labels, same trained model, features recomputed through final0.

Each fold's held-out crops are scored twice -- once through their fold-honest checkpoint
(the baseline), once through final0 -- while the model, the training data and the ground
truth are held fixed. A drop between the two columns is caused by the checkpoint swap and
by nothing else.

An earlier version of this file pre-registered "tf512 is predicted to drop and autocorr
not", on the grounds that the autocorr head is basis-immune. THAT IS FALSE and the run
falsified it: AutocorrHead's first layer is a LEARNED Linear(512 -> channels) fitted on
one checkpoint's basis (vbpm/heads.py), so it reads the 512 basis exactly as the
transformer does, and it dropped just as far (-0.162 against -0.188). There is therefore
NO basis-immune 512-reader in this design, and the drop of the 512 arms cannot be
attributed to basis rotation by this experiment alone -- see the caveat below.

NOTE the final0 column is deliberately NOT fold-honest (final0 trained on these songs), so
it is a diagnostic about feature geometry and is never a fold-honest score.

WHAT THIS EXPERIMENT CAN AND CANNOT CONCLUDE. The leak biases the swap in its FAVOUR, and
the measured size of that favourable bias is large: the pinned 2-channel arm GAINS +0.168.
So the leak is the same order as the -0.16..-0.21 the 512 arms lose, and the drop cannot
be attributed to basis misalignment quantitatively. What the result does support is the
weaker, still useful claim: 512-dim readers are checkpoint-FRAGILE and 2-channel readers
are not. Separating "rotated basis" from "final0's representation simply differs" needs a
swap between two FOLD checkpoints that BOTH held the song out (score fold-3's held-out
crops through fold-5), which removes both the leak and the extra-training-data confound.

Run it as a module (the package layout, not a loose script), after warming both feature
passes -- the fold-honest one and the swapped one:

    python -m vbpm.warm_cache --gpus 0 1 3 --output features+activations
    python -m vbpm.warm_cache --gpus 0 1 3 --output features+activations --override final0
    CUDA_VISIBLE_DEVICES=0 python -m experiments.stage0_checkpoint_swap --folds 0 1
    ...                                                                 (one shard per GPU)
    python -m experiments.stage0_checkpoint_swap --aggregate
"""
import argparse
import pathlib
import pickle

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


def aggregate(shard_dir, arms):
    """Pool the shards' held-out predictions and report the two columns per arm."""
    shards = sorted(pathlib.Path(shard_dir).glob("shard_*.pkl"))
    assert shards, f"no shards under {shard_dir}"
    merged: dict = {}
    seen: dict = {}
    for path in shards:
        with open(path, "rb") as fh:
            for arm, (true, base, swap) in pickle.load(fh).items():
                # a stale shard from an earlier run would pool in silently and double-count
                # whole folds, so every (arm, shard) contribution has to be new
                assert (arm, path.name) not in seen, f"duplicate shard {path.name} for {arm}"
                seen[(arm, path.name)] = len(true)
                entry = merged.setdefault(arm, ([], [], []))
                entry[0].extend(true)
                entry[1].extend(base)
                entry[2].extend(swap)
    sizes = {arm: sum(n for (a, _p), n in seen.items() if a == arm) for arm in merged}
    assert len(set(sizes.values())) == 1, \
        f"arms pooled over different crop counts -- the shards do not partition: {sizes}"

    for arm in arms:
        if arm not in merged:
            continue
        true, base_pred, swap_pred = merged[arm]
        base = balanced_accuracy(true, base_pred, VALUES)
        swap = balanced_accuracy(true, swap_pred, VALUES)
        agree = float(np.mean(np.asarray(base_pred) == np.asarray(swap_pred)))
        subset = [{"m_true": t} for t in true]
        print(f"\n######## arm: {arm} ########  (n={len(true)})")
        print(f"  fold-honest checkpoint : balanced={base:.3f}")
        print(f"  {SWAP_CHECKPOINT} features       : balanced={swap:.3f}   "
              f"(delta {swap - base:+.3f}, predictions agree on {agree:.3f})")
        score(f"{arm}/baseline", subset, base_pred, VALUES)
        score(f"{arm}/{SWAP_CHECKPOINT}", subset, swap_pred, VALUES)


def main():
    """Baseline vs checkpoint-swapped scores for each arm, on identical crops."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-fold", type=int, default=None)
    ap.add_argument("--arms", nargs="*", default=PROBE_ARMS)
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="this shard's held-out folds; omit for all")
    ap.add_argument("--shard-dir", default="/disk4/jaehoon/vbpm_swap")
    ap.add_argument("--aggregate", action="store_true",
                    help="pool the shards written by earlier runs and report")
    args = ap.parse_args()
    device = "cuda"

    if args.aggregate:
        aggregate(args.shard_dir, args.arms)
        return

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

    folds = sorted({c["fold"] for c in cv})
    if args.folds is not None:
        folds = [f for f in folds if f in args.folds]
    results: dict = {}

    for arm in args.arms:
        base_true, base_pred, swap_pred = [], [], []
        for fold in folds:
            train = [c for c in cv if c["fold"] != fold]
            held = [c for c in cv if c["fold"] == fold]
            fitted = fit(arm, train, device)
            # the standardizer is part of the fitted model and is NOT refitted: the swap
            # must be the only thing that changes between the two columns
            base_pred += predict(arm, fitted, held, device)
            swap_pred += predict(arm, fitted, [swapped[id(c)] for c in held], device)
            base_true += [c["m_true"] for c in held]
            print(f"  [{arm}] fold {fold} done", flush=True)

        results[arm] = (base_true, base_pred, swap_pred)
        agree = float(np.mean(np.asarray(base_pred) == np.asarray(swap_pred)))
        print(f"  [{arm}] shard folds {folds}: "
              f"baseline={balanced_accuracy(base_true, base_pred, VALUES):.3f}  "
              f"{SWAP_CHECKPOINT}={balanced_accuracy(base_true, swap_pred, VALUES):.3f}  "
              f"agree={agree:.3f}", flush=True)

    shard_dir = pathlib.Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    # the tag names the arms as well as the folds: sharding by ARM would otherwise have
    # every shard write the same file, each silently destroying the last
    tag = ("all" if args.folds is None else "-".join(str(f) for f in folds)
           ) + "__" + "-".join(sorted(args.arms))
    with open(shard_dir / f"shard_{tag}.pkl", "wb") as fh:
        pickle.dump(results, fh)


if __name__ == "__main__":
    main()
