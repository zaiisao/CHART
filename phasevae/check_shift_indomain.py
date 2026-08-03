"""Control 6 on an IN-DOMAIN dataset, from a saved checkpoint.

The in-run shift test uses gtzan, which turned out to be out of domain for this model
(near-chance offset accuracy there). A control run on data the model cannot do at all
tells you nothing about whether the mechanism works, so it is repeated on a held-out
fold of a dataset the model does handle.

Cut the same audio one beat later: r_true moves by -1 (mod m), so the model's predicted
offset must move by exactly -1 too. The distribution over the observed move is printed
in full -- a spike at 0 means a degenerate constant prediction, not a tracking model.

    PYTHONPATH=. python -m phasevae.check_shift_indomain --gpu 1
        --checkpoint checkpoint_seed0_anchor.pt --anchor-init --dataset ballroom
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch

from vbpm.data import derive_y, iter_frontend_features

from .crops import CROP_BARS, build_crop, crop_starts, song_bar_length
from .model import PhaseVAE
from .run import collate

M = 4
VAL_FOLD = 7


def main() -> None:
    """Run the one-beat-later re-cut on held-out songs of one dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1, choices=(1, 3))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchor-init", action="store_true")
    parser.add_argument("--dataset", default="ballroom")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    model = None
    predicted, truth, correct = Counter(), Counter(), []
    rng = np.random.default_rng(11)
    for song, features in iter_frontend_features(output="features+activations",
                                                 datasets=[args.dataset],
                                                 folds=[VAL_FOLD], verbose=False):
        beat_times, downbeat_times = song.beats()
        if len(downbeat_times) < 3:
            continue
        y_song, _ = derive_y(beat_times, downbeat_times)
        if song_bar_length(y_song) != M:
            continue
        starts, _ = crop_starts(y_song, M, CROP_BARS)
        pairs = [(s, r) for s, r in starts if (s + 1, (r - 1) % M) in set(starts)]
        if not pairs:
            continue
        start, r_true = pairs[int(rng.integers(len(pairs)))]
        crops = [build_crop(features, beat_times, start, r_true, M),
                 build_crop(features, beat_times, start + 1, (r_true - 1) % M, M)]
        if any(c is None for c in crops):
            continue
        if model is None:
            model = PhaseVAE(crops[0]["h"].shape[1],
                             anchor_init=args.anchor_init).to(device)
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            model.eval()
        truth[(crops[0]["r_true"] - crops[1]["r_true"]) % M] += 1
        hats = []
        with torch.no_grad():
            for crop in crops:
                batch = collate([crop], device)
                scores, _ = model.deploy_offset_scores(
                    batch["h"], batch["delta"], batch["mask"], batch["beat_frames"], M)
                hats.append(int(scores.argmax()))
        predicted[(hats[0] - hats[1]) % M] += 1
        correct.append(float(hats[0] == crops[0]["r_true"]))
        if len(correct) >= args.limit:
            break

    print(f"dataset {args.dataset}, fold {VAL_FOLD}, n = {len(correct)}")
    print(f"  true move in r        : {dict(truth)}   (must be all 1)")
    print(f"  predicted move in r   : {dict(predicted)}  (1 = tracks, "
          f"0 = constant prediction)")
    print(f"  tracks correctly      : "
          f"{predicted[1] / max(sum(predicted.values()), 1):.3f}")
    print(f"  offset accuracy here  : {np.mean(correct):.3f} (chance {1 / M:.3f})")


if __name__ == "__main__":
    main()
