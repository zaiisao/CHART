"""Is asap's failure about Beat This, or about piano audio?

asap is our worst corpus (balanced 0.359 with the reducer, 0.489 with the best trained head,
against a 0.333 floor) while the synthetic-h control reaches 0.988 on the SAME crops. So the
failure sits in the evidence path, not downstream of it. Two explanations survive that
control, and it cannot separate them, because its bumps are placed at the annotated
downbeats -- the very thing m_true is derived from:

    (1) Beat This's downbeat channel is weak on solo piano (no percussive onsets), or
    (2) the annotation's metrical level disagrees with what the audio expresses (classical
        notation puts bar lines where a listener may hear a different grouping).

A SECOND, INDEPENDENT frontend separates them. If Beat Transformer fails on asap the same
way, the deafness is not one checkpoint's quirk and (2) moves to the front; if it does
markedly better, (1) leads and the fix is a frontend, not a latent.

Beat Transformer is the right second opinion here: its 8 folds are over ballroom /
hainsworth / carnatic / harmonix / smc, so asap is in NONE of its training data and every
checkpoint is leak-free on it -- the same free pass gtzan gets.

Controlled comparison. Beat Transformer emits only [T, 2] activations, so the arms are
restricted to the ones that read two channels; the rich-feature arms have no counterpart
and are deliberately absent. Both frontends are trained asap-ONLY on the same song folds,
so neither borrows strength from the other corpora. The all-corpora Beat This numbers are
quoted for context but are NOT the comparison -- they were fitted on 16,925 crops, not
9,742, and mixing the two would confound frontend with training set.

Run (features cache, then the arms):
    CUDA_VISIBLE_DEVICES=0 python -m experiments.stage0_beat_transformer --warm --folds 0 1 2
    ...                                                                  (one shard per GPU)
    python -m experiments.stage0_beat_transformer
"""
import argparse
import pathlib

import numpy as np
import soundfile
import torch

from data.songs import iter_songs
from experiments.stage0_transformer_prior import predict_with, standardizer, train_arm
from vbpm.data import FPS, VALUES, extract_crops, slice_h
from vbpm.fitting import cv_out_of_fold, emission_counts, score
from vbpm.heads import AutocorrHead, TransformerPrior
from vbpm.reducers import peak_summary

DATASET = "asap"
BT_CHECKPOINT = "fold_0"          # asap is outside every Beat Transformer fold (see above)
CACHE_DIR = "/disk4/jaehoon/vbpm_bt_cache"

# arm name -> (builder, field, width, normalise); only [T, 2] readers -- see module docstring
ARMS = {
    "linear": (lambda: torch.nn.Linear(10, len(VALUES)), "s10", None, False),
    "autocorr2": (lambda: AutocorrHead(in_dim=2), "h2", 2, False),
    "tf2": (lambda: TransformerPrior(in_dim=2), "h2", 2, False),
}


def beat_transformer_features(songs, cache_dir=CACHE_DIR, device="cuda"):
    """Yield (song, [T, 2] activations) through Beat Transformer, memoized on disk."""
    from frontends.beat_transformer import BeatTransformerFrontend

    group = pathlib.Path(cache_dir) / BT_CHECKPOINT
    group.mkdir(parents=True, exist_ok=True)
    frontend = None
    for song in songs:
        path = group / f"{song.stem}.npy"
        if path.exists():
            yield song, np.load(path)
            continue
        if frontend is None:                      # lazy: a warm cache needs no model at all
            frontend = BeatTransformerFrontend(checkpoint=BT_CHECKPOINT, device=device,
                                               target_fps=FPS)
            assert abs(frontend.fps - FPS) < 1e-9, "frontend fps must match vbpm.data.FPS"
        signal, sample_rate = soundfile.read(str(song.audio_path), dtype="float32")
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        features = frontend.get_features(signal, sample_rate).numpy().astype(np.float32)
        np.save(path, features)
        yield song, features


def beat_this_features(songs, device="cuda"):
    """The same songs through Beat This, fold-honestly, as [T, 2] activations.

    Read off the cached features+activations pass (last two channels ARE the activation
    head's output), so this costs nothing beyond a disk read.
    """
    from vbpm.data import iter_frontend_features

    wanted = {s.stem for s in songs}
    for song, h in iter_frontend_features(datasets=[DATASET], output="features+activations",
                                          device=device, verbose=False):
        if song.stem in wanted:
            yield song, h[:, -2:]


def build_crops(pairs):
    """(song, h) pairs -> the crop entries the arms consume."""
    crops = []
    for song, h in pairs:
        beat_times, downbeat_times = song.beats()
        song_crops, _ = extract_crops(beat_times, downbeat_times, VALUES)
        for crop in song_crops:
            h_crop, _t0 = slice_h(h, crop["beats"])
            acts = np.asarray(h_crop, dtype=np.float32)
            counts, mask = emission_counts(crop["y"], VALUES)
            crops.append({"h2": acts.astype(np.float16),
                          "s10": peak_summary(acts).numpy().astype(np.float32),
                          "C": counts.astype(np.float32), "mask": mask.astype(np.float32),
                          "m_true": crop["m_true"], "dataset": song.dataset,
                          "fold": song.fold})
    return crops


def run_arms(crops, tag, device="cuda"):
    """Fold-honest CV over the asap song folds, one line per arm."""
    import experiments.stage0_transformer_prior as campaign
    saved, saved_field = campaign.ARMS, campaign.BUCKET_FIELD
    campaign.ARMS = ARMS                       # the arms are the [T, 2] subset here
    campaign.BUCKET_FIELD = "h2"               # no h512 exists in this campaign
    try:
        for arm in ARMS:
            pooled, preds, _ = cv_out_of_fold(
                crops, [], lambda cs, a=arm: (train_arm(a, cs, standardizer(cs, device),
                                                        device), standardizer(cs, device)),
                lambda fitted, cs, a=arm: predict_with(a, fitted[0], cs, fitted[1], device),
                verbose=False)
            score(f"{tag}/{arm}", pooled, preds, VALUES)
    finally:
        campaign.ARMS, campaign.BUCKET_FIELD = saved, saved_field


def main():
    """Warm the Beat Transformer cache, or run both frontends' arms on asap."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", action="store_true", help="only compute+cache BT features")
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="shard by song fold")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    songs = [s for s in iter_songs(datasets=[DATASET])]
    if args.folds is not None:
        songs = [s for s in songs if s.fold in args.folds]
    if args.limit:
        songs = songs[:args.limit]
    print(f"{DATASET}: {len(songs)} songs", flush=True)

    if args.warm:
        for n, (song, h) in enumerate(beat_transformer_features(songs), 1):
            if n % 25 == 0 or n == 1:
                print(f"  {n}/{len(songs)}  {song.stem}  {h.shape}", flush=True)
        print("beat transformer cache warm.", flush=True)
        return

    bt = build_crops(beat_transformer_features(songs))
    bthis = build_crops(beat_this_features(songs))
    print(f"crops: beat_transformer {len(bt)}  beat_this {len(bthis)}", flush=True)
    assert len(bt) == len(bthis), "the two frontends must yield the same crops"

    run_arms(bthis, "beat_this")
    run_arms(bt, "beat_transformer")


if __name__ == "__main__":
    main()
