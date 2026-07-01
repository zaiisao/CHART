"""Train + evaluate the FIVO/SMC bar-pointer prototype: fixed prior + strong emission + particle filter.

    python -m experiments.smc_prototype.train_smc --device cuda:1

Pipeline:
  1. Load cached Beat-This [T,512] features exactly like the baseline (data/dataset.py: 400 train
     songs seed=1, 40 val songs seed=2, matching train.py's convention).
  2. PRETRAIN the emission head (experiments/smc_prototype/emission.py): a small supervised MLP
     detector, [T,512] -> (p_beat, p_downbeat), trained by BCE against the ground-truth targets. This
     is the "strong, expressive activation" the diagnosis said the earlier (weak-onset-envelope) FIVO
     attempt was missing.
  3. Train the bar-pointer FIVO (experiments/smc_prototype/bar_pointer_smc.py): a FIXED
     (non-audio-conditioned) prior + a geometric-bump observation model compared against the trained
     emission's output, propagated as K particles/song, with systematic resampling on ESS drop. The
     loss is the negative FIVO bound, backpropagated through the reparameterized transition samples.
     The emission head is fine-tuned JOINTLY during this stage (small LR) so its output distribution
     can adapt to what makes the particle filter's Gaussian bump comparison work well, while its
     supervised pretraining (stage 2) is what prevents it collapsing to something uninformative.
  4. Evaluate with the SAME leak-test protocol as evaluate.py (real/shuffle/zero conditions, mir_eval
     beat F-measure @ 0.07s), deployed by the particle filter with NO labels (deploy_smc).
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from config import Config
from data.dataset import load_songs, sample_training_batch

from .bar_pointer_smc import BarPointerFIVO
from .emission import ActivationEmissionHead, pretrain_emission
from .evaluate_smc import evaluate_smc_with_leak_test, print_leak_test


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_feature_dir", default="cache/acts/bt_train_rich")
    parser.add_argument("--val_feature_dir", default="cache/acts/bt_val_rich")
    parser.add_argument("--num_train_songs", type=int, default=400)
    parser.add_argument("--num_val_songs", type=int, default=40)
    parser.add_argument("--num_meters", type=int, default=4)
    parser.add_argument("--beats_per_bar", type=int, default=4)
    # emission pretraining
    parser.add_argument("--emission_pretrain_steps", type=int, default=1500)
    parser.add_argument("--emission_batch_size", type=int, default=16)
    parser.add_argument("--emission_crop_frames", type=int, default=512)
    parser.add_argument("--emission_lr", type=float, default=1e-3)
    # FIVO training
    parser.add_argument("--fivo_steps", type=int, default=1500)
    parser.add_argument("--fivo_crop_frames", type=int, default=512)
    parser.add_argument("--fivo_songs_per_step", type=int, default=8, help="sequences per FIVO step (looped; PF is per-sequence)")
    parser.add_argument("--num_particles_train", type=int, default=128)
    parser.add_argument("--num_particles_eval", type=int, default=300)
    parser.add_argument("--ess_frac", type=float, default=0.5)
    parser.add_argument("--fivo_lr", type=float, default=3e-3)
    parser.add_argument("--emission_finetune_lr", type=float, default=1e-4)
    parser.add_argument("--freeze_emission", action="store_true",
                        help="do NOT fine-tune the emission head during FIVO (diagnostic: FIVO was "
                             "observed to collapse a good pretrained emission head toward a flat, "
                             "uninformative output when fine-tuned jointly -- see MEMORY/report)")
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--readout_mode", default="map", choices=["map", "weighted_mean"])
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_max_frames", type=int, default=1600)
    parser.add_argument("--eval_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--save_path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    device = args.device
    print(f"[smc] device={device}", flush=True)

    config = Config(
        train_feature_dir=args.train_feature_dir, val_feature_dir=args.val_feature_dir,
        num_train_songs=args.num_train_songs, num_val_songs=args.num_val_songs,
        batch_size=args.emission_batch_size, crop_length_frames=args.emission_crop_frames,
        num_meters=args.num_meters, beats_per_bar=args.beats_per_bar, device=device,
    )
    train_songs = load_songs(config.train_feature_dir, config.num_train_songs, seed=1)
    val_songs = load_songs(config.val_feature_dir, config.num_val_songs, seed=2)
    print(f"[smc] train={len(train_songs)} val={len(val_songs)} songs loaded", flush=True)

    feature_dim = train_songs[0].features.shape[1]

    # ---- Stage 1: pretrain the strong emission head ----
    print("[smc] === Stage 1: supervised emission pretraining ===", flush=True)
    emission_head = ActivationEmissionHead(feature_dim).to(device)
    t0 = time.time()
    pretrain_emission(emission_head, train_songs, config, steps=args.emission_pretrain_steps, device=device,
                      lr=args.emission_lr)
    print(f"[smc] emission pretrain done in {time.time() - t0:.0f}s", flush=True)

    # Sanity-check the emission head alone via simple peak-picking (no PF), so we know its ceiling.
    _peak_pick_sanity_check(emission_head, val_songs, device, args.eval_max_frames)

    # ---- Stage 2: FIVO training of the fixed prior + geometric-bump likelihood (+ emission fine-tune) ----
    print("\n[smc] === Stage 2: FIVO training ===", flush=True)
    fivo = BarPointerFIVO(num_meters=args.num_meters, beats_per_bar=args.beats_per_bar).to(device)
    if args.freeze_emission:
        for parameter in emission_head.parameters():
            parameter.requires_grad_(False)
        emission_head.eval()
        optimizer = torch.optim.Adam(fivo.parameters(), lr=args.fivo_lr)
        trainable_emission_params: list = []
    else:
        optimizer = torch.optim.Adam([
            {"params": fivo.parameters(), "lr": args.fivo_lr},
            {"params": emission_head.parameters(), "lr": args.emission_finetune_lr},
        ])
        trainable_emission_params = list(emission_head.parameters())

    t0 = time.time()
    running_bound, running_ess, running_resamples = [], [], []
    for step in range(1, args.fivo_steps + 1):
        features, beat_targets, downbeat_targets = sample_training_batch(
            train_songs, args.fivo_crop_frames, args.fivo_songs_per_step, device)
        optimizer.zero_grad()
        observed_activations = emission_head.probabilities(features)     # [batch, T, 2]
        result = fivo.fivo_bound(observed_activations, num_particles=args.num_particles_train,
                                 ess_frac=args.ess_frac)
        loss = -(result.bound / features.shape[1]).mean()   # per-frame FIVO bound, averaged over the batch
        loss.backward()
        running_ess.append(result.mean_ess_fraction)
        running_resamples.append(result.num_resamples)
        total_norm = torch.nn.utils.clip_grad_norm_(
            list(fivo.parameters()) + trainable_emission_params, args.grad_clip_norm)
        optimizer.step()
        running_bound.append(float((result.bound / features.shape[1]).mean().item()))

        if step % args.eval_every == 0 or step == 1 or step == args.fivo_steps:
            mean_bound = float(np.mean(running_bound[-args.eval_every:]))
            mean_ess = float(np.mean(running_ess[-args.eval_every:]))
            mean_resamples = float(np.mean(running_resamples[-args.eval_every:]))
            elapsed = time.time() - t0
            print(f"  step {step:5d} | fivo_bound/frame {mean_bound:+.4f} | mean_ESS_frac {mean_ess:.3f} "
                 f"| resamples/seq {mean_resamples:.1f} | grad_norm {float(total_norm):.2f} "
                 f"| kappa {float(fivo.prior.phase_kappa):.2f} sigma {float(fivo.prior.log_tempo_sigma):.4f} "
                 f"| {elapsed:.0f}s", flush=True)

        if step % (args.eval_every * 2) == 0 or step == args.fivo_steps:
            print(f"  [smc] mid-training leak-test @ step {step} (val subset, {min(10, len(val_songs))} songs):", flush=True)
            subset = val_songs[:10]
            leak = evaluate_smc_with_leak_test(
                fivo, emission_head, subset, device, args.beats_per_bar,
                num_particles=args.num_particles_eval, ess_frac=args.ess_frac,
                readout_mode=args.readout_mode, max_frames=args.eval_max_frames,
                eval_tolerance_seconds=args.eval_tolerance_seconds)
            print_leak_test(leak)

    print(f"[smc] FIVO training done in {time.time() - t0:.0f}s", flush=True)

    # ---- Final leak-test on the full val set ----
    print("\n[smc] === Final leak-test (full val set, particle-filter deploy, no labels) ===", flush=True)
    leak = evaluate_smc_with_leak_test(
        fivo, emission_head, val_songs, device, args.beats_per_bar,
        num_particles=args.num_particles_eval, ess_frac=args.ess_frac,
        readout_mode=args.readout_mode, max_frames=args.eval_max_frames,
        eval_tolerance_seconds=args.eval_tolerance_seconds)
    print_leak_test(leak)

    if args.save_path:
        torch.save({"fivo": fivo.state_dict(), "emission": emission_head.state_dict(), "args": vars(args)},
                  args.save_path)
        print(f"[smc] saved -> {args.save_path}", flush=True)


@torch.no_grad()
def _peak_pick_sanity_check(emission_head: ActivationEmissionHead, val_songs, device: str, max_frames: int) -> None:
    """Report the emission head's OWN peak-picked F-measure (no dynamics/PF at all) -- the ceiling the
    particle filter's structured read-out is trying to (at least partially) match/exceed via temporal
    consistency, not per-frame accuracy alone."""
    from config import FRAMES_PER_SECOND
    from data.targets import ground_truth_beat_times
    from model import readout

    beat_scores, downbeat_scores = [], []
    for song in val_songs:
        num_frames = min(song.features.shape[0], song.beat_targets.shape[0], max_frames)
        features = song.features[:num_frames].to(device)
        probabilities = emission_head.probabilities(features).cpu().numpy()
        reference_beats = ground_truth_beat_times(song.beat_targets.numpy()[:num_frames], FRAMES_PER_SECOND)
        reference_downbeats = ground_truth_beat_times(song.downbeat_targets.numpy()[:num_frames], FRAMES_PER_SECOND)
        if len(reference_beats) >= 2:
            estimated_beats = readout.peak_pick_times(probabilities[:, 0], FRAMES_PER_SECOND)
            beat_scores.append(readout.f_measure(reference_beats, estimated_beats))
        if len(reference_downbeats) >= 2:
            estimated_downbeats = readout.peak_pick_times(probabilities[:, 1], FRAMES_PER_SECOND)
            downbeat_scores.append(readout.f_measure(reference_downbeats, estimated_downbeats))
    print(f"[smc] emission-head-alone peak-pick ceiling: beat {np.nanmean(beat_scores):.3f} "
         f"downbeat {np.nanmean(downbeat_scores):.3f}  (this is NOT the PF result, just the detector's own ceiling)",
         flush=True)


if __name__ == "__main__":
    main()
