"""Train KalmanVAEBarPointer on the ORIGINAL bt_train_rich pool PLUS local-rubato-augmented variants
(cache/acts/bt_train_rubato, from extract_rubato_cache.py), added not substituted -- isolates the
augmentation's effect for a clean single-variable comparison against kvae_m1_repro_400ep1000.pt.

Every other setting is IDENTICAL to kvae_m1_repro_400ep1000 (1000 steps, batch 16, crop 256, a_dim=z_dim=8,
K=5, same pos_weights/lr/Q_reg) -- see train_kvae.py's main() for that baseline's exact argument defaults,
duplicated here unchanged except for the training pool.

SMC and GTZAN audio are NEVER touched by extract_rubato_cache.py (it only reads from
ballroom/beatles/hains/rwc_popular, the same 4 datasets bt_train_rich itself draws from) -- both remain
fully zero-shot for evaluation exactly as before.

Usage:
    python experiments/kvae_prototype/train_kvae_rubato.py --save_path experiments/kvae_prototype/kvae_rubato.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import load_songs
from model.kalman_vae_bar_pointer import KalmanVAEBarPointer, kvae_elbo
from train_kvae import set_all_seeds, batch_time_major, evaluate_leak_condition_fast, evaluate_with_leak_test_fast

_THIRD_PARTY_KVAE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "third_party", "kalman-vae")
if _THIRD_PARTY_KVAE not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_KVAE)
from kvae.sample_control import SampleControl


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_feature_dir", type=str, default="cache/acts/bt_train_rich")
    parser.add_argument("--rubato_feature_dir", type=str, default="cache/acts/bt_train_rubato")
    parser.add_argument("--val_feature_dir", type=str, default="cache/acts/bt_val_rich")
    parser.add_argument("--num_train_songs", type=int, default=400)      # matches kvae_m1_repro_400ep1000
    parser.add_argument("--num_rubato_songs", type=int, default=800)     # 400 source songs x 2 variants
    parser.add_argument("--num_val_songs", type=int, default=40)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--crop_length_frames", type=int, default=256)
    parser.add_argument("--a_dim", type=int, default=8)
    parser.add_argument("--z_dim", type=int, default=8)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--Q_reg", type=float, default=1e-3)
    parser.add_argument("--beat_loss_weight", type=float, default=5.0)
    parser.add_argument("--beat_pos_weight", type=float, default=8.0)
    parser.add_argument("--downbeat_pos_weight", type=float, default=20.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--eval_max_frames", type=int, default=1600)
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=0)  # matches kvae_m1_repro_400ep1000's seed
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_path", type=str, default="experiments/kvae_prototype/kvae_rubato.pt")
    parser.add_argument("--eval_every", type=int, default=200)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    original_songs = load_songs(args.train_feature_dir, args.num_train_songs, seed=1)      # same seed=1 as baseline
    rubato_songs = load_songs(args.rubato_feature_dir, args.num_rubato_songs, seed=1, min_frames=200)
    train_songs = original_songs + rubato_songs                                            # UNION: added, not replaced
    val_songs = load_songs(args.val_feature_dir, args.num_val_songs, seed=2)

    print(f"[train_kvae_rubato] original={len(original_songs)} rubato={len(rubato_songs)} "
         f"union_train={len(train_songs)} val={len(val_songs)} device={device}", flush=True)
    print(f"[train_kvae_rubato] a_dim={args.a_dim} z_dim={args.z_dim} K={args.K} Q_reg={args.Q_reg} "
         f"steps={args.num_steps} batch={args.batch_size} (all IDENTICAL to kvae_m1_repro_400ep1000 "
         f"except the training pool)", flush=True)

    model = KalmanVAEBarPointer(feature_dim=512, a_dim=args.a_dim, z_dim=args.z_dim, K=args.K, Q_reg=args.Q_reg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_sample_control = SampleControl(encoder="sample", decoder="mean", state_transition="sample", observation="sample")
    positive_weight = torch.tensor([args.beat_pos_weight, args.downbeat_pos_weight], device=device)

    for step in range(1, args.num_steps + 1):
        features, beats, downbeats = batch_time_major(train_songs, args.crop_length_frames, args.batch_size, device)
        elbo, z, info = kvae_elbo(model, features, train_sample_control)
        beat_logits = model.head(z.reshape(-1, model.z_dim)).view(*z.shape[:2], 2)
        beat_bce = F.binary_cross_entropy_with_logits(
            beat_logits, torch.stack([beats, downbeats], dim=-1), pos_weight=positive_weight)
        loss = -elbo + args.beat_loss_weight * beat_bce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.num_steps:
            real = evaluate_leak_condition_fast(model, val_songs, device, "real", args.eval_max_frames,
                                                args.eval_beat_tolerance_seconds)
            print(f"  step {step:5d} | elbo {float(elbo):.1f} | bce {float(beat_bce):.3f} "
                 f"| GEOM beat {real['beat_f']:.3f} db {real['downbeat_f']:.3f}", flush=True)

    leak = evaluate_with_leak_test_fast(model, val_songs, device, args.eval_max_frames, args.eval_beat_tolerance_seconds)
    print("\n[final] Kalman-filter deploy, rubato-augmented training pool:", flush=True)
    for condition in ("real", "shuffle", "zero"):
        print(f"{condition:8s}: beat {leak[condition]['beat_f']:.3f}  downbeat {leak[condition]['downbeat_f']:.3f}", flush=True)

    if args.save_path:
        torch.save({"model": model.state_dict(), "args": vars(args)}, args.save_path)
        print(f"[train_kvae_rubato] saved -> {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
