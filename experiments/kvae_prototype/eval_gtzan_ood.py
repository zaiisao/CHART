"""Genuine OOD check on GTZAN (never in bt_train_rich/bt_val_rich's training mix).

bt_train_rich/bt_val_rich were built from --dataset_include ballroom,beatles,hains,rwc_popular (see the
archived run_rich.sh) -- gtzan was never in that list, so this is a real held-out generalization check,
complementary to the SMC check (which is beats-only; GTZAN has real downbeat labels too).

Unlike SMC, extract_gtzan_cache.py wrote GTZAN into the SAME .pt schema as bt_train_rich (activations/
beat_targets/downbeat_targets), using the SAME fresh "final0"-checkpoint 86.1328125fps extraction recipe
as eval_smc_ood_fresh.py -- so this script reuses data/dataset.py's load_songs VERBATIM, no custom loader.

Deploy path: identical to train_kvae.py's evaluate_leak_condition (encoder(h) mean -> causal deterministic
Kalman filter -> head(filtered z) -> sigmoid -> peak-pick). Same leak-test protocol (real/shuffle/zero),
beat AND downbeat F-measure (GTZAN has both), same 0.07s tolerance.

Usage:
    python experiments/kvae_prototype/eval_gtzan_ood.py --checkpoint experiments/kvae_prototype/kvae_m1_repro_400ep1000.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FRAMES_PER_SECOND
from data.dataset import load_songs
from model.kalman_vae_bar_pointer import KalmanVAEBarPointer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_kvae import evaluate_with_leak_test  # reuse the SAME leak-test driver train_kvae.py uses


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="experiments/kvae_prototype/kvae_m1_repro_400ep1000.pt")
    parser.add_argument("--feature_dir", type=str, default="cache/acts/gtzan_rich")
    parser.add_argument("--num_songs", type=int, default=0, help="0 = all available songs")
    parser.add_argument("--eval_max_frames", type=int, default=4000, help="GTZAN clips are ~30s (~2585 frames); "
                        "raised above bt_val_rich's 1600-frame default so no GTZAN clip is truncated")
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    num_songs = args.num_songs if args.num_songs > 0 else 10_000  # load_songs caps at what's on disk anyway
    songs = load_songs(args.feature_dir, num_songs, seed=2, min_frames=400, min_beats=8)
    print(f"[eval_gtzan_ood] loaded {len(songs)} GTZAN songs from {args.feature_dir} "
         f"(never in bt_train_rich/bt_val_rich's training mix)", flush=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = KalmanVAEBarPointer(
        feature_dim=512, a_dim=ckpt_args.get("a_dim", 8), z_dim=ckpt_args.get("z_dim", 8), K=ckpt_args.get("K", 5),
        Q_reg=ckpt_args.get("Q_reg", 1e-3),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[eval_gtzan_ood] loaded {args.checkpoint} (a_dim={model.a_dim} z_dim={model.z_dim} K={model.ssm.K})", flush=True)

    leak = evaluate_with_leak_test(model, songs, device, args.eval_max_frames, args.eval_beat_tolerance_seconds)
    print("\n--- GTZAN OOD leak test: fresh final0 features, exact Kalman-filter deploy ---", flush=True)
    for condition in ("real", "shuffle", "zero"):
        print(f"{condition:8s}: beat {leak[condition]['beat_f']:.3f}  downbeat {leak[condition]['downbeat_f']:.3f}", flush=True)
    print("(real high + shuffle/zero collapsed => genuinely OOD-audio-driven, not an in-domain-only template)", flush=True)


if __name__ == "__main__":
    main()
