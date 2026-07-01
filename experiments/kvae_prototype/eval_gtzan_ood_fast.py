"""GTZAN OOD eval using the FAST batched Kalman-filter deploy path -- see fast_deploy.py and
train_kvae.py's evaluate_leak_condition_fast (regression-validated bit-exact against the slow per-song
path on bt_val_rich, and against eval_smc_ood_fresh.py's already-landed SMC numbers 0.654/0.139/0.000
via eval_smc_ood_fast.py, which reproduced 0.6540/0.1392/0.0000).

Same data as eval_gtzan_ood.py (cache/acts/gtzan_rich, extracted by extract_gtzan_cache.py with the fresh
final0-checkpoint 86.1328125fps recipe, byte-compatible with data/dataset.py's Song schema) -- just the
fast evaluation driver instead of the slow one that was killed after ~185 min with no output.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import load_songs
from model.kalman_vae_bar_pointer import KalmanVAEBarPointer
from train_kvae import evaluate_with_leak_test_fast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="experiments/kvae_prototype/kvae_m1_repro_400ep1000.pt")
    parser.add_argument("--feature_dir", type=str, default="cache/acts/gtzan_rich")
    parser.add_argument("--num_songs", type=int, default=0, help="0 = all available songs")
    parser.add_argument("--eval_max_frames", type=int, default=4000)
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--batch_chunk_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    num_songs = args.num_songs if args.num_songs > 0 else 10_000
    songs = load_songs(args.feature_dir, num_songs, seed=2, min_frames=400, min_beats=8)
    print(f"[eval_gtzan_ood_fast] loaded {len(songs)} GTZAN songs from {args.feature_dir} "
         f"(never in bt_train_rich/bt_val_rich's training mix)", flush=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = KalmanVAEBarPointer(
        feature_dim=512, a_dim=ckpt_args.get("a_dim", 8), z_dim=ckpt_args.get("z_dim", 8), K=ckpt_args.get("K", 5),
        Q_reg=ckpt_args.get("Q_reg", 1e-3),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"[eval_gtzan_ood_fast] loaded {args.checkpoint} (a_dim={model.a_dim} z_dim={model.z_dim} K={model.ssm.K})", flush=True)

    leak = evaluate_with_leak_test_fast(model, songs, device, args.eval_max_frames,
                                        args.eval_beat_tolerance_seconds, args.batch_chunk_size)
    print("\n--- GTZAN OOD leak test: fresh final0 features, FAST batched Kalman-filter deploy ---", flush=True)
    for condition in ("real", "shuffle", "zero"):
        print(f"{condition:8s}: beat {leak[condition]['beat_f']:.3f}  downbeat {leak[condition]['downbeat_f']:.3f}", flush=True)
    print("(real high + shuffle/zero collapsed => genuinely OOD-audio-driven, not an in-domain-only template)", flush=True)


if __name__ == "__main__":
    main()
