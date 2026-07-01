"""Q_reg generalization sweep: does loosening the Kalman-VAE process-noise floor help zero-shot SMC?

Motivation (from this project's own history -- see memory project_smc_mirex_ood_generalization.md and the
"deployment cap" / "VAE re-diagnosis" notes): the historical worry is a model that looks great in-domain
(same 4-dataset mix as training) but collapses zero-shot on SMC (Holzapfel), whose tempo/phrasing is far
more rubato/non-mechanical than ballroom/beatles/hains/rwc_popular. StateSpaceModel's process-noise
covariance mat_Q = L L^T + Q_reg*I (see third_party/kalman-vae/kvae/state_space_model.py) has an additive
floor Q_reg (vendored default 1e-3) below which the LEARNED noise cannot go -- but the learned L L^T part
CAN make mat_Q arbitrarily small (tight dynamics prior, confident/rigid tempo tracking) if that is what
minimizes the training ELBO on the training mix's more metronomic timing. A tight/rigid dynamics prior is
exactly what would be expected to fail on SMC's rubato: the filter's prediction step pins z_t close to
A@z_{t-1} and has little room to absorb an actual tempo fluctuation the training distribution never
exercised. RAISING Q_reg forces mat_Q to stay at least this large regardless of what L learns, loosening
that grip -- more the filter can attribute to "the tempo genuinely changed here" rather than "the audio
observation is noisy, trust the dynamics prediction over it."

This sweep trains KalmanVAEBarPointer at several Q_reg floors (holding EVERY OTHER setting identical to
the already-reported kvae_m1_repro_400ep1000 checkpoint: 400 train songs / 1000 steps / batch 16 / crop
256 / a_dim=z_dim=8 / K=5 / beat_pos_weight 8,20 / lr 1e-3 -- the SAME unchanged bt_train_rich 4-dataset
mix, no new data), then evaluates EACH checkpoint on both:
  (a) in-domain bt_val_rich (40 songs, real/shuffle/zero) -- the number this project already tracks.
  (b) zero-shot SMC (217 songs, real/shuffle/zero, beat-only) -- via eval_smc_ood_fast's already-cached
      fresh final0 features (cache/acts/smc_fresh_final0) and the fast batched deploy path (fast_deploy.py,
      regression-validated to match the slow path bit-for-bit on bt_val_rich and 4-decimal on SMC).

Reports a table of (Q_reg, in-domain real beat/db, SMC real beat) so the delta in/out-of-domain gap can be
read off directly -- the quantity of interest is NOT which Q_reg gets the best in-domain number (that's
expected to be roughly flat/slightly down as Q_reg rises, since it's a floor pulling AWAY from whatever
the ELBO's unconstrained optimum was) but whether the in-domain-to-SMC GAP shrinks as Q_reg rises.

Usage:
    python experiments/kvae_prototype/qreg_sweep.py --q_reg_values 0.001 0.01 0.05 0.1
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import load_songs
from model.kalman_vae_bar_pointer import KalmanVAEBarPointer, kvae_elbo
from train_kvae import (
    set_all_seeds, batch_time_major, evaluate_leak_condition_fast, evaluate_with_leak_test_fast,
)
from eval_smc_ood_fresh import build_smc_index, load_smc_songs, FEATURE_CACHE_DIR as SMC_FEATURE_CACHE_DIR
from eval_smc_ood_fast import evaluate_smc_condition_fast

_THIRD_PARTY_KVAE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "third_party", "kalman-vae")
if _THIRD_PARTY_KVAE not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_KVAE)
from kvae.sample_control import SampleControl


def train_one_variant(q_reg: float, train_songs, val_songs, args, device: str) -> KalmanVAEBarPointer:
    """Trains one Q_reg variant with IDENTICAL settings to kvae_m1_repro_400ep1000, only Q_reg differs."""
    set_all_seeds(args.seed)
    model = KalmanVAEBarPointer(feature_dim=512, a_dim=args.a_dim, z_dim=args.z_dim, K=args.K, Q_reg=q_reg).to(device)
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
            mat_Q_trace = float(model.ssm.mat_Q.trace())
            print(f"  [Q_reg={q_reg}] step {step:5d} | bce {float(beat_bce):.3f} | "
                 f"mat_Q trace {mat_Q_trace:.3f} | GEOM beat {real['beat_f']:.3f} db {real['downbeat_f']:.3f}", flush=True)

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--q_reg_values", type=float, nargs="+", default=[1e-3, 1e-2, 5e-2, 1e-1])
    parser.add_argument("--train_feature_dir", type=str, default="cache/acts/bt_train_rich")
    parser.add_argument("--val_feature_dir", type=str, default="cache/acts/bt_val_rich")
    parser.add_argument("--num_train_songs", type=int, default=400)
    parser.add_argument("--num_val_songs", type=int, default=40)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--crop_length_frames", type=int, default=256)
    parser.add_argument("--a_dim", type=int, default=8)
    parser.add_argument("--z_dim", type=int, default=8)
    parser.add_argument("--K", type=int, default=5)
    # See train_kvae.py's identical defaults for provenance: inherited unchanged from the historical M1 run.
    parser.add_argument("--beat_loss_weight", type=float, default=5.0)
    parser.add_argument("--beat_pos_weight", type=float, default=8.0)
    parser.add_argument("--downbeat_pos_weight", type=float, default=20.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--eval_max_frames", type=int, default=1600)
    parser.add_argument("--eval_beat_tolerance_seconds", type=float, default=0.07)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="experiments/kvae_prototype")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    train_songs = load_songs(args.train_feature_dir, args.num_train_songs, seed=1)
    val_songs = load_songs(args.val_feature_dir, args.num_val_songs, seed=2)
    print(f"[qreg_sweep] train={len(train_songs)} val={len(val_songs)} device={device} "
         f"Q_reg values = {args.q_reg_values}", flush=True)

    smc_index = build_smc_index()
    smc_songs = load_smc_songs(smc_index, SMC_FEATURE_CACHE_DIR)
    print(f"[qreg_sweep] SMC zero-shot set: {len(smc_songs)} songs (fresh final0 features, cached)", flush=True)

    results = []
    for q_reg in args.q_reg_values:
        print(f"\n===== Q_reg = {q_reg} =====", flush=True)
        model = train_one_variant(q_reg, train_songs, val_songs, args, device)

        save_path = f"{args.save_dir}/kvae_qreg_{q_reg:.0e}.pt".replace("+", "")
        torch.save({"model": model.state_dict(), "args": {**vars(args), "Q_reg": q_reg}}, save_path)

        in_domain = evaluate_with_leak_test_fast(model, val_songs, device, args.eval_max_frames,
                                                  args.eval_beat_tolerance_seconds)
        smc = {
            condition: evaluate_smc_condition_fast(model, smc_songs, device, condition,
                                                    args.eval_beat_tolerance_seconds)
            for condition in ("real", "shuffle", "zero")
        }
        mat_Q_trace = float(model.ssm.mat_Q.trace())
        mat_R_trace = float(model.ssm.mat_R.trace())

        print(f"[Q_reg={q_reg}] FINAL mat_Q trace={mat_Q_trace:.3f} mat_R trace={mat_R_trace:.3f}", flush=True)
        print(f"  in-domain : real beat {in_domain['real']['beat_f']:.3f} db {in_domain['real']['downbeat_f']:.3f} "
             f"| shuffle beat {in_domain['shuffle']['beat_f']:.3f} db {in_domain['shuffle']['downbeat_f']:.3f} "
             f"| zero beat {in_domain['zero']['beat_f']:.3f} db {in_domain['zero']['downbeat_f']:.3f}", flush=True)
        print(f"  SMC (OOD) : real beat {smc['real']['beat_f']:.3f} "
             f"| shuffle beat {smc['shuffle']['beat_f']:.3f} | zero beat {smc['zero']['beat_f']:.3f}", flush=True)
        gap = in_domain["real"]["beat_f"] - smc["real"]["beat_f"]
        print(f"  in-domain-to-SMC GAP (real beat F): {gap:.3f}", flush=True)

        results.append({
            "q_reg": q_reg, "mat_Q_trace": mat_Q_trace, "mat_R_trace": mat_R_trace,
            "in_domain_real_beat": in_domain["real"]["beat_f"], "in_domain_real_db": in_domain["real"]["downbeat_f"],
            "in_domain_shuffle_beat": in_domain["shuffle"]["beat_f"], "in_domain_zero_beat": in_domain["zero"]["beat_f"],
            "smc_real_beat": smc["real"]["beat_f"], "smc_shuffle_beat": smc["shuffle"]["beat_f"],
            "smc_zero_beat": smc["zero"]["beat_f"], "gap": gap,
        })

    print("\n\n===== Q_reg SWEEP SUMMARY =====", flush=True)
    header = f"{'Q_reg':>10} {'matQ_tr':>8} {'ID_real_b':>10} {'ID_real_db':>11} {'SMC_real_b':>11} {'GAP':>8} {'ID_zero_b':>10} {'SMC_zero_b':>11}"
    print(header, flush=True)
    for r in results:
        print(f"{r['q_reg']:>10.4f} {r['mat_Q_trace']:>8.3f} {r['in_domain_real_beat']:>10.3f} "
             f"{r['in_domain_real_db']:>11.3f} {r['smc_real_beat']:>11.3f} {r['gap']:>8.3f} "
             f"{r['in_domain_zero_beat']:>10.3f} {r['smc_zero_beat']:>11.3f}", flush=True)


if __name__ == "__main__":
    main()
