"""Train the reality-adjusted VBPM END-TO-END FROM RANDOM WEIGHTS on real audio.

Same strict ELBO (beta=1, no free-bits / anneal / supervision) as faithful/, but on the
reality-adjusted generative model (Student-t tempo level + AR(1) deviation, wrapped-Cauchy
phase, bar-gated meter, beat+downbeat emission). faithful/ remains the A/B baseline.

Usage (chart env):
  python -m vbpm.train --data_root /home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data \
      --datasets ballroom,beatles,hains,rwc_popular --frames 256 --batch_size 16 \
      --steps 2000 --eval_every 200 --max_eval_songs 12 --out runs/vbpm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import FPS, N_MELS, LogMel, build_train_loader, iter_val_songs
from .elbo import strict_elbo
from .evaluate import evaluate
from .model import BarPointerVAE


def parse_args():
    p = argparse.ArgumentParser(description="Reality-adjusted VBPM (end-to-end, random init)")
    p.add_argument("--data_root", required=True)
    p.add_argument("--datasets", default="ballroom,beatles,hains,rwc_popular")
    p.add_argument("--frames", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num_meters", type=int, default=4)
    p.add_argument("--examples_per_epoch", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--max_eval_songs", type=int, default=12)
    p.add_argument("--max_eval_frames", type=int, default=4000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--decoder_reads_h", action="store_true",
                   help="COLLAPSE ABLATION: let the likelihood read h (default: latent-only emission)")
    p.add_argument("--out", default="runs/vbpm")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 78)
    print("REALITY-ADJUSTED VBPM  (end-to-end from random weights)")
    print(f"  likelihood latent-only: {not args.decoder_reads_h} | Student-t tempo level + AR(1) dev |")
    print("  wrapped-Cauchy phase | bar-gated meter | beat+downbeat emission | beta=1 strict ELBO")
    print(f"  datasets={keys} frames={args.frames} batch={args.batch_size} steps={args.steps} lr={args.lr}")
    print("=" * 78, flush=True)

    logmel = LogMel().to(device)
    model = BarPointerVAE(h_dim=N_MELS, hidden=args.hidden, num_meters=args.num_meters,
                          latent_only=not args.decoder_reads_h).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = build_train_loader(args.data_root, keys, args.frames, args.batch_size,
                                examples_per_epoch=args.examples_per_epoch,
                                num_workers=args.num_workers, seed=args.seed)
    print("materialising val songs ...", flush=True)
    val_songs = list(iter_val_songs(args.data_root, keys, max_per_dataset=args.max_eval_songs, seed=args.seed))
    print(f"  {len(val_songs)} val songs", flush=True)

    mf = (out / "metrics.jsonl").open("w")
    best_pw = -1.0; step = 0; t0 = time.time()
    model.train()
    data_iter = iter(loader)
    while step < args.steps:
        try:
            audio, beats, dbs = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); audio, beats, dbs = next(data_iter)
        step += 1
        temperature = 1.0 + (0.3 - 1.0) * min(step / max(args.steps, 1), 1.0)

        audio = audio.to(device)
        h = logmel(audio)[:, :args.frames]
        beats = beats[:, :args.frames].to(device)
        dbs = dbs[:, :args.frames].to(device)
        T = min(h.shape[1], beats.shape[1], dbs.shape[1])
        h, beats, dbs = h[:, :T], beats[:, :T], dbs[:, :T]

        opt.zero_grad()
        loss, info = strict_elbo(model, h, beats, dbs, temperature=temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % 20 == 0 or step == 1:
            sps = step / (time.time() - t0)
            print(f"step {step:5d} | L={info['loss']:8.2f} rec={info['recon']:7.2f}"
                  f"(b={info['recon_beat']:.1f} db={info['recon_db']:.1f}) "
                  f"KL={info['kl']:6.3f}(m={info['kl_meter']:.3f} phi={info['kl_phase']:.3f} "
                  f"lv={info['kl_level']:.3f} dv={info['kl_dev']:.3f}) "
                  f"nu={info['tempo_dof']:.2f} xr={info['n_cross']:.1f} | {sps:.2f} it/s", flush=True)
            rec = {"step": step, "phase": "train", **{k: info[k] for k in
                   ("loss", "recon", "recon_beat", "recon_db", "kl", "kl_meter", "kl_phase",
                    "kl_level", "kl_dev", "n_cross", "tempo_dof")}, "temp": temperature}
            mf.write(json.dumps(rec) + "\n"); mf.flush()

        if step % args.eval_every == 0 or step == args.steps:
            summary, _ = evaluate(model, logmel, val_songs, device, fps=FPS,
                                  max_frames=args.max_eval_frames)
            print(f"  [eval @ {step}] beat_phase_F={summary['beat_phase']:.3f} "
                  f"downbeat_phase_F={summary['downbeat_phase']:.3f} decoder_F={summary['decoder']:.3f} "
                  f"db_emit_F={summary['downbeat_emit']:.3f} metronome_F={summary['metronome']:.3f} "
                  f"(n={summary['n_songs']})", flush=True)
            mf.write(json.dumps({"step": step, "phase": "eval", **summary}) + "\n"); mf.flush()
            torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, out / "final.pt")
            if summary["beat_phase"] == summary["beat_phase"] and summary["beat_phase"] > best_pw:
                best_pw = summary["beat_phase"]
                torch.save({"model": model.state_dict(), "args": vars(args), "step": step,
                            "beat_phase_F": best_pw}, out / "best.pt")

    mf.close()
    print(f"DONE. best beat_phase_F={best_pw:.3f}", flush=True)


if __name__ == "__main__":
    main()
