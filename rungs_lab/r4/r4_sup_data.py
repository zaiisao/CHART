"""Supervised-arm data: beat-aligned <=1400-frame crops with CLAMPED annotated state paths
(jitter-smoothed beat frames), for train (300) and val-selection (24). Fold-0, fold_0 frontend."""
import sys
from pathlib import Path
import numpy as np, torch

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))
from training import frontend, data
from rungs.r1_2016_dbn import DBN2016

CACHE = Path(__file__).resolve().parent / "cache_sup_fold0.pt"
DEVICE = "cuda:1"; MAXF = 1400


def aligned_crop(entry, rng, chassis, start=None):
    bf, bib, bpb = entry["beat_frames"], entry["beat_in_bar"], entry["beats_per_bar"]
    if len(bf) < 5:
        return None
    s = int(rng.integers(0, max(1, len(bf) - 4))) if start is None else start
    e = s + 1
    while e + 1 < len(bf) and bf[e + 1] - bf[s] <= MAXF:
        e += 1
    built = chassis.annotated_state_path(bf[s:e+1] - bf[s], bib[s:e+1], bpb)
    if built is None:
        return None
    path, mi = built
    return int(bf[s]), int(bf[e]), path.astype(np.int64), mi


def build():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    chassis = DBN2016(fps=data.FPS, device=DEVICE, dtype=torch.float32, observation_lambda=6,
                      num_tempi=None, threshold=0.0, correct=False)
    train_e, val_e, _ = data.load_songs(chassis.annotated_state_path)
    data.apply_smooth_targets(train_e + val_e)          # remove +-1 frame quantization jitter
    model = frontend.load_frozen_model(str(LAB / "checkpoints/bt_fold0_repacked.pt"), DEVICE)

    def crops_for(entries, n, centered):
        feats, acts = frontend.features_for(model, entries, DEVICE)
        out = []
        for entry in entries:
            got = aligned_crop(entry, rng, chassis,
                               start=(max(0, len(entry["beat_frames"]) // 2 - 8) if centered else None))
            if got is None:
                continue
            f0, f1, path, mi = got
            a = acts[entry["stem"]]
            if f1 > a.shape[0]:
                continue
            out.append({"stem": entry["stem"],
                        "acts": a[f0:f1].astype(np.float32),
                        "feats": feats[entry["stem"]].numpy()[f0:f1].astype(np.float32),
                        "path": path[:f1 - f0], "meter": mi})
            if len(out) >= n:
                break
        return out

    train_crops = crops_for(train_e[:340], 300, centered=False)
    val_crops = crops_for(val_e[:30], 24, centered=True)
    print(f"train crops {len(train_crops)} | val-selection crops {len(val_crops)}", flush=True)
    torch.save({"train": train_crops, "val_sel": val_crops}, CACHE)
    print("cached", CACHE, flush=True)


if __name__ == "__main__":
    build()
