"""Fold-0 data cache for R4: 300 train crops (acts+feats, 700 frames) + whole-song val acts/feats.
Frozen OFFICIAL BT fold_0 frontend (fold-honest). Run once; scripts load the .pt."""
import sys, time
from pathlib import Path
import numpy as np, torch

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))
from training import frontend, data
from rungs.r1_2016_dbn import DBN2016

CROP = 700
CACHE = Path(__file__).resolve().parent / "cache_fold0.pt"
DEVICE = "cuda:1"


def build(crop=CROP, cache_path=None):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    chassis = DBN2016(fps=data.FPS, device=DEVICE, dtype=torch.float32, observation_lambda=6,
                      num_tempi=None, threshold=0.0, correct=False)
    train_e, val_e, skipped = data.load_songs(chassis.annotated_state_path)
    print(f"train {len(train_e)} val {len(val_e)} skipped {skipped}", flush=True)
    model = frontend.load_frozen_model(str(LAB / "checkpoints/bt_fold0_repacked.pt"), DEVICE)
    t0 = time.time()
    entries = train_e[:300]                       # same protocol as r3_lab / campaign
    tr_feats, tr_acts = frontend.features_for(model, entries, DEVICE)
    crops = []
    for e in entries:
        a, f = tr_acts[e["stem"]], tr_feats[e["stem"]].numpy()
        if a.shape[0] > crop + 1:
            s = int(rng.integers(0, a.shape[0] - crop))
            a, f = a[s:s+crop], f[s:s+crop]
        crops.append({"acts": a.astype(np.float32), "feats": f.astype(np.float32),
                      "stem": e["stem"]})
    val_feats, val_acts = frontend.features_for(model, val_e, DEVICE)
    # feature standardization fit on the train crops (frontend.fit_projection style, simplified)
    stacked = torch.cat([torch.from_numpy(c["feats"]) for c in crops]).double()
    mean, std = stacked.mean(0).float(), stacked.std(0).clamp(min=1e-3).float()
    payload = {
        "crops": crops,
        "val_entries": [{k: e[k] for k in ("stem", "dataset", "beat_times", "downbeat_times",
                                           "beat_frames", "beat_in_bar", "beats_per_bar")}
                        for e in val_e],
        "val_acts": {k: v.astype(np.float32) for k, v in val_acts.items()},
        "val_feats": {k: v.numpy().astype(np.float16) for k, v in val_feats.items()},
        "feat_mean": mean, "feat_std": std,
    }
    target = cache_path or CACHE
    torch.save(payload, target)
    print(f"cached {target} in {time.time()-t0:.0f}s", flush=True)


def load(cache_path=None):
    return torch.load(cache_path or CACHE, weights_only=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--crop", type=int, default=CROP)
    a = ap.parse_args()
    build(a.crop, None if a.crop == CROP else CACHE.with_name(f"cache_fold0_c{a.crop}.pt"))
