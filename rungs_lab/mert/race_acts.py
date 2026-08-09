"""BT activations (fold-honest checkpoint per song) for the 160 demixed race songs -> npz.
vbpm env. GPU from argv. Saves beat-channel activation [T] per stem + fps."""
import json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path.insert(0, str(LAB))
from training import frontend
from training.data import DEMIX_ROOT, FPS

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
RACE = Path("/disk4/jaehoon/VBPM_cache/mert/race")
manifest = [m for m in json.load(open(RACE / "manifest.json")) if m["dataset"] != "gtzan"]
by_fold = {}
for m in manifest:
    by_fold.setdefault(m["fold"], []).append(m)
out = {}
t0 = time.time()
for fold, ms in sorted(by_fold.items()):
    model = frontend.load_frozen_model(str(LAB / f"checkpoints/bt_fold{fold}_repacked.pt"), DEVICE)
    entries = [dict(stem=m["stem"], mel_path=DEMIX_ROOT / m["dataset"] / f"{m['stem']}.npz")
               for m in ms]
    acts = frontend.activations_for(model, entries, DEVICE)
    for k, v in acts.items():
        out[k] = v[:, 0].astype(np.float32)          # beat channel
    print(f"fold {fold}: {len(ms)} songs {time.time()-t0:.0f}s", flush=True)
np.savez(RACE / "bt_acts_beat.npz", fps=FPS, **out)
print("DONE", flush=True)
