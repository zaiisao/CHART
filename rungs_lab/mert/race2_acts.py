"""BT beat activations (fold-honest) for era-cache stems: corpus 300 + smc 217. vbpm env."""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path.insert(0, str(LAB))
from training import frontend
from training.data import DEMIX_ROOT, FPS
from data.songs import iter_songs
from smc_data import load_smc

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
OUT = Path("/disk4/jaehoon/VBPM_cache/mert/race")
corpus_stems = {p.stem for p in Path("/disk1/jaehoon/vbpm_mert_layers/corpus").glob("*.pt")}
jobs = []                                             # (stem, fold, mel_path)
for s in iter_songs():
    if s.stem in corpus_stems and s.fold is not None:
        mel = DEMIX_ROOT / s.dataset / f"{s.stem}.npz"
        if mel.exists():
            jobs.append((s.stem, s.fold, mel))
for e in load_smc():
    if Path(f"/disk1/jaehoon/vbpm_mert_layers/smc/{e['stem']}.pt").exists():
        jobs.append((e["stem"], e["fold"], e["mel_path"]))
print(f"{len(jobs)} jobs", flush=True)
by_fold = {}
for stem, fold, mel in jobs:
    by_fold.setdefault(fold, []).append((stem, mel))
out, t0 = {}, time.time()
for fold, ms in sorted(by_fold.items()):
    model = frontend.load_frozen_model(str(LAB / f"checkpoints/bt_fold{fold}_repacked.pt"), DEVICE)
    acts = frontend.activations_for(model, [dict(stem=s, mel_path=m) for s, m in ms], DEVICE)
    for k, v in acts.items():
        out[k] = v[:, 0].astype(np.float32)
    print(f"fold {fold}: {len(ms)} {time.time()-t0:.0f}s", flush=True)
np.savez(OUT / "bt_acts_beat_v2.npz", fps=FPS, **out)
print("DONE", flush=True)
