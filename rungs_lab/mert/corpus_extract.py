"""Stage 3: full-corpus extraction of the WINNING MERT layers, fp16 @ 50 fps, one npz per song.
chart env. Usage: corpus_extract.py <layers e.g. 5,8,11> [cuda:X] [--shard i/n]"""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from mert_backbone import load_mert, extract_song
from data.songs import iter_songs

layers = [int(x) for x in sys.argv[1].split(",")]
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
shard, nshard = (0, 1)
if len(sys.argv) > 4 and sys.argv[3] == "--shard":
    shard, nshard = map(int, sys.argv[4].split("/"))
OUT = Path("/disk4/jaehoon/VBPM_cache/mert/corpus")
OUT.mkdir(parents=True, exist_ok=True)
songs = [s for i, s in enumerate(sorted(iter_songs(), key=lambda s: s.stem))
         if i % nshard == shard]
model = load_mert(DEVICE)
mani, t0, done = {}, time.time(), 0
for s in songs:
    outp = OUT / f"{s.stem}.npz"
    if not outp.exists():
        try:
            f = extract_song(model, s.audio_path, DEVICE)          # [13, T, 768] @50fps
        except Exception as e:
            print("skip", s.stem, e, flush=True)
            continue
        sel = np.ascontiguousarray(f[layers].transpose(1, 0, 2).reshape(f.shape[1], -1))
        np.savez(outp, feats=sel.astype(np.float16))
    done += 1
    if done % 200 == 0:
        print(f"{done}/{len(songs)} {time.time()-t0:.0f}s", flush=True)
# manifest written by shard 0 collector pass at the end of the last shard is racy; write per-shard
for s in songs:
    p = OUT / f"{s.stem}.npz"
    if p.exists():
        mani[s.stem] = dict(dataset=s.dataset, layers=layers, fps=50)
mp = OUT / f"manifest_shard{shard}.json"
json.dump(mani, open(mp, "w"))
print(f"DONE shard {shard}/{nshard}: {done} songs {time.time()-t0:.0f}s", flush=True)
