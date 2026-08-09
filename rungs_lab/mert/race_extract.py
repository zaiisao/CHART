"""Extract all-13-layer MERT (50 fps fp16) for the race set. chart env, GPU from argv."""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mert_backbone import load_mert, extract_song

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
RACE = Path("/disk4/jaehoon/VBPM_cache/mert/race")
manifest = json.load(open(RACE / "manifest.json"))
model = load_mert(DEVICE)
t0, done = time.time(), 0
for m in manifest:
    outp = RACE / f"{m['stem']}.npy"
    if outp.exists():
        done += 1
        continue
    f = extract_song(model, m["audio"], DEVICE)      # [13, T, 768] fp16
    np.save(outp, f)
    done += 1
    if done % 20 == 0:
        print(f"{done}/{len(manifest)} {time.time()-t0:.0f}s", flush=True)
print(f"DONE {done} in {time.time()-t0:.0f}s", flush=True)
