"""Select the layer-race song set (200 songs) and write manifest.json.

40 each from ballroom/beatles/hainsworth/hjdb (tempo + octave probes; demix cache exists ->
BT activations available) + 40 gtzan (genre + tempo probes). Annotated tempo = 60 / median IBI.
Run with the vbpm env (uses rungs_lab data catalog).
"""
import json, sys
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))
from data.songs import iter_songs

rng = np.random.default_rng(0)
OUT = Path("/disk4/jaehoon/VBPM_cache/mert/race")
OUT.mkdir(parents=True, exist_ok=True)

per = {"ballroom": 40, "beatles": 40, "hainsworth": 40, "hjdb": 40, "gtzan": 40}
songs = list(iter_songs())
manifest = []
for ds, cap in per.items():
    pool = [s for s in songs if s.dataset == ds]
    if ds == "gtzan":                      # stratify by genre (stem gtzan_<genre>_<id>)
        by_genre = {}
        for s in pool:
            by_genre.setdefault(s.stem.split("_")[1], []).append(s)
        pick = []
        for g, lst in sorted(by_genre.items()):
            idx = rng.choice(len(lst), 4, replace=False)
            pick += [lst[i] for i in idx]
    else:
        idx = rng.choice(len(pool), cap, replace=False)
        pick = [pool[i] for i in idx]
    for s in pick:
        bt, _ = s.beats()
        if len(bt) < 8:
            continue
        ibi = np.diff(bt)
        tempo = 60.0 / float(np.median(ibi))
        manifest.append(dict(stem=s.stem, dataset=s.dataset, fold=s.fold,
                             audio=str(s.audio_path), tempo_ann=tempo,
                             genre=s.stem.split("_")[1] if ds == "gtzan" else None))
json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
from collections import Counter
print(len(manifest), Counter(m["dataset"] for m in manifest))
