"""Fold-0 data cache for the MERT-conditioned R4 (r4/r4_data.py protocol, crop offsets RECORDED).

Same song set, same seed-0 rng stream and crop draws as r4/r4_data.py build(crop=1400), so the BT
side of every crop is bit-identical to the run2b cache; additionally stores each crop's start
offset so a second feature stream can be sliced in alignment later.

  phase A (vbpm env):  mert_r4_data.py bt [cuda:X]         -> BT crops/val + offsets
  phase B (chart env): mert_r4_data.py mert 7,8,11 [cuda:X] -> adds aligned MERT stream + stats

MERT stream is extracted at the BT frame rate (44100/1024 ~ 43.07 fps, frame centers (k+.5)/fps),
selected layers concatenated -> [T, k*768] fp16; length mismatches vs the mel frame count (+-2
frames) are edge-padded/truncated to the acts length before slicing.
Cache: /disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(LAB))

CACHE = Path("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt")
BT_FPS = 44100 / 1024
CROP = 1400


def build_bt(device):
    from training import frontend, data
    from rungs.r1_2016_dbn import DBN2016
    torch.manual_seed(0); rng = np.random.default_rng(0)
    chassis = DBN2016(fps=data.FPS, device=device, dtype=torch.float32, observation_lambda=6,
                      num_tempi=None, threshold=0.0, correct=False)
    train_e, val_e, skipped = data.load_songs(chassis.annotated_state_path)
    print(f"train {len(train_e)} val {len(val_e)} skipped {skipped}", flush=True)
    model = frontend.load_frozen_model(str(LAB / "checkpoints/bt_fold0_repacked.pt"), device)
    t0 = time.time()
    entries = train_e[:300]
    tr_feats, tr_acts = frontend.features_for(model, entries, device)
    crops = []
    for e in entries:
        a, f = tr_acts[e["stem"]], tr_feats[e["stem"]].numpy()
        s = 0
        if a.shape[0] > CROP + 1:
            s = int(rng.integers(0, a.shape[0] - CROP))       # same rng stream as r4_data.build
            a, f = a[s:s+CROP], f[s:s+CROP]
        crops.append({"acts": a.astype(np.float32), "feats": f.astype(np.float32),
                      "stem": e["stem"], "start": s})
    val_feats, val_acts = frontend.features_for(model, val_e, device)
    stacked = torch.cat([torch.from_numpy(c["feats"]) for c in crops]).double()
    payload = {
        "crops": crops,
        "val_entries": [{k: e[k] for k in ("stem", "dataset", "beat_times", "downbeat_times",
                                           "beat_frames", "beat_in_bar", "beats_per_bar")}
                        for e in val_e],
        "val_acts": {k: v.astype(np.float32) for k, v in val_acts.items()},
        "val_feats": {k: v.numpy().astype(np.float16) for k, v in val_feats.items()},
        "feat_mean": stacked.mean(0).float(), "feat_std": stacked.std(0).clamp(min=1e-3).float(),
    }
    torch.save(payload, CACHE)
    print(f"phase A cached {CACHE} in {time.time()-t0:.0f}s", flush=True)


def build_mert(layers, device):
    from mert_backbone import load_mert, extract_song
    from data.songs import iter_songs
    cache = torch.load(CACHE, weights_only=False)
    audio_of = {s.stem: s.audio_path for s in iter_songs()}
    model = load_mert(device)
    need = list(dict.fromkeys([c["stem"] for c in cache["crops"]]
                              + [e["stem"] for e in cache["val_entries"]]))
    t0 = time.time()
    full = {}
    for i, stem in enumerate(need):
        f = extract_song(model, audio_of[stem], device, fps=BT_FPS)          # [13, T, 768] fp16
        full[stem] = np.ascontiguousarray(f[layers].transpose(1, 0, 2)
                                          .reshape(f.shape[1], -1))          # [T, k*768]
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(need)} extracted {time.time()-t0:.0f}s", flush=True)

    def fit_len(x, n):
        if x.shape[0] >= n:
            return x[:n]
        return np.concatenate([x, np.repeat(x[-1:], n - x.shape[0], axis=0)])

    for c in cache["crops"]:
        m = fit_len(full[c["stem"]], c["start"] + c["acts"].shape[0])[c["start"]:]
        c["mert"] = m.astype(np.float16)
    cache["val_mert"] = {e["stem"]: fit_len(full[e["stem"]],
                                            cache["val_acts"][e["stem"]].shape[0])
                         for e in cache["val_entries"]}
    stacked = torch.cat([torch.from_numpy(c["mert"].astype(np.float32))
                         for c in cache["crops"]]).double()
    cache["mert_mean"] = stacked.mean(0).float()
    cache["mert_std"] = stacked.std(0).clamp(min=1e-3).float()
    cache["mert_layers"] = layers
    torch.save(cache, CACHE)
    print(f"phase B done ({len(need)} songs, layers {layers}) {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "bt":
        build_bt(sys.argv[2] if len(sys.argv) > 2 else "cuda:1")
    else:
        build_mert([int(x) for x in sys.argv[2].split(",")],
                   sys.argv[3] if len(sys.argv) > 3 else "cuda:0")
