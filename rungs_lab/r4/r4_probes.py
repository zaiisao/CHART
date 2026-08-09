"""R4 discovery probes (the point of the model). Usage: r4_probes.py <tag>
1. Initial-prior sharpness/accuracy: entropy of p(s_0|x); Spearman(argmax tempo, annotated tempo);
   octave resolution vs the zero-trunk (global) decode.
2. Kernel conditioning: Spearman(per-frame change-hazard & width, annotated local tempo
   volatility) -- bar to beat: R3's rho ~ 0.19. Half/double weights vs songs with octave jumps.
3. History: attention mass from tempo-change frames onto EARLIER change locations vs random
   frames (spot check, 3 songs).
"""
import sys, json
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import r4_data
from r4_model import R4Conditioned, _rope_cache

TAG = sys.argv[1]
DEVICE = "cuda:1"; FPS = 44100 / 1024
cache = r4_data.load()
ckpt = torch.load(HERE / f"runs/r4_{TAG}.pt", weights_only=False)
INPUT = ckpt["input"]
model = R4Conditioned(fps=FPS, input_mode=INPUT, device=DEVICE)
model.load_state_dict(ckpt["model"]); model.eval()
mean, std = cache["feat_mean"].to(DEVICE), cache["feat_std"].to(DEVICE)
bpm_grid = 60.0 * FPS / (model._min_interval + np.arange(model.num_tempi))   # tempo of each bin

def trunk_input(e):
    acts = cache["val_acts"][e["stem"]]
    if INPUT == "acts":
        return torch.from_numpy(acts).to(DEVICE)
    f = torch.from_numpy(cache["val_feats"][e["stem"]].astype(np.float32)).to(DEVICE)
    return (f - mean) / std

report = {"tag": TAG, "input": INPUT}
entropies, argmax_bpm, annotated_bpm = [], [], []
hazard_rho_frames, width_rho_frames = [], []
per_song = []
with torch.no_grad():
    for e in cache["val_entries"]:
        tin = trunk_input(e)
        log_prior, _, d = model.head_outputs(tin)
        prior = d["prior"].cpu().numpy()
        entropies.append(float(-(prior * np.log(prior)).sum()))
        argmax_bpm.append(float(bpm_grid[prior.argmax()]))
        ibi = np.diff(e["beat_times"]); ibi = ibi[ibi > 1e-3]
        annotated_bpm.append(60.0 / np.median(ibi))
        # per-frame annotated volatility: |relative interval change| at each beat, held between beats
        bf = e["beat_frames"]; intervals = np.diff(bf).astype(float)
        vol_beat = np.abs(np.diff(intervals)) / intervals[:-1]           # at beats 1..n-1
        T = tin.shape[0]
        vol = np.zeros(T)
        for i, v in enumerate(vol_beat):
            lo, hi = bf[i + 1], min(bf[i + 2], T)
            if lo < T:
                vol[lo:hi] = v
        w = d["component_weights"].cpu().numpy()                          # [T, 5]
        hazard = 1.0 - w[:, 0]                                            # 1 - hold weight
        width = 1.0 / d["lambda_t"].cpu().numpy()                         # kernel looseness
        lo, hi = bf[0], min(bf[-1], T)                                    # annotated span only
        if hi - lo > 100 and vol[lo:hi].std() > 0:
            hazard_rho_frames.append(spearmanr(hazard[lo:hi], vol[lo:hi]).statistic)
            width_rho_frames.append(spearmanr(width[lo:hi], vol[lo:hi]).statistic)
        per_song.append({"stem": e["stem"], "dataset": e["dataset"],
                         "prior_entropy": entropies[-1], "prior_bpm": argmax_bpm[-1],
                         "annotated_bpm": annotated_bpm[-1],
                         "mean_w_half": float(w[:, 3].mean()),
                         "mean_w_double": float(w[:, 4].mean()),
                         "mean_hazard": float(hazard.mean()),
                         "has_octave_jump": bool((np.abs(np.log2(intervals[1:] / intervals[:-1]))
                                                  > 0.7).any())})

uniform_entropy = float(np.log(model.num_tempi))
report["prior"] = {
    "mean_entropy": float(np.mean(entropies)), "uniform_entropy": uniform_entropy,
    "spearman_argmax_vs_annotated": float(spearmanr(argmax_bpm, annotated_bpm).statistic),
    "n": len(entropies)}

# octave probe: songs where the ZERO-TRUNK (global) decode lands on the wrong octave
model._zero_trunk = True
octave_cases = []
with torch.no_grad():
    for i, e in enumerate(cache["val_entries"]):
        acts = cache["val_acts"][e["stem"]]
        ev = model.decode(acts, acts if INPUT == "acts" else trunk_input(e).cpu().numpy(),
                          deploy=True)
        if len(ev["beats"]) < 4:
            continue
        est_bpm = 60.0 / np.median(np.diff(ev["beats"]))
        r = est_bpm / annotated_bpm[i]
        if 1.8 < r < 2.2 or 0.45 < r < 0.55:
            octave_cases.append({"stem": e["stem"], "global_est_bpm": float(est_bpm),
                                 "annotated_bpm": float(annotated_bpm[i]), "ratio": float(r)})
model._zero_trunk = False
# now score the conditioned prior on those cases
resolved = 0
with torch.no_grad():
    for case in octave_cases:
        e = next(e for e in cache["val_entries"] if e["stem"] == case["stem"])
        _, _, d = model.head_outputs(trunk_input(e))
        prior = d["prior"].cpu().numpy()
        def mass(bpm):
            sel = np.abs(bpm_grid / bpm - 1.0) < 0.10
            return float(prior[sel].sum())
        case["prior_mass_correct"] = mass(case["annotated_bpm"])
        case["prior_mass_wrong_octave"] = mass(case["global_est_bpm"])
        resolved += case["prior_mass_correct"] > case["prior_mass_wrong_octave"]
report["octave"] = {"n_wrong_octave_global": len(octave_cases),
                    "prior_prefers_correct": int(resolved), "cases": octave_cases}

report["kernel"] = {
    "spearman_hazard_vs_volatility_per_song_mean": float(np.nanmean(hazard_rho_frames)),
    "spearman_width_vs_volatility_per_song_mean": float(np.nanmean(width_rho_frames)),
    "n_songs": len(hazard_rho_frames),
    "r3_bar": 0.19}
oct_w = [s["mean_w_half"] + s["mean_w_double"] for s in per_song]
has = np.array([s["has_octave_jump"] for s in per_song])
report["kernel"]["octave_weight_jump_songs"] = float(np.mean([w for w, h in zip(oct_w, has) if h])) if has.any() else None
report["kernel"]["octave_weight_other_songs"] = float(np.mean([w for w, h in zip(oct_w, has) if not h]))
report["kernel"]["n_jump_songs"] = int(has.sum())

# 3. attention history spot check: 3 songs with most volatility spikes
spike_counts = []
for e in cache["val_entries"]:
    iv = np.diff(e["beat_frames"]).astype(float)
    spike_counts.append(int((np.abs(np.diff(iv)) / iv[:-1] > 0.08).sum()))
picks = np.argsort(spike_counts)[::-1][:3]
attn_results = []
with torch.no_grad():
    for pi in picks:
        e = cache["val_entries"][pi]
        tin = trunk_input(e)[:6000]
        h = model.trunk.embed((torch.zeros_like(tin) if model._ablate_input else tin)).unsqueeze(0)
        cos, sin = _rope_cache(tin.shape[0], model.trunk.head_dim, tin.device, h.dtype)
        for block in model.trunk.blocks[:-1]:
            h = block(h, cos, sin)
        attn = model.trunk.blocks[-1].attention_weights(h, cos, sin).mean(0).cpu().numpy()  # [T,T]
        iv = np.diff(e["beat_frames"]).astype(float)
        change_beats = np.where(np.abs(np.diff(iv)) / iv[:-1] > 0.08)[0] + 1
        change_frames = e["beat_frames"][change_beats]
        change_frames = change_frames[change_frames < tin.shape[0]]
        if len(change_frames) < 2:
            continue
        half_window = int(FPS)                                            # +-1 s
        rng = np.random.default_rng(0)
        mass_on_changes, mass_random = [], []
        for f in change_frames[1:]:
            earlier = change_frames[change_frames < f - half_window]
            if not len(earlier):
                continue
            row = attn[f]
            sel = np.zeros(len(row), bool)
            for g in earlier:
                sel[max(0, g - half_window):g + half_window] = True
            mass_on_changes.append(float(row[sel].sum() / sel.sum()))
            rand = rng.choice(len(earlier) * 2 * half_window, size=1)     # matched-size random mass
            rsel = np.zeros(len(row), bool)
            for g in rng.integers(half_window, f - half_window, size=len(earlier)):
                rsel[g - half_window:g + half_window] = True
            mass_random.append(float(row[rsel].sum() / max(rsel.sum(), 1)))
        if mass_on_changes:
            attn_results.append({"stem": e["stem"], "n_changes": int(len(change_frames)),
                                 "mean_attn_density_on_earlier_changes": float(np.mean(mass_on_changes)),
                                 "mean_attn_density_random": float(np.mean(mass_random))})
report["attention_history"] = attn_results
json.dump(report, open(HERE / f"results_probes_{TAG}.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in report.items() if k != "octave"}, indent=1, default=float))
print("octave:", report["octave"]["n_wrong_octave_global"], "wrong-octave global cases;",
      report["octave"]["prior_prefers_correct"], "resolved by conditioned prior")
