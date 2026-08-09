"""Mandatory R4 controls (run at INIT, before any result is reported):
1. gradient audit  : every head + trunk gets nonzero grad on a real batch
2. position ablation: zeroed audio input -> per-frame kernels/prior go uninformative (constant/uniform)
3. degeneracy      : trunk output zeroed -> decode parity with global-kernel R2mix (w0, lam0)
"""
import sys, json
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import r4_data
from r4_model import R4Conditioned
from mixture import MixtureLambda

DEVICE = "cuda:1"; FPS = 44100 / 1024
torch.manual_seed(0)
cache = r4_data.load()
crops = [(torch.from_numpy(c["acts"]).to(DEVICE),
          torch.from_numpy(c["feats"]).to(DEVICE)) for c in cache["crops"][:8]]
mean, std = cache["feat_mean"].to(DEVICE), cache["feat_std"].to(DEVICE)
report = {}

for input_mode in ("acts", "feats"):
    model = R4Conditioned(fps=FPS, input_mode=input_mode, device=DEVICE)
    tin = lambda p: p[0] if input_mode == "acts" else (p[1] - mean) / std

    # 1. gradient audit
    loss = sum(-model.marginal_ll(p[0], tin(p)) / p[0].shape[0] for p in crops) / len(crops)
    model.zero_grad(); loss.backward()
    norms = {name: float(torch.cat([p.grad.flatten() for p in group.parameters()]).norm())
             for name, group in (("trunk", model.trunk), ("prior_head", model.prior_head),
                                 ("kernel_head", model.kernel_head))}
    report[f"grad_norms_{input_mode}"] = norms
    print(f"[{input_mode}] grad norms:", norms, flush=True)

    # 2. position ablation: zero the input
    model._ablate_input = True
    with torch.no_grad():
        _, _, d = model.head_outputs(tin(crops[0]))
    lam_std = float(d["lambda_t"].std())
    w_std = float(d["component_weights"].std(0).max())
    prior_dev = float((d["prior"] - 1.0 / model.num_tempi).abs().max())
    report[f"ablation_{input_mode}"] = {"lambda_t_std": lam_std, "weight_std_max": w_std,
                                        "prior_max_dev_from_uniform": prior_dev}
    print(f"[{input_mode}] ablation: lam_t std {lam_std:.2e} w std {w_std:.2e} "
          f"prior dev {prior_dev:.2e}", flush=True)
    model._ablate_input = False

# 3. degeneracy: zero trunk output vs R2mix (banked fold-0 optimum w=0.39, lam=104.43)
model = R4Conditioned(fps=FPS, input_mode="acts", device=DEVICE)
model._zero_trunk = True
r2mix = MixtureLambda(fps=FPS, device=DEVICE, observation_lambda=6)
r2mix.mixture_weight, r2mix.transition_lambda = 0.39, 104.43
import mir_eval.beat as meb
def bf(ref, est):
    ref, est = meb.trim_beats(ref), meb.trim_beats(est)
    return meb.f_measure(ref, est) if len(est) and len(ref) else 0.0
subset = cache["val_entries"][:30]
r4_f, r2_f, agree = [], [], []
with torch.no_grad():
    for e in subset:
        acts = cache["val_acts"][e["stem"]]
        ev4 = model.decode(acts, acts, deploy=True)
        ev2 = r2mix.decode(acts.astype(np.float64), deploy=True)
        r4_f.append(bf(e["beat_times"], ev4["beats"]))
        r2_f.append(bf(e["beat_times"], ev2["beats"]))
        b4, b2 = ev4["beats"], ev2["beats"]
        n = min(len(b4), len(b2))
        agree.append(float(np.mean(np.abs(b4[:n] - b2[:n]) < 0.07)) if n else 0.0)
report["degeneracy"] = {"r4_zerotrunk_beatF": float(np.mean(r4_f)),
                        "r2mix_beatF": float(np.mean(r2_f)),
                        "event_agreement_70ms": float(np.mean(agree)), "n_songs": len(subset)}
print("degeneracy:", report["degeneracy"], flush=True)
json.dump(report, open(HERE / "results_controls.json", "w"), indent=1)
