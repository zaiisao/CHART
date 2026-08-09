"""MERT-R4 fold-0 training: COPY of r4/r4_train.py (run2b protocol: unsupervised marginal NLL,
Adam 1e-3 batch 16 grad-clip 1.0, 1400-frame crops, --select best-val-NLL checkpointing saved to
disk at selection time) with a third input mode `featsmert`: [BT feats 256 ; MERT k*768], each
stream standardized with its own train-crop mean/std. vbpm env may lack nothing new (torch only).
Usage: mert_r4_train.py <tag> [--input featsmert] [--steps N] [--select N] ..."""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch
import mir_eval.beat as meb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from mert_r4_model import R4Conditioned, KERNEL_COMPONENTS

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--input", default="featsmert", choices=("acts", "feats", "featsmert"))
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--no-eval", action="store_true")
ap.add_argument("--select", type=int, default=0)
ap.add_argument("--select-crops", type=int, default=24)
ap.add_argument("--device", default="cuda:1")
args = ap.parse_args()

DEVICE = args.device
FPS = 44100 / 1024
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

CACHE = Path("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt")
cache = torch.load(CACHE, weights_only=False)
mean, std = cache["feat_mean"].to(DEVICE), cache["feat_std"].to(DEVICE)
mmean, mstd = cache["mert_mean"].to(DEVICE), cache["mert_std"].to(DEVICE)
MERT_DIM = int(mmean.shape[0])
INPUT_DIM = {"acts": 2, "feats": 256, "featsmert": 256 + MERT_DIM}[args.input]

def trunk_input_of(triple):
    acts, feats, mert = triple
    if args.input == "acts":
        return acts
    f = (feats - mean) / std
    if args.input == "feats":
        return f
    return torch.cat([f, (mert - mmean) / mstd], dim=1)

crops = [(torch.from_numpy(c["acts"]).to(DEVICE),
          torch.from_numpy(c["feats"]).to(DEVICE),
          torch.from_numpy(c["mert"].astype(np.float32)).to(DEVICE)) for c in cache["crops"]]

def val_triple(stem, max_len=None, center=False):
    a = cache["val_acts"][stem]
    f = cache["val_feats"][stem].astype(np.float32)
    m = cache["val_mert"][stem].astype(np.float32)
    L = a.shape[0]
    if max_len and L > max_len:
        s = (L - max_len) // 2 if center else 0
        a, f, m = a[s:s+max_len], f[s:s+max_len], m[s:s+max_len]
    return (torch.from_numpy(a).to(DEVICE), torch.from_numpy(f).to(DEVICE),
            torch.from_numpy(m).to(DEVICE))

model = R4Conditioned(fps=FPS, input_mode=args.input, device=DEVICE, input_dim=INPUT_DIM)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
history = []

def selection_nll():
    with torch.no_grad():
        vals = []
        for e in cache["val_entries"][:args.select_crops]:
            t = val_triple(e["stem"], max_len=1400, center=True)
            vals.append(float(-model.marginal_ll(t[0], trunk_input_of(t)) / t[0].shape[0]))
    return float(np.mean(vals))

best_sel = {"nll": None, "step": -1, "state": None}
(HERE / "runs").mkdir(exist_ok=True)
t0 = time.time()
for step in range(args.steps):
    idx = rng.choice(len(crops), args.batch, replace=False)
    opt.zero_grad()
    total = 0.0
    for i in idx:
        crop_loss = (-model.marginal_ll(crops[i][0], trunk_input_of(crops[i]))
                     / crops[i][0].shape[0]) / args.batch
        crop_loss.backward()
        total += float(crop_loss)
    loss = torch.tensor(total)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 25 == 0 or step == args.steps - 1:
        with torch.no_grad():
            _, _, d = model.head_outputs(trunk_input_of(crops[0]))
            w = d["component_weights"].mean(0)
        line = (f"step {step:3d} nll/frame {loss.item():.4f} lam_t {d['lambda_t'].mean():.1f} "
                + " ".join(f"{n}={w[i]:.3f}" for i, n in enumerate(KERNEL_COMPONENTS)))
        if args.select:
            sel = selection_nll()
            line += f" | sel_nll {sel:.4f}"
            if best_sel["nll"] is None or sel < best_sel["nll"]:
                best_sel = {"nll": sel, "step": step,
                            "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                torch.save({"model": best_sel["state"], "input": args.input, "seed": args.seed,
                            "input_dim": INPUT_DIM, "selected_step": step, "selected_nll": sel},
                           HERE / f"runs/mertr4_{args.tag}_bestsel.pt")
        print(line, flush=True); history.append(line)
wall_train = time.time() - t0
print(f"train wall {wall_train:.0f}s", flush=True)
if args.select and best_sel["state"] is not None:
    print(f"checkpoint selection: best sel_nll {best_sel['nll']:.4f} at step {best_sel['step']}",
          flush=True)
    model.load_state_dict(best_sel["state"])
torch.save({"model": model.state_dict(), "input": args.input, "seed": args.seed,
            "input_dim": INPUT_DIM, "steps": args.steps,
            "selected_step": best_sel["step"] if args.select else None},
           HERE / f"runs/mertr4_{args.tag}.pt")

out = {"tag": args.tag, "input": args.input, "seed": args.seed, "steps": args.steps,
       "mert_layers": cache.get("mert_layers"), "input_dim": INPUT_DIM,
       "selected_step": best_sel["step"] if args.select else None,
       "selected_nll": best_sel["nll"] if args.select else None,
       "wall_train_s": round(wall_train), "history": history}

def score(ref, est):
    ref, est = meb.trim_beats(ref), meb.trim_beats(est)
    f = meb.f_measure(ref, est) if len(est) and len(ref) else 0.0
    if len(est) and len(ref) > 1:
        _, cmlt, _, amlt = meb.continuity(ref, est)     # (CMLc, CMLt, AMLc, AMLt) -- explicit
    else:
        cmlt = amlt = 0.0
    return f, cmlt, amlt

model.eval()
if not args.no_eval:
    t1 = time.time()
    for deploy, key in ((False, "bare"), (True, "deploy")):
        acc = []
        for e in cache["val_entries"]:
            t = val_triple(e["stem"])
            ev = model.decode(t[0].cpu().numpy(), trunk_input_of(t).cpu().numpy(), deploy=deploy)
            acc.append(score(e["beat_times"], ev["beats"])
                       + score(e["downbeat_times"], ev["downbeats"]))
        out[key] = dict(zip(["beatF", "CMLt", "AMLt", "downbeatF", "dbCMLt", "dbAMLt"],
                            map(float, np.mean(np.array(acc), axis=0))))
        print(key, out[key], flush=True)
    out["wall_eval_s"] = round(time.time() - t1)
json.dump(out, open(HERE / f"results_mertr4_{args.tag}.json", "w"), indent=1)
print("DONE", args.tag, flush=True)
