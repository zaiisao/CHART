"""R4 fold-0 training: UNSUPERVISED exact meter-marginal NLL, frozen fold-0 frontend (cached).
Usage: r4_train.py <tag> [--input acts|feats] [--steps N] [--seed S] [--no-eval]
Protocol mirrors r3_lab.py: Adam 1e-3, batch 16, grad clip 1.0."""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch
import mir_eval.beat as meb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import r4_data
from r4_model import R4Conditioned, KERNEL_COMPONENTS

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--input", default="acts", choices=("acts", "feats"))
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--no-eval", action="store_true")
ap.add_argument("--val-nll", type=int, default=0, help="score mean val NLL/frame on N crops")
ap.add_argument("--cache", default=None, help="alternate cache .pt (e.g. long crops)")
ap.add_argument("--select", type=int, default=0, help="every N steps track best val-NLL checkpoint")
ap.add_argument("--select-crops", type=int, default=24)
args = ap.parse_args()

DEVICE = "cuda:1"
FPS = 44100 / 1024
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

cache = r4_data.load(args.cache)
mean, std = cache["feat_mean"].to(DEVICE), cache["feat_std"].to(DEVICE)

def trunk_input_of(crop_or_pair):
    acts, feats = crop_or_pair
    return acts if args.input == "acts" else (feats - mean) / std

crops = [(torch.from_numpy(c["acts"]).to(DEVICE),
          torch.from_numpy(c["feats"]).to(DEVICE)) for c in cache["crops"]]

model = R4Conditioned(fps=FPS, input_mode=args.input, device=DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
history = []

def selection_nll():
    """Mean val NLL/frame on fixed centered crops (checkpoint-selection criterion)."""
    with torch.no_grad():
        vals = []
        for e in cache["val_entries"][:args.select_crops]:
            a = cache["val_acts"][e["stem"]]; f = cache["val_feats"][e["stem"]].astype(np.float32)
            L = min(a.shape[0], 1400)
            s = (a.shape[0] - L) // 2
            a, f = a[s:s+L], f[s:s+L]
            pair = (torch.from_numpy(a).to(DEVICE), torch.from_numpy(f).to(DEVICE))
            vals.append(float(-model.marginal_ll(pair[0], trunk_input_of(pair)) / L))
    return float(np.mean(vals))

best_sel = {"nll": None, "step": -1, "state": None}
t0 = time.time()
for step in range(args.steps):
    idx = rng.choice(len(crops), args.batch, replace=False)
    # per-crop gradient accumulation: identical gradients to the summed loss, ~batch-size lower
    # peak memory (run 2 died at 1400-frame crops with all 16 graphs held before one backward)
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
                # crash-proof: the best-selected state goes to disk NOW (run 2 lesson)
                torch.save({"model": best_sel["state"], "input": args.input, "seed": args.seed,
                            "selected_step": step, "selected_nll": sel},
                           HERE / f"runs/r4_{args.tag}_bestsel.pt")
        print(line, flush=True); history.append(line)
wall_train = time.time() - t0
print(f"train wall {wall_train:.0f}s", flush=True)
torch.save({"model": model.state_dict(), "input": args.input, "seed": args.seed,
            "steps": args.steps}, HERE / f"runs/r4_{args.tag}_final.pt")
if args.select and best_sel["state"] is not None:
    print(f"checkpoint selection: best sel_nll {best_sel['nll']:.4f} at step {best_sel['step']}",
          flush=True)
    model.load_state_dict(best_sel["state"])
torch.save({"model": model.state_dict(), "input": args.input, "seed": args.seed,
            "steps": args.steps, "selected_step": best_sel["step"] if args.select else None},
           HERE / f"runs/r4_{args.tag}.pt")

out = {"tag": args.tag, "input": args.input, "seed": args.seed, "steps": args.steps,
       "selected_step": best_sel["step"] if args.select else None,
       "selected_nll": best_sel["nll"] if args.select else None,
       "wall_train_s": round(wall_train), "history": history}

model.eval()
if args.val_nll:
    with torch.no_grad():
        nlls = []
        for e in cache["val_entries"][:args.val_nll]:
            a = cache["val_acts"][e["stem"]]; f = cache["val_feats"][e["stem"]].astype(np.float32)
            if a.shape[0] > 701:
                s = (a.shape[0] - 700) // 2
                a, f = a[s:s+700], f[s:s+700]
            pair = (torch.from_numpy(a).to(DEVICE), torch.from_numpy(f).to(DEVICE))
            nlls.append(float(-model.marginal_ll(pair[0], trunk_input_of(pair)) / a.shape[0]))
        out["val_nll_per_frame"] = float(np.mean(nlls))
        print(f"val nll/frame {out['val_nll_per_frame']:.4f} over {len(nlls)} crops", flush=True)

def score(ref, est):
    ref, est = meb.trim_beats(ref), meb.trim_beats(est)
    f = meb.f_measure(ref, est) if len(est) and len(ref) else 0.0
    if len(est) and len(ref) > 1:
        _, cmlt, _, amlt = meb.continuity(ref, est)     # (CMLc, CMLt, AMLc, AMLt) -- explicit
    else:
        cmlt = amlt = 0.0
    return f, cmlt, amlt

if not args.no_eval:
    t1 = time.time()
    for deploy, key in ((False, "bare"), (True, "deploy")):
        acc = []
        for e in cache["val_entries"]:
            acts = cache["val_acts"][e["stem"]]
            trunk_np = (acts if args.input == "acts" else
                        ((torch.from_numpy(cache["val_feats"][e["stem"]].astype(np.float32))
                          .to(DEVICE) - mean) / std).cpu().numpy())
            ev = model.decode(acts, trunk_np, deploy=deploy)
            acc.append(score(e["beat_times"], ev["beats"])
                       + score(e["downbeat_times"], ev["downbeats"]))
        out[key] = dict(zip(["beatF", "CMLt", "AMLt", "downbeatF", "dbCMLt", "dbAMLt"],
                            map(float, np.mean(np.array(acc), axis=0))))
        print(key, out[key], flush=True)
    out["wall_eval_s"] = round(time.time() - t1)
json.dump(out, open(HERE / f"results_r4_{args.tag}.json", "w"), indent=1)
print("DONE", args.tag, flush=True)
