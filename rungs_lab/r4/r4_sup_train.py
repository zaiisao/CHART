"""R4 SUPERVISED arm: clamped-path CRF NLL (exact logZ - path_score, conditioned prior +
per-frame mixture kernel in both). Telemetry every 10 steps for the stiffening watch
(era CRF failure: lambda -> ~1000 vs F-optimal ~150; mixture dither is the hypothesized antidote).
Usage: r4_sup_train.py <tag> [--steps N] [--select 25] [--no-eval]"""
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
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--select", type=int, default=0)
ap.add_argument("--no-eval", action="store_true")
ap.add_argument("--objective", default="crf", choices=("crf", "joint"))
args = ap.parse_args()

DEVICE = "cuda:1"; FPS = 44100 / 1024
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

sup = torch.load(HERE / "cache_sup_fold0.pt", weights_only=False)
cache = r4_data.load(str(HERE / "cache_fold0_c1400.pt"))     # val songs + feat standardization
mean, std = cache["feat_mean"].to(DEVICE), cache["feat_std"].to(DEVICE)

def to_gpu(c):
    return (torch.from_numpy(c["acts"]).to(DEVICE),
            (torch.from_numpy(c["feats"]).to(DEVICE) - mean) / std, c["path"], c["meter"])
train = [to_gpu(c) for c in sup["train"]]
val_sel = [to_gpu(c) for c in sup["val_sel"]]

model = R4Conditioned(fps=FPS, input_mode="feats", device=DEVICE)
loss_fn = model.path_nll if args.objective == "crf" else model.joint_path_nll
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def selection_nll():
    with torch.no_grad():
        return float(np.mean([float(loss_fn(a, f, p, m)) / len(p)
                              for a, f, p, m in val_sel]))

best_sel = {"nll": None, "step": -1}
history = []
t0 = time.time()
for step in range(args.steps):
    idx = rng.choice(len(train), args.batch, replace=False)
    opt.zero_grad()
    total = 0.0
    for i in idx:
        a, f, p, mi = train[i]
        loss = (loss_fn(a, f, p, mi) / len(p)) / args.batch
        loss.backward()
        total += float(loss)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 10 == 0 or step == args.steps - 1:
        with torch.no_grad():
            _, _, d = model.head_outputs(train[0][1])
            w = d["component_weights"].mean(0)
        line = (f"step {step:3d} path_nll/frame {total:.4f} lam_t {d['lambda_t'].mean():.1f} "
                f"lam_max {d['lambda_t'].max():.0f} "
                + " ".join(f"{n}={w[i]:.3f}" for i, n in enumerate(KERNEL_COMPONENTS)))
        if args.select and (step % args.select == 0 or step == args.steps - 1):
            sel = selection_nll()
            line += f" | sel_nll {sel:.4f}"
            if best_sel["nll"] is None or sel < best_sel["nll"]:
                best_sel = {"nll": sel, "step": step}
                torch.save({"model": model.state_dict(), "input": "feats", "seed": args.seed,
                            "selected_step": step, "selected_nll": sel},
                           HERE / f"runs/r4_{args.tag}_bestsel.pt")
        print(line, flush=True); history.append(line)
print(f"train wall {time.time()-t0:.0f}s", flush=True)
torch.save({"model": model.state_dict(), "input": "feats", "seed": args.seed,
            "steps": args.steps}, HERE / f"runs/r4_{args.tag}_final.pt")
json.dump({"tag": args.tag, "history": history, "best_sel": best_sel},
          open(HERE / f"results_r4_{args.tag}_train.json", "w"), indent=1)
print("TRAIN_DONE", args.tag, flush=True)
