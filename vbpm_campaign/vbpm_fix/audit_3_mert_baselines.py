"""AUDIT 3 -- HONEST MERT BASELINES (the bar the fixed VBPM must clear).

Reproduces vbpm/probe_linear.py EXACTLY (same arch/steps/lr/pos_weight/seed) to
re-confirm 0.725 (linear probe) and 0.804 (small conv), then re-scores the SAME
trained probes under the honest protocol (all 79 eval songs, full length) and
per dataset. Also runs the TIME-ROLL leak control on the probes themselves.
"""
import sys, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
from audit_common import (load_split, truncate, agg, banner, beats_from_activation,
                          metronome, f_measure, FPS, PRIOR_MAX_FRAMES, PRIOR_N_EVAL)

dev = "cuda:0"
ROLL = 1000          # 20 s time-roll for the leak control

print("loading cache (feats) ...", flush=True)
t0 = time.time()
tr = load_split("train", with_feats=True)
ev = load_split("eval", with_feats=True)
print(f"  train {len(tr)} eval {len(ev)} in {time.time()-t0:.0f}s", flush=True)


class Probe(nn.Module):
    """Verbatim vbpm/probe_linear.py."""
    def __init__(self, kind="linear"):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(13))
        if kind == "linear":
            self.head = nn.Linear(768, 1)
        else:
            self.head = nn.Sequential(nn.Conv1d(768, 128, 5, padding=2), nn.ReLU(), nn.Conv1d(128, 1, 1))
        self.kind = kind
    def forward(self, feats):
        w = torch.softmax(self.layer_logits, 0)
        m = torch.einsum("l,bltf->btf", w, feats)
        if self.kind == "linear":
            return self.head(m).squeeze(-1)
        return self.head(m.transpose(1, 2)).squeeze(1)


def tgt(beats, start, n):
    y = np.zeros(n, np.float32)
    for t in beats:
        i = int(round(t * FPS)) - start
        if 0 <= i < n: y[i] = 1.0
    return y


def train_probe(kind, steps=800, frames=512, bs=16):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    p = Probe(kind).to(dev); opt = torch.optim.AdamW(p.parameters(), lr=3e-4)
    pw = torch.tensor([12.0], device=dev)
    for step in range(1, steps + 1):
        fe, ys = [], []
        for _ in range(bs):
            s = tr[rng.integers(len(tr))]; T = s["feats"].shape[1]
            if T <= frames: continue
            st = int(rng.integers(0, T - frames))
            fe.append(torch.from_numpy(s["feats"][:, st:st + frames].astype(np.float32)))
            ys.append(torch.from_numpy(tgt(s["beats"], st, frames)))
        fe = torch.stack(fe).to(dev); ys = torch.stack(ys).to(dev)
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(p(fe), ys, pos_weight=pw)
        loss.backward(); opt.step()
        if step % 400 == 0:
            print(f"    [{kind}] s{step} loss={loss.item():.3f}", flush=True)
    return p


@torch.no_grad()
def eval_probe(p, songs, cap, roll=0, thr=0.5):
    p.eval(); rows = []
    for s in songs:
        T, ref, dref = truncate(s, cap)
        if len(ref) < 2: continue
        f = s["feats"][:, :T].astype(np.float32)
        if roll: f = np.roll(f, roll, axis=1)         # features slide, labels do not
        prob = torch.sigmoid(p(torch.from_numpy(f).unsqueeze(0).to(dev)))[0].cpu().numpy()
        est = beats_from_activation(prob, FPS, thr=thr)
        rows.append(dict(beat_F=f_measure(ref, est), metronome_F=f_measure(ref, metronome(T, FPS)),
                         n_est=len(est), n_true=len(ref), dataset=s["dataset"]))
    p.train()
    a = agg(rows, ["beat_F", "metronome_F"])
    a["ratio"] = sum(r["n_est"] for r in rows) / max(sum(r["n_true"] for r in rows), 1)
    a["N"] = len(rows); a["rows"] = rows
    return a


PROTOCOLS = [("eval[:30] <=1600 (PRIOR PROTOCOL)", ev[:PRIOR_N_EVAL], PRIOR_MAX_FRAMES),
             ("ALL 79 eval <=1600",                ev,                PRIOR_MAX_FRAMES),
             ("ALL 79 eval FULL length",           ev,                None)]

banner("(a) 120 BPM METRONOME FLOOR")
for name, songs, cap in PROTOCOLS:
    Fm = []
    for s in songs:
        T, ref, _ = truncate(s, cap)
        if len(ref) >= 2: Fm.append(f_measure(ref, metronome(T, FPS)))
    print(f"  {name:38s} metronome beat_F = {np.mean(Fm):.3f}  (N={len(Fm)})")

out = {}
for kind, prior in [("linear", 0.725), ("conv", 0.804)]:
    banner(f"({'b' if kind=='linear' else 'c'}) {kind.upper()} PROBE + PEAK-PICK   (previously reported {prior})")
    p = train_probe(kind)
    for name, songs, cap in PROTOCOLS:
        a = eval_probe(p, songs, cap)
        tag = "  <-- reproduces the previously reported number" if cap == PRIOR_MAX_FRAMES and songs is ev[:PRIOR_N_EVAL] else ""
        print(f"  {name:38s} beat_F={a['beat_F']:.3f}  metro={a['metronome_F']:.3f}  "
              f"n_est/n_true={a['ratio']:.3f}  N={a['N']}{tag}", flush=True)
        out[f"{kind}|{name}"] = a["beat_F"]
    a_full = eval_probe(p, ev, None)
    ds = {}
    for r in a_full["rows"]: ds.setdefault(r["dataset"], []).append(r["beat_F"])
    print("  per-dataset (ALL 79 FULL): " + "  ".join(f"{k}={np.mean(v):.3f}(n={len(v)})" for k, v in sorted(ds.items())))
    # ---- TIME-ROLL LEAK CONTROL on the probe itself ----
    a_roll = eval_probe(p, ev, None, roll=ROLL)
    print(f"  TIME-ROLL CONTROL (+{ROLL} frames = {ROLL/FPS:.0f}s): beat_F={a_roll['beat_F']:.3f} "
          f"(vs aligned {a_full['beat_F']:.3f}, metronome {a_full['metronome_F']:.3f})  "
          f"-> {'COLLAPSES = clean' if a_roll['beat_F'] < a_full['metronome_F'] + 0.05 else '*** DOES NOT COLLAPSE: LEAK ***'}")
    out[f"{kind}|roll"] = a_roll["beat_F"]
    torch.save(p.state_dict(), f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/probe_{kind}.pt")

json.dump(out, open("/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/audit3_out.json", "w"), indent=1)
print("\n" + json.dumps(out, indent=1))
