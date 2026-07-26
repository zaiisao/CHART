"""SHARED FROZEN ACTIVATION HEAD  (ground truth for every downstream arm).

Trains a small conv head on frozen MERT features -> 2 channels
(beat activation, downbeat activation), supervised with beat/downbeat labels,
ON THE TRAIN FOLD ONLY (147 songs).  Eval fold (79 songs, fold 0) is NEVER
touched during training.

  learnable softmax over 13 MERT layers
    -> Conv1d(768,128,5,pad=2) -> ReLU -> Conv1d(128,2,1)
  BCE-with-logits, pos_weight 12, crops of 512 frames, AdamW 3e-4.

Outputs
  /home/sogang/jaehoon/VBPM_reintegration/vbpm_ground/act_head_shared.pt
  /home/sogang/jaehoon/VBPM_reintegration/vbpm_arms/act_eval.npz
  /home/sogang/jaehoon/VBPM_reintegration/vbpm_arms/act_train.npz
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")

from vbpm.evaluate import (  # noqa: E402
    beats_from_activation, f_measure, metronome, _estimate_meter,
)
from audit_common import load_split, ideal_barphase, banner, FPS  # noqa: E402

DEV = "cuda:0"
GROUND = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_ground"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
TWO_PI = 2.0 * math.pi

CONFIG = dict(n_layers=13, feat_dim=768, hidden=128, kernel=5, out_ch=2,
              pos_weight=12.0, steps=2200, crop=512, batch=16, lr=3e-4,
              seed=0, fps=FPS, channels=["beat", "downbeat"])


# --------------------------------------------------------------------------- head
class ActHead(nn.Module):
    def __init__(self, cfg=CONFIG):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(cfg["n_layers"]))
        self.head = nn.Sequential(
            nn.Conv1d(cfg["feat_dim"], cfg["hidden"], cfg["kernel"],
                      padding=cfg["kernel"] // 2),
            nn.ReLU(),
            nn.Conv1d(cfg["hidden"], cfg["out_ch"], 1))

    def forward(self, feats):                 # feats [B,13,T,768] -> logits [B,T,2]
        w = torch.softmax(self.layer_logits, 0)
        m = torch.einsum("l,bltf->btf", w, feats)
        return self.head(m.transpose(1, 2)).transpose(1, 2)


def targets(beats, downs, start, n):
    y = np.zeros((n, 2), np.float32)
    for t in beats:
        i = int(round(t * FPS)) - start
        if 0 <= i < n:
            y[i, 0] = 1.0
    for t in downs:
        i = int(round(t * FPS)) - start
        if 0 <= i < n:
            y[i, 1] = 1.0
    return y


# --------------------------------------------------------------------------- controls
def blind_grid_controls(ref, T, n_est, n_off=12):
    """Density-matched blind control: uniform grid with the SAME number of beats.

    Returns (same_density_F_at_offset_0, best_of_12_offsets_F)."""
    dur = T / FPS
    if n_est < 2:
        return float("nan"), float("nan")
    per = dur / n_est
    base = np.arange(n_est) * per
    f0 = f_measure(ref, base)
    best = max(f_measure(ref, base + k * per / n_off) for k in range(n_off))
    return f0, max(best, f0)


def circ_lin_corr(a, phi):
    """Circular-linear correlation coefficient (Mardia) between linear a and circular phi."""
    c, s = np.cos(phi), np.sin(phi)
    if a.std() < 1e-9:
        return 0.0
    rca = np.corrcoef(a, c)[0, 1]
    rsa = np.corrcoef(a, s)[0, 1]
    rcs = np.corrcoef(c, s)[0, 1]
    v = (rca ** 2 + rsa ** 2 - 2 * rca * rsa * rcs) / max(1.0 - rcs ** 2, 1e-9)
    return float(np.sqrt(max(v, 0.0)))


def phase_lock(a, phi):
    """Activation-weighted circular resultant length |E[a e^{i phi}]|/E[a] (0 = no locking)."""
    w = np.clip(a, 0, None)
    if w.sum() < 1e-9:
        return 0.0, 0.0
    z = (w * np.exp(1j * phi)).sum()
    return float(np.abs(z) / w.sum()), float(np.angle(z) % TWO_PI)


def ac_peak_lag(a, lo=6, hi=140):
    """Lag (frames) of the highest autocorrelation peak in [lo,hi]."""
    x = a - a.mean()
    if x.std() < 1e-9:
        return float("nan")
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / max(ac[0], 1e-12)
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return float("nan")
    return int(lo + np.argmax(ac[lo:hi]))


# --------------------------------------------------------------------------- main
def main():
    torch.manual_seed(CONFIG["seed"])
    rng = np.random.default_rng(CONFIG["seed"])

    print("loading cache ...", flush=True)
    t0 = time.time()
    tr = load_split("train", with_feats=True)
    ev = load_split("eval", with_feats=True)
    print(f"  train {len(tr)}  eval {len(ev)}  in {time.time() - t0:.0f}s", flush=True)
    assert all(s["fold"] != 0 for s in tr), "train split contains fold 0!"
    assert all(s["fold"] == 0 for s in ev), "eval split is not fold 0!"

    net = ActHead().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=CONFIG["lr"])
    pw = torch.tensor([CONFIG["pos_weight"], CONFIG["pos_weight"]], device=DEV)
    crop, bs = CONFIG["crop"], CONFIG["batch"]

    banner("(1) TRAIN SHARED ACTIVATION HEAD  (train fold only, 147 songs)")
    net.train()
    for step in range(1, CONFIG["steps"] + 1):
        fe, ys = [], []
        while len(fe) < bs:
            s = tr[rng.integers(len(tr))]
            T = s["feats"].shape[1]
            if T <= crop:
                continue
            st = int(rng.integers(0, T - crop))
            fe.append(torch.from_numpy(s["feats"][:, st:st + crop].astype(np.float32)))
            ys.append(torch.from_numpy(targets(s["beats"], s["downs"], st, crop)))
        fe = torch.stack(fe).to(DEV)
        ys = torch.stack(ys).to(DEV)
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(net(fe), ys, pos_weight=pw)
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"  step {step:5d}  loss={loss.item():.4f}", flush=True)

    net.eval()
    lw = torch.softmax(net.layer_logits.detach(), 0).cpu().numpy()
    print("  layer-merge softmax weights (0..12): " + " ".join(f"{v:.3f}" for v in lw))
    print(f"  argmax layer = {int(lw.argmax())}")

    # ------------------------------------------------------------- activations
    @torch.no_grad()
    def activations(song):
        f = torch.from_numpy(song["feats"].astype(np.float32)).unsqueeze(0).to(DEV)
        return torch.sigmoid(net(f))[0].cpu().numpy()          # [T,2]

    banner("(2) STANDALONE PEAK-PICK PROBE ON THE EVAL FOLD  (fold 0, never trained on)")
    rows = []
    act_ev = {}
    for s in ev:
        T = s["feats"].shape[1]
        ref = s["beats"][s["beats"] < T / FPS]
        dref = s["downs"][s["downs"] < T / FPS]
        A = activations(s)
        act_ev[s["stem"]] = A
        if len(ref) < 3:
            continue
        est_b = beats_from_activation(A[:, 0], FPS, thr=0.5, min_dist_sec=0.15)
        est_d = beats_from_activation(A[:, 1], FPS, thr=0.5, min_dist_sec=0.30)
        bf0, bfb = blind_grid_controls(ref, T, len(est_b))
        df0, dfb = (blind_grid_controls(dref, T, len(est_d))
                    if len(dref) >= 2 else (float("nan"), float("nan")))
        rows.append(dict(
            stem=s["stem"], dataset=s["dataset"], T=T,
            beat_F=f_measure(ref, est_b),
            downbeat_F=f_measure(dref, est_d) if len(dref) >= 2 else float("nan"),
            n_est=len(est_b), n_true=len(ref),
            n_est_db=len(est_d), n_true_db=len(dref),
            blind_same_density=bf0, blind_best_offset=bfb,
            blind_db_same=df0, blind_db_best=dfb,
            metronome_F=f_measure(ref, metronome(T, FPS)),
            meter=_estimate_meter(ref, dref)))

    def M(k):
        v = [r[k] for r in rows if not (isinstance(r[k], float) and math.isnan(r[k]))]
        return float(np.mean(v)) if v else float("nan")

    n_ratio = sum(r["n_est"] for r in rows) / max(sum(r["n_true"] for r in rows), 1)
    n_ratio_db = sum(r["n_est_db"] for r in rows) / max(sum(r["n_true_db"] for r in rows), 1)
    print(f"  N songs scored              : {len(rows)}")
    print(f"  beat_F      (peak-pick)     : {M('beat_F'):.4f}")
    print(f"  downbeat_F  (peak-pick)     : {M('downbeat_F'):.4f}")
    print(f"  n_est/n_true beats          : {n_ratio:.3f}")
    print(f"  n_est/n_true downbeats      : {n_ratio_db:.3f}")
    print(f"  blind grid @ same density   : {M('blind_same_density'):.4f}")
    print(f"  blind grid best-of-12-offset: {M('blind_best_offset'):.4f}")
    print(f"  MARGIN over blind best      : {M('beat_F') - M('blind_best_offset'):+.4f}")
    print(f"  blind db grid best-of-12    : {M('blind_db_best'):.4f}")
    print(f"  MARGIN downbeat over blind  : {M('downbeat_F') - M('blind_db_best'):+.4f}")
    print(f"  120-BPM metronome floor     : {M('metronome_F'):.4f}")
    ds = {}
    for r in rows:
        ds.setdefault(r["dataset"], []).append(r["beat_F"])
    print("  per-dataset beat_F: " +
          "  ".join(f"{k}={np.mean(v):.3f}(n={len(v)})" for k, v in sorted(ds.items())))

    # ------------------------------------------------------------- sanity
    banner("(4) BEAT-SALIENCE SANITY  (activation vs TRUE bar phase / beat period)")
    g = torch.Generator().manual_seed(0)
    P_rand = (torch.randn(768, 32, generator=g) / math.sqrt(768)).numpy()

    san = []
    for s in ev:
        T = s["feats"].shape[1]
        ref = s["beats"][s["beats"] < T / FPS]
        dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 4 or len(dref) < 3:
            continue
        A = act_ev[s["stem"]]
        phi = ideal_barphase(dref, T, FPS, mode="extrap")
        if phi is None:
            continue
        m = _estimate_meter(ref, dref)
        psi = (m * phi) % TWO_PI                       # true BEAT phase
        ibi = float(np.median(np.diff(ref))) * FPS     # beat period in frames

        # old target for contrast: random 32-d projection of layer-mean MERT
        h = s["feats"].astype(np.float32).mean(0)      # [T,768]
        o = h @ P_rand
        o = (o - o.mean(0)) / (o.std(0) + 1e-6)
        o0 = o[:, 0]

        idx = np.clip(np.round(ref * FPS).astype(int), 0, T - 1)
        mask = np.zeros(T, bool); mask[idx] = True
        con = float(A[mask, 0].mean() / max(A[~mask, 0].mean(), 1e-9))
        didx = np.clip(np.round(dref * FPS).astype(int), 0, T - 1)
        dmask = np.zeros(T, bool); dmask[didx] = True
        dcon = float(A[dmask, 1].mean() / max(A[~dmask, 1].mean(), 1e-9))

        Rb, _ = phase_lock(A[:, 0], psi)
        Rd, _ = phase_lock(A[:, 1], phi)
        san.append(dict(
            ibi_frames=ibi,
            corr_beat_vs_barphase=circ_lin_corr(A[:, 0], phi),
            corr_beat_vs_beatphase=circ_lin_corr(A[:, 0], psi),
            corr_db_vs_barphase=circ_lin_corr(A[:, 1], phi),
            corr_randproj_vs_barphase=circ_lin_corr(o0, phi),
            corr_randproj_vs_beatphase=circ_lin_corr(o0, psi),
            lock_beat=Rb, lock_db=Rd,
            lock_randproj=phase_lock(np.abs(o0), psi)[0],
            ac_lag_beat=ac_peak_lag(A[:, 0]),
            ac_lag_db=ac_peak_lag(A[:, 1], lo=20, hi=400),
            ac_lag_randproj=ac_peak_lag(o0),
            bar_frames=float(np.median(np.diff(dref))) * FPS,
            contrast_beat=con, contrast_db=dcon))

    def S(k):
        v = [r[k] for r in san if not (isinstance(r[k], float) and math.isnan(r[k]))]
        return float(np.mean(v)) if v else float("nan")

    def Smed(k):
        v = [r[k] for r in san if not (isinstance(r[k], float) and math.isnan(r[k]))]
        return float(np.median(v)) if v else float("nan")

    print(f"  songs used                              : {len(san)}")
    print(f"  true beat period (frames, median)       : {Smed('ibi_frames'):.2f}")
    print(f"  true bar  period (frames, median)       : {Smed('bar_frames'):.2f}")
    print()
    print("  -- circular-linear correlation (Mardia, 0 = none) --")
    print(f"  HEAD beat-act  vs true BAR  phase       : {S('corr_beat_vs_barphase'):.4f}")
    print(f"  HEAD beat-act  vs true BEAT phase       : {S('corr_beat_vs_beatphase'):.4f}")
    print(f"  HEAD db-act    vs true BAR  phase       : {S('corr_db_vs_barphase'):.4f}")
    print(f"  OLD rand-proj  vs true BAR  phase       : {S('corr_randproj_vs_barphase'):.4f}")
    print(f"  OLD rand-proj  vs true BEAT phase       : {S('corr_randproj_vs_beatphase'):.4f}")
    print()
    print("  -- activation-weighted phase locking |R| (0 = uniform in phase) --")
    print(f"  HEAD beat-act  at BEAT phase            : {S('lock_beat'):.4f}")
    print(f"  HEAD db-act    at BAR  phase            : {S('lock_db'):.4f}")
    print(f"  OLD rand-proj  at BEAT phase            : {S('lock_randproj'):.4f}")
    print()
    print("  -- autocorrelation peak lag (frames) --")
    print(f"  HEAD beat-act peak lag / true beat per. : {Smed('ac_lag_beat'):.2f} / {Smed('ibi_frames'):.2f}")
    print(f"  HEAD db-act   peak lag / true bar  per. : {Smed('ac_lag_db'):.2f} / {Smed('bar_frames'):.2f}")
    print(f"  OLD rand-proj peak lag / true beat per. : {Smed('ac_lag_randproj'):.2f} / {Smed('ibi_frames'):.2f}")
    lagr = [r["ac_lag_beat"] / r["ibi_frames"] for r in san if not math.isnan(r["ac_lag_beat"])]
    close = float(np.mean([abs(x - 1.0) < 0.10 for x in lagr]))
    close2 = float(np.mean([min(abs(x - 1.0), abs(x - 0.5), abs(x - 2.0)) < 0.10 for x in lagr]))
    print(f"  frac songs peak lag within 10% of beat period       : {close:.3f}")
    print(f"  frac within 10% of beat period or 1/2x / 2x thereof : {close2:.3f}")
    print()
    print("  -- likelihood contrast (mean activation on-beat / off-beat) --")
    print(f"  HEAD beat channel contrast              : {S('contrast_beat'):.3f}")
    print(f"  HEAD downbeat channel contrast          : {S('contrast_db'):.3f}")

    # ------------------------------------------------------------- save
    banner("(3) FREEZE + SAVE")
    ckpt = dict(layer_logits=net.layer_logits.detach().cpu(),
                state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
                config=CONFIG)
    torch.save(ckpt, f"{GROUND}/act_head_shared.pt")
    print(f"  saved head  -> {GROUND}/act_head_shared.pt")

    for split, songs, cache, path in [
            ("eval", ev, act_ev, f"{ARMS}/act_eval.npz"),
            ("train", tr, None, f"{ARMS}/act_train.npz")]:
        out = {}
        stems = []
        for s in songs:
            A = cache[s["stem"]] if cache is not None else activations(s)
            stems.append(s["stem"])
            out[f"{s['stem']}|act"] = A.astype(np.float16)
            out[f"{s['stem']}|beats"] = s["beats"].astype(np.float32)
            out[f"{s['stem']}|downs"] = s["downs"].astype(np.float32)
            out[f"{s['stem']}|fps"] = np.float32(FPS)
            out[f"{s['stem']}|dataset"] = np.array(s["dataset"])
        out["__stems__"] = np.array(sorted(stems))
        np.savez_compressed(path, **out)
        print(f"  saved {split} activations ({len(songs)} songs) -> {path}")

    json.dump(dict(rows=rows, sanity=san,
                   summary=dict(beat_F=M("beat_F"), downbeat_F=M("downbeat_F"),
                                n_ratio=n_ratio, n_ratio_db=n_ratio_db,
                                blind_same_density=M("blind_same_density"),
                                blind_best_offset=M("blind_best_offset"),
                                blind_db_best=M("blind_db_best"),
                                metronome=M("metronome_F"),
                                corr_beat_vs_barphase=S("corr_beat_vs_barphase"),
                                corr_beat_vs_beatphase=S("corr_beat_vs_beatphase"),
                                corr_randproj_vs_barphase=S("corr_randproj_vs_barphase"),
                                lock_beat=S("lock_beat"), lock_db=S("lock_db"),
                                lock_randproj=S("lock_randproj"),
                                ac_lag_beat=Smed("ac_lag_beat"),
                                ibi_frames=Smed("ibi_frames"),
                                ac_lag_randproj=Smed("ac_lag_randproj"),
                                contrast_beat=S("contrast_beat"),
                                contrast_db=S("contrast_db"),
                                layer_weights=lw.tolist())),
              open(f"{ARMS}/act_head_report.json", "w"), indent=1, default=float)
    print(f"  saved report -> {ARMS}/act_head_report.json")


if __name__ == "__main__":
    main()
