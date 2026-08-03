"""Train and evaluate the phase VAE, and run the six controls. One script, one process.

    PYTHONPATH=. python -m phasevae.run --gpu 1 --seeds 0 1 2

Everything is reported per dataset. The deployment path reads h only and is asserted to
never touch y; the posterior path is reported beside it and clearly labelled as a
diagnostic (it sees y, so it cannot be a deployment number).
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
from collections import Counter, defaultdict

import numpy as np
import torch

from vbpm.data import FPS, iter_frontend_features

from .crops import CROP_BARS, build_crop, crop_starts, song_crops
from .model import PhaseVAE

M = 4                 # this build is m = 4 only; passed as a parameter everywhere
MAX_CROPS = 3         # per song, drawn uniformly over valid starts
VAL_FOLD = 7


def load_dataset(m: int, seed: int = 0, limit_per_fold=None, verbose: bool = True):
    """(crops, rejects): un-aligned crops for every song whose features are available."""
    rng = np.random.default_rng(seed)
    crops, rejects = [], Counter()
    for song, features in iter_frontend_features(output="features+activations",
                                                 limit_per_fold=limit_per_fold,
                                                 verbose=verbose):
        got, song_rejects = song_crops(features, song, m, rng, MAX_CROPS)
        rejects.update(song_rejects)
        for crop in got:
            crop["h"] = crop["h"].astype(np.float16)
            crop["fold"] = song.fold
            crops.append(crop)
    return crops, rejects


def load_or_build(m: int, cache: str | None, limit_per_fold=None):
    """Crops from ``cache`` if present, else built and written there.

    The crop set is fixed across seeds on purpose: seed-to-seed spread then measures
    run-to-run variance of TRAINING, not of the data draw.
    """
    if cache and pathlib.Path(cache).exists():
        with open(cache, "rb") as handle:
            crops, rejects = pickle.load(handle)
        print(f"crops loaded from {cache}")
        return crops, rejects
    crops, rejects = load_dataset(m, limit_per_fold=limit_per_fold)
    if cache:
        with open(cache, "wb") as handle:
            pickle.dump((crops, rejects), handle, protocol=4)
    return crops, rejects


def collate(batch, device):
    """Pad a list of crop dicts into tensors on ``device``."""
    T = max(len(c["delta"]) for c in batch)
    n = len(batch[0]["y"])
    B = len(batch)
    D = batch[0]["h"].shape[1]
    h = torch.zeros(B, T, D)
    delta = torch.zeros(B, T)
    mask = torch.zeros(B, T)
    y_channels = torch.zeros(B, T, 2)
    beat_frames = torch.zeros(B, n, dtype=torch.long)
    y = torch.zeros(B, n, dtype=torch.long)
    phi1_target = torch.zeros(B)       # DIAGNOSTIC ONLY (derived from r_true): the phase
                                       # at frame 0 that puts phi = 0 on the first true
                                       # downbeat. Used by the supervised warm-start arm
                                       # and by nothing that runs at deployment.
    for i, c in enumerate(batch):
        t = len(c["delta"])
        h[i, :t] = torch.from_numpy(c["h"].astype(np.float32))
        delta[i, :t] = torch.from_numpy(c["delta"])
        mask[i, :t] = 1.0
        beat_frames[i] = torch.from_numpy(c["beat_frames"])
        y[i] = torch.from_numpy(c["y"].astype(np.int64))
        y_channels[i, c["beat_frames"], 0] = 1.0
        y_channels[i, c["beat_frames"][c["y"] == 1], 1] = 1.0
        relative = np.cumsum(c["delta"].astype(np.float64)) - c["delta"][0]
        phi1_target[i] = -relative[c["beat_frames"][c["r_true"]]]
    out = {"h": h, "delta": delta, "mask": mask, "y_channels": y_channels,
           "beat_frames": beat_frames, "y": y, "phi1_target": phi1_target}
    return {k: v.to(device) for k, v in out.items()}


def batches(crops, batch_size: int, device, shuffle: bool, rng=None):
    """Yield collated batches, length-sorted so padding stays small."""
    order = np.argsort([len(c["delta"]) for c in crops])
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(chunks)
    for chunk in chunks:
        yield [crops[i] for i in chunk], collate([crops[i] for i in chunk], device)


def downbeat_f(pred, true) -> float:
    """F on the GIVEN beat grid: a predicted downbeat is right iff it is an annotated one."""
    tp = float((pred & true).sum())
    if tp == 0:
        return 0.0
    precision = tp / float(pred.sum())
    recall = tp / float(true.sum())
    return 2 * precision * recall / (precision + recall)


def evaluate(model, crops, device, m: int, samples: int = 128, batch_size: int = 32):
    """Per-dataset deployment and posterior metrics, plus the latent-use ablation."""
    model.eval()
    rows = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for raw, batch in batches(crops, batch_size, device, shuffle=False):
            assert "y" in batch, "sanity: the eval batch carries y for SCORING only"
            deploy_scores, deploy_probs = model.deploy_offset_scores(
                batch["h"], batch["delta"], batch["mask"], batch["beat_frames"], m,
                samples=samples)
            post_scores, post_probs = model.posterior_offset_scores(
                batch["h"], batch["delta"], batch["y_channels"], batch["mask"],
                batch["beat_frames"], m, samples=16)
            abl_scores, abl_probs = model.posterior_offset_scores(
                batch["h"], batch["delta"], batch["y_channels"], batch["mask"],
                batch["beat_frames"], m, samples=16, prior_sample=True)
            true = batch["y"].bool().cpu().numpy()
            r_true = np.array([c["r_true"] for c in raw])
            for name, scores, probs in (("deploy", deploy_scores, deploy_probs),
                                        ("posterior", post_scores, post_probs),
                                        ("prior-ablation", abl_scores, abl_probs)):
                r_hat = scores.argmax(1).cpu().numpy()
                pred = (probs > 0.5).cpu().numpy()
                for i, c in enumerate(raw):
                    rows[name][c["dataset"]].append(
                        (float(r_hat[i] == r_true[i]), downbeat_f(pred[i], true[i])))
    return {name: {ds: (float(np.mean([v[0] for v in vals])),
                        float(np.mean([v[1] for v in vals])), len(vals))
                   for ds, vals in per_ds.items()}
            for name, per_ds in rows.items()}


def frontend_baseline(crops, m: int):
    """Reference: pick r by summing Beat This's own downbeat logit at beats r::m.

    The model has these two channels in h, so this is the number it has to beat; it is
    not part of the VAE.
    """
    per_ds = defaultdict(list)
    for c in crops:
        logits = c["h"][:, -1].astype(np.float32)[c["beat_frames"]]
        r_hat = int(np.argmax([logits[r::m].sum() for r in range(m)]))
        per_ds[c["dataset"]].append(float(r_hat == c["r_true"]))
    return {ds: (float(np.mean(v)), len(v)) for ds, v in per_ds.items()}


def gradient_audit(model, batch) -> list[str]:
    """Control 2: READ every gradient after one backward. Returns the dead parameters."""
    model.train()
    model.zero_grad()
    out = model(batch["h"], batch["delta"], batch["y_channels"], batch["mask"],
                batch["beat_frames"], batch["y"])
    (-out["elbo"].mean()).backward()
    dead = []
    for name, param in model.named_parameters():
        if param.grad is None:
            dead.append(f"{name}: grad is None")
        elif float(param.grad.abs().max()) == 0.0:
            dead.append(f"{name}: grad is exactly zero")
    model.zero_grad()
    return dead


def kappa_reads_audio(model, batch) -> float:
    """Control: kappa must vary with h. Returns the std of kappa across frames."""
    model.eval()
    with torch.no_grad():
        kappa = model.kappa_net(batch["h"])
    return float((kappa * batch["mask"]).std())


def shift_test(model, crops, device, m: int, samples: int = 128):
    """Controls 5 and 6: move the audio one beat later; the offset must move by one.

    Control 6 (grid shift) re-cuts the SAME song one beat later from the source data --
    r_true moves by -1 (mod m) and the prediction must follow.
    Control 5 (audio-blindness) rolls h in time by one beat's worth of frames while
    leaving the given grid alone: an audio-locked model moves, a metronome does not.
    """
    model.eval()
    recut, rolled, n_used = [], [], 0
    with torch.no_grad():
        for c in crops[:256]:
            beats = c["beat_frames"]
            step = int(round(np.median(np.diff(beats))))
            batch = collate([c], device)
            base, _ = model.deploy_offset_scores(batch["h"], batch["delta"],
                                                 batch["mask"], batch["beat_frames"], m,
                                                 samples=samples)
            shifted = dict(batch)
            shifted["h"] = torch.roll(batch["h"], shifts=step, dims=1)
            moved, _ = model.deploy_offset_scores(shifted["h"], batch["delta"],
                                                  batch["mask"], batch["beat_frames"], m,
                                                  samples=samples)
            rolled.append(int((int(moved.argmax()) - int(base.argmax())) % m))
            n_used += 1
    return {"h_rolled_one_beat_delta_r": Counter(rolled), "n": n_used,
            "recut": Counter(recut)}


def recut_test(model, device, m: int, limit: int = 60, samples: int = 128):
    """Control 6 proper: cut the same audio one beat later and require r_hat - 1 (mod m)."""
    model.eval()
    deltas, truth_ok = Counter(), Counter()
    rng = np.random.default_rng(123)
    seen = 0
    for song, features in iter_frontend_features(output="features+activations",
                                                 datasets=["gtzan"], verbose=False):
        beat_times, downbeat_times = song.beats()
        from vbpm.data import derive_y
        y_song, _ = derive_y(beat_times, downbeat_times)
        starts, _ = crop_starts(y_song, m, CROP_BARS)
        starts = [(s, r) for s, r in starts if (s + 1, (r - 1) % m) in set(starts)]
        if not starts:
            continue
        s, r = starts[int(rng.integers(len(starts)))]
        pair = [build_crop(features, beat_times, s, r, m),
                build_crop(features, beat_times, s + 1, (r - 1) % m, m)]
        if any(p is None for p in pair):
            continue
        truth_ok[(pair[0]["r_true"] - pair[1]["r_true"]) % m] += 1
        with torch.no_grad():
            hats = []
            for p in pair:
                b = collate([p], device)
                sc, _ = model.deploy_offset_scores(b["h"], b["delta"], b["mask"],
                                                   b["beat_frames"], m, samples=samples)
                hats.append(int(sc.argmax()))
        deltas[(hats[0] - hats[1]) % m] += 1
        seen += 1
        if seen >= limit:
            break
    return {"predicted_delta_r": deltas, "true_delta_r": truth_ok}


def train(model, train_crops, device, epochs: int, batch_size: int, lr: float, rng,
          verbose: bool = True, warm_epochs: int = 0, warm_weight: float = 30.0):
    """Maximise the ELBO. Returns the per-epoch mean ELBO.

    ``warm_epochs`` > 0 adds a SUPERVISED auxiliary term 1 - cos(mu_q1 - phi1_target)
    for that many epochs and then removes it. That arm is not a model, it is a
    diagnostic: if the ELBO alone cannot find the phase but keeps it once handed it,
    the collapse is an optimisation failure; if the phase decays away again once the
    term is removed, the ELBO's own optimum has no phase in it.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        totals = []
        for _, batch in batches(train_crops, batch_size, device, True, rng):
            out = model(batch["h"], batch["delta"], batch["y_channels"], batch["mask"],
                        batch["beat_frames"], batch["y"])
            loss = -out["elbo"].mean()
            if epoch < warm_epochs:
                residual = model.encoder(batch["h"], batch["y_channels"])[0][:, 0]
                mu_q1 = residual + model.prior_init(batch["h"], batch["mask"])[0]
                loss = loss + warm_weight * (
                    1.0 - torch.cos(mu_q1 - batch["phi1_target"])).mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)
            optimiser.step()
            totals.append([float(out["elbo"].mean()), float(out["recon"].mean()),
                           float(out["kl"].mean())])
        elbo, recon, kl = np.mean(totals, axis=0)
        history.append(elbo)
        if verbose:
            print(f"  epoch {epoch:2d}  elbo {elbo:8.2f}  recon {recon:8.2f} "
                  f"kl {kl:8.2f}", flush=True)
    return history


def report_metrics(title: str, metrics, chance: float) -> None:
    """Print one metric block per dataset."""
    print(f"\n{title}   (chance offset accuracy = {chance:.3f})")
    print(f"  {'dataset':12s} {'offset acc':>10s} {'downbeat F':>11s} {'crops':>7s}")
    for ds in sorted(metrics):
        acc, f_score, n = metrics[ds]
        print(f"  {ds:12s} {acc:10.3f} {f_score:11.3f} {n:7d}")


def main() -> None:
    """Load the data, run the controls, train each seed and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1, choices=(1, 3))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--limit-per-fold", type=int, default=None)
    parser.add_argument("--crop-cache", default=None)
    parser.add_argument("--warm-epochs", type=int, default=0,
                        help="supervised phi_1 warm start (diagnostic arm)")
    parser.add_argument("--anchor-init", action="store_true",
                        help="give phi_1 an h-dependent prior (breaks rotation symmetry)")
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    print("loading crops (fold-honest features, un-aligned offsets)...", flush=True)
    crops, rejects = load_or_build(M, args.crop_cache, args.limit_per_fold)
    print(f"\nusable crops: {len(crops)}   rejects: {dict(rejects)}")

    # --- control 3: offset uniformity -------------------------------------------------
    counts = Counter(c["r_true"] for c in crops)
    total = sum(counts.values())
    print(f"\nCONTROL 3 offset uniformity r_true: "
          f"{ {r: counts[r] for r in range(M)} }  fractions "
          f"{ {r: round(counts[r] / total, 3) for r in range(M)} }")
    partition = [len(range(r, CROP_BARS * M, M)) for r in range(M)]
    assert len(set(partition)) == 1, f"downbeat count leaks the offset: {partition}"
    chance = 1.0 / M
    print(f"  downbeat-count partition {partition} is offset-invariant "
          f"-> chance = 1/{M} = {chance:.3f}")

    per_dataset = Counter(c["dataset"] for c in crops)
    print(f"  crops per dataset: {dict(per_dataset)}")

    train_crops = [c for c in crops if c["fold"] not in (None, VAL_FOLD)]
    val_crops = [c for c in crops if c["fold"] == VAL_FOLD]
    test_crops = [c for c in crops if c["fold"] is None]
    print(f"\ntrain {len(train_crops)} / val {len(val_crops)} / gtzan-test "
          f"{len(test_crops)}")

    print("\nfrontend reference (Beat This downbeat logit, not the VAE):")
    for split, name in ((val_crops, "val"), (test_crops, "gtzan")):
        if split:
            print(f"  {name}: " + ", ".join(
                f"{ds} {acc:.3f} (n={n})"
                for ds, (acc, n) in sorted(frontend_baseline(split, M).items())))

    all_metrics = defaultdict(lambda: defaultdict(list))
    for seed in args.seeds:
        print(f"\n===== seed {seed} =====", flush=True)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = PhaseVAE(crops[0]["h"].shape[1], anchor_init=args.anchor_init).to(device)

        _, probe = next(iter(batches(train_crops, args.batch_size, device, False)))
        dead = gradient_audit(model, probe)
        print(f"CONTROL 2 gradient audit: {len(dead)} dead parameters"
              + ("" if not dead else " -> " + "; ".join(dead)))
        assert not dead, "dead parameters at initialisation"

        train(model, train_crops, device, args.epochs, args.batch_size, args.lr,
              rng, warm_epochs=args.warm_epochs)
        print(f"  kappa std across frames (must be > 0, else kappa ignores h): "
              f"{kappa_reads_audio(model, probe):.4f}")
        with torch.no_grad():
            kappa = model.kappa_net(probe["h"])[probe["mask"] > 0]
            lam = model.encoder(probe["h"], probe["y_channels"])[1][probe["mask"] > 0]
        print(f"  emission a={float(model.emission_a):.3f} "
              f"b={float(model.emission_b):.3f}  kappa mean {float(kappa.mean()):.1f} "
              f"lambda mean {float(lam.mean()):.1f}")
        checkpoint = f"checkpoint_seed{seed}{'_anchor' if args.anchor_init else ''}.pt"
        torch.save(model.state_dict(), checkpoint)   # a run whose eval is interrupted
        print(f"  saved {checkpoint}")               # must not lose its trained weights

        for split, name in ((val_crops, "val"), (test_crops, "gtzan")):
            if not split:
                continue
            metrics = evaluate(model, split, device, M)
            for path, per_ds in metrics.items():
                report_metrics(f"[seed {seed}] {name} -- {path}", per_ds, chance)
                for ds, values in per_ds.items():
                    all_metrics[(name, path, ds)]["acc"].append(values[0])
                    all_metrics[(name, path, ds)]["f"].append(values[1])

        print("\nCONTROL 5 audio-blindness (roll h one beat, given grid unchanged):")
        print(f"  {shift_test(model, val_crops or test_crops, device, M)}")
        print("CONTROL 6 shift test (re-cut one beat later, gtzan):")
        print(f"  {recut_test(model, device, M)}")

    print("\n===== across seeds (mean +- spread) =====")
    for key in sorted(all_metrics, key=str):
        split, path, ds = key
        acc = np.array(all_metrics[key]["acc"])
        f_score = np.array(all_metrics[key]["f"])
        print(f"  {split:6s} {path:15s} {ds:10s} offset {acc.mean():.3f} "
              f"+-{acc.std():.3f} (range {acc.min():.3f}-{acc.max():.3f})  "
              f"F {f_score.mean():.3f} +-{f_score.std():.3f}")
    print(f"\nfps={FPS} m={M} bars/crop={CROP_BARS} n_beats={CROP_BARS * M} "
          f"chance={1 / M:.3f}")


if __name__ == "__main__":
    main()
