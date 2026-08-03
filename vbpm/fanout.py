"""Run a campaign's (arm, fold) fits in parallel: build crops once, fan the fits out.

The serial protocol in ``cv_out_of_fold`` fits arms x folds one after another, and every
experiment process rebuilds the crop set from scratch. Both are pure waste: the fits are
mutually independent (each trains on its own fold complement) and the crop set is
identical for all of them. This module keeps the arithmetic bit-identical -- the same
``fit``/``predict`` calls, the same seed, the same train/held partition -- and changes
only WHEN and WHERE each call happens.

Two pieces:

    store    the crop set, built ONCE by the certified ``load_crops`` path and written as
             memmapped arrays. Workers map it read-only, so N concurrent workers share one
             copy in the page cache instead of holding N copies in RAM.
    jobs     one (arm, job) pair per fit; ``job`` is a CV fold id or "test" (train on all
             CV crops, predict the test-only split). Scheduled over GPU slots, collected,
             then scored by the same ``score_per_dataset``/``score`` as the serial path.

A campaign module supplies the experiment-specific half:

    OUTPUT       frontend output string for load_crops
    SEQ_FIELDS   {field: width} -- per-frame arrays, memmapped
    FLAT_FIELDS  [field] -- fixed-size per-crop arrays, held in the meta pickle
    ARMS         ordered mapping of arm name -> whatever the campaign needs
    make_entry(song, crop, h_crop, t0) -> dict
    fit(arm, train_crops, device) -> fitted
    predict(arm, fitted, crops, device) -> list of beats-per-bar counts
"""
from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import pickle
import subprocess
import sys
import time

import numpy as np

from vbpm.data import VALUES, load_crops
from vbpm.fitting import score, score_per_dataset

TEST_JOB = "test"


# ---- store ----------------------------------------------------------------------------
def build_store(campaign, store_dir, limit_per_fold=None, device="cuda"):
    """Build the crop set once and write it as memmaps + a meta pickle."""
    store = pathlib.Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)
    crops, report = load_crops(limit_per_fold=limit_per_fold, device=device,
                              output=campaign.OUTPUT, make_entry=campaign.make_entry)
    print(f"crops: {len(crops)}  rejects: {report['rejects']}", flush=True)

    meta = []
    for field, width in campaign.SEQ_FIELDS.items():
        total = sum(len(c[field]) for c in crops)
        dtype = crops[0][field].dtype
        out = np.lib.format.open_memmap(store / f"{field}.npy", mode="w+",
                                        dtype=dtype, shape=(total, width))
        at = 0
        for i, crop in enumerate(crops):
            n = len(crop[field])
            out[at:at + n] = crop[field]
            if len(meta) <= i:
                meta.append({})
            meta[i][f"{field}_span"] = (at, at + n)
            at += n
        out.flush()
        del out

    for i, crop in enumerate(crops):
        for field in campaign.FLAT_FIELDS:
            meta[i][field] = crop[field]
        for field in ("m_true", "dataset", "fold"):
            meta[i][field] = crop[field]
    with open(store / "meta.pkl", "wb") as fh:
        pickle.dump({"meta": meta, "report": report,
                     "seq_fields": dict(campaign.SEQ_FIELDS)}, fh)
    return len(crops)


class _MappedCrop(dict):
    """A crop whose per-frame fields are read on demand from the shared memmap."""

    def __init__(self, entry, maps):
        super().__init__(entry)
        self._maps = maps

    def __getitem__(self, key):
        if key in self._maps:
            start, end = super().__getitem__(f"{key}_span")
            return np.asarray(self._maps[key][start:end])
        return super().__getitem__(key)


def load_store(store_dir):
    """Crop list backed by read-only memmaps; identical contents to the built crops."""
    store = pathlib.Path(store_dir)
    with open(store / "meta.pkl", "rb") as fh:
        blob = pickle.load(fh)
    maps = {field: np.load(store / f"{field}.npy", mmap_mode="r")
            for field in blob["seq_fields"]}
    return [_MappedCrop(entry, maps) for entry in blob["meta"]], blob["report"]


# ---- one job --------------------------------------------------------------------------
def split_for(crops, job):
    """(train, held) for a job -- the SAME partition cv_out_of_fold makes serially."""
    cv = [c for c in crops if c["fold"] is not None]
    if job == TEST_JOB:
        return cv, [c for c in crops if c["fold"] is None]
    fold = int(job)
    return [c for c in cv if c["fold"] != fold], [c for c in cv if c["fold"] == fold]


def run_job(campaign, crops, arm, job, device="cuda"):
    """Fit one arm on one job's train split and predict its held split."""
    train, held = split_for(crops, job)
    if not held:
        return []
    fitted = campaign.fit(arm, train, device)
    return campaign.predict(arm, fitted, held, device)


# ---- launcher -------------------------------------------------------------------------
def _jobs(crops, arms):
    folds = sorted({c["fold"] for c in crops if c["fold"] is not None})
    ids = [str(f) for f in folds] + ([TEST_JOB] if any(c["fold"] is None for c in crops)
                                     else [])
    return [(arm, job) for arm in arms for job in ids]


def _schedule(commands, slots):
    """Run commands over a fixed pool of (gpu) slots; return the failures."""
    pending, running, failed = list(commands), [], []
    while pending or running:
        while pending and len(running) < len(slots):
            gpu = [g for g in slots
                   if g not in {r[1] for r in running}][0]
            label, cmd = pending.pop(0)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            print(f"  -> [{label}] on GPU {gpu}", flush=True)
            running.append((subprocess.Popen(cmd, env=env), gpu, label))
        time.sleep(2.0)
        for entry in list(running):
            proc, _gpu, label = entry
            if proc.poll() is not None:
                running.remove(entry)
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                print(f"  <- [{label}] {status}", flush=True)
                if proc.returncode != 0:
                    failed.append(label)
    return failed


def main(argv=None):
    """Launcher, or -- with --worker -- the single fit a launched process runs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", help="module path, e.g. experiments.stage0_transformer_prior")
    ap.add_argument("--store", default=None, help="crop store dir (default: under --work)")
    ap.add_argument("--work", default="/disk4/jaehoon/vbpm_fanout")
    ap.add_argument("--gpus", nargs="*", type=int, default=[0, 1, 3])
    ap.add_argument("--per-gpu", type=int, default=1,
                    help="concurrent fits per GPU; raise only if a fit underuses the GPU")
    ap.add_argument("--arms", nargs="*", default=None, help="subset of arms to run")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the crop store")
    ap.add_argument("--limit-per-fold", type=int, default=None)
    ap.add_argument("--worker", nargs=2, metavar=("ARM", "JOB"), default=None)
    args = ap.parse_args(argv)

    campaign = importlib.import_module(args.campaign)
    work = pathlib.Path(args.work) / args.campaign.rsplit(".", 1)[-1]
    store = pathlib.Path(args.store) if args.store else work / "store"
    preds_dir = work / "preds"

    if args.worker:                                     # one fit, in its own process
        arm, job = args.worker
        crops, _ = load_store(store)
        preds = run_job(campaign, crops, arm, job)
        preds_dir.mkdir(parents=True, exist_ok=True)
        with open(preds_dir / f"{arm}.{job}.pkl", "wb") as fh:
            pickle.dump(preds, fh)
        return

    if args.rebuild or not (store / "meta.pkl").exists():
        print(f"building crop store at {store}", flush=True)
        build_store(campaign, store, limit_per_fold=args.limit_per_fold)

    crops, report = load_store(store)
    print(f"store: {len(crops)} crops  rejects: {report['rejects']}", flush=True)
    arms = args.arms or list(campaign.ARMS)
    jobs = _jobs(crops, arms)
    preds_dir.mkdir(parents=True, exist_ok=True)

    commands = [(f"{arm}.{job}",
                 [sys.executable, "-m", "vbpm.fanout", args.campaign,
                  "--work", args.work, "--store", str(store), "--worker", arm, job])
                for arm, job in jobs]
    slots = [g for g in args.gpus for _ in range(args.per_gpu)]
    print(f"{len(commands)} fits over {len(slots)} slots", flush=True)
    failed = _schedule(commands, slots)
    assert not failed, f"fits failed: {failed}"

    for arm in arms:                                    # collect and score, serially
        pooled_crops, pooled_preds, test_preds = [], [], []
        for _arm, job in jobs:
            if _arm != arm:
                continue
            with open(preds_dir / f"{arm}.{job}.pkl", "rb") as fh:
                preds = pickle.load(fh)
            _train, held = split_for(crops, job)
            if job == TEST_JOB:
                test_preds = preds
            else:
                pooled_crops += held
                pooled_preds += preds
        print(f"\n######## arm: {arm} ########")
        score_per_dataset(pooled_crops, pooled_preds, VALUES)
        if test_preds:
            score("gtzan", [c for c in crops if c["fold"] is None], test_preds, VALUES)


if __name__ == "__main__":
    main()
