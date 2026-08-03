"""Warm the feature cache across GPUs: shard the fold-honest frontend pass.

The nine checkpoint groups (fold0-7 + final0) are independent, so they can run on
different GPUs concurrently. Each worker iterates the certified pass (which writes the
cache as a side effect) for its share of groups; a subsequent experiment then loads
everything from cache in seconds.

Usage:
    python -m vbpm.warm_cache --gpus 0 1 3 --output features+activations
    python -m vbpm.warm_cache --folds 2 5 test --device cuda    (single worker; used
                                                                 internally by --gpus)
"""
import argparse
import os
import subprocess
import sys

ALL_GROUPS = ["0", "1", "2", "3", "4", "5", "6", "7", "test"]   # test = final0/gtzan


def _parse_folds(tokens):
    return [None if tok == "test" else int(tok) for tok in tokens]


def _worker(folds, output, device, override=None):
    from vbpm.data import iter_frontend_features
    n = 0
    for _s, _h in iter_frontend_features(output=output, folds=_parse_folds(folds),
                                         device=device, override_checkpoint=override):
        n += 1
    print(f"worker done: {n} songs for folds {folds}", flush=True)


def main(argv=None):
    """Spawn one worker per GPU, round-robin over checkpoint groups; wait for all."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="*", type=int, default=None,
                    help="GPU ids to shard across; omit to run as a single worker")
    ap.add_argument("--folds", nargs="*", default=ALL_GROUPS,
                    help="checkpoint groups: 0-7 and/or 'test' (final0)")
    ap.add_argument("--output", default="activations")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--override", default=None,
                    help="force every song through one checkpoint (checkpoint-swap probe "
                         "only: this is deliberately NOT fold-honest)")
    args = ap.parse_args(argv)

    if args.gpus is None:
        _worker(args.folds, args.output, args.device, args.override)
        return

    shards = {gpu: [] for gpu in args.gpus}
    for i, group in enumerate(args.folds):
        shards[args.gpus[i % len(args.gpus)]].append(group)

    procs = []
    for gpu, groups in shards.items():
        if not groups:
            continue
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        cmd = [sys.executable, "-m", "vbpm.warm_cache",
               "--folds", *groups, "--output", args.output]
        if args.override:
            cmd += ["--override", args.override]
        print(f"GPU {gpu}: groups {groups}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))

    failed = [p.args for p in procs if p.wait() != 0]
    assert not failed, f"warm-cache workers failed: {failed}"
    print("cache warm.", flush=True)


if __name__ == "__main__":
    main()
