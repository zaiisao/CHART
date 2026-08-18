"""Where we lose to Beat This, and why.

Reads compare_beatthis.py's table and sorts the deficit into error modes by
the only evidence available without labels at run time: the count ratio and
the inferred bar period against the reference one.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np


def classify(r):
    """Name the failure mode for one song from its counts and periods."""
    ratio, ref, est = r["ratio_ours"], r["ref_bar"], r["est_bar"]
    rel = est / ref if ref > 0 else float("nan")
    for name, lo, hi in (("half (2x slow)", 1.7, 2.4), ("double (2x fast)", 0.42, 0.58),
                         ("third", 2.6, 3.5), ("triple", 0.28, 0.40)):
        if lo <= rel <= hi:
            return name
    if abs(rel - 1) <= 0.10:
        return "period right, phase wrong"
    if ratio < 0.5:
        return "under-emits"
    return "period off (non-integer)"


def main():
    """Print the deficit table grouped by mode."""
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/tmp/scratch/compare_bt.csv")
    p.add_argument("--worst", type=int, default=25)
    args = p.parse_args()

    with open(args.csv) as fh:
        rows = [{k: (v if k == "song" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(fh)]
    for r in rows:
        r["mode"] = classify(r)

    lose = sorted([r for r in rows if r["delta"] < -0.05], key=lambda r: r["delta"])
    win = [r for r in rows if r["delta"] > 0.05]
    print(f"n={len(rows)}  ours {np.mean([r['f_ours'] for r in rows]):.4f}  "
          f"BT {np.mean([r['f_bt'] for r in rows]):.4f}")
    print(f"lose {len(lose)}  win {len(win)}  tie {len(rows) - len(lose) - len(win)}")
    gap = -sum(r["delta"] for r in lose) / len(rows)
    print(f"deficit from losing songs: {gap:.4f} F  (recovering all of it -> "
          f"{np.mean([r['f_ours'] for r in rows]) + gap:.4f})\n")

    print(f"{'mode':>26} {'n':>4} {'mean d':>7} {'F pool':>7} {'BT':>6}")
    for mode in sorted({r["mode"] for r in lose}):
        g = [r for r in lose if r["mode"] == mode]
        print(f"{mode:>26} {len(g):4d} {np.mean([r['delta'] for r in g]):+7.3f} "
              f"{np.mean([r['f_ours'] for r in g]):7.3f} "
              f"{np.mean([r['f_bt'] for r in g]):6.3f}")

    print(f"\nworst {args.worst}:")
    print(f"{'song':>22} {'ours':>6} {'BT':>6} {'delta':>7} {'ref s':>6} {'est s':>6} "
          f"{'rel':>5}  mode")
    for r in lose[:args.worst]:
        print(f"{r['song']:>22} {r['f_ours']:6.3f} {r['f_bt']:6.3f} {r['delta']:+7.3f} "
              f"{r['ref_bar']:6.2f} {r['est_bar']:6.2f} "
              f"{r['est_bar'] / r['ref_bar']:5.2f}  {r['mode']}")

    print("\nby genre:")
    for gen in sorted({r["song"].split(":")[1].split("_")[1] for r in rows
                       if len(r["song"].split(":")[1].split("_")) > 1}):
        g = [r for r in rows if f"_{gen}_" in r["song"]]
        if not g:
            continue
        print(f"  {gen:>12} n={len(g):3d}  ours {np.mean([r['f_ours'] for r in g]):.3f}  "
              f"BT {np.mean([r['f_bt'] for r in g]):.3f}  "
              f"delta {np.mean([r['delta'] for r in g]):+.3f}")


if __name__ == "__main__":
    main()
