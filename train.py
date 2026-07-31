"""Fold-honest Stage-0 training + evaluation on the real corpora.

Protocol: 8-fold cross-validation over the CV-eligible datasets; every crop is predicted
by the model that held ITS fold out; balanced accuracy is computed once over the pooled
out-of-fold predictions and reported per dataset (never pooled across datasets for meter
claims — some corpora are 100% 4/4 and carry no meter signal). gtzan has no CV folds and
is test-only: scored with a model trained on all CV crops.

Thin CLI over vbpm.fitting + vbpm.data — no protocol logic lives here.

Usage:
    /disk4/anaconda3/envs/vbpm/bin/python train.py [--datasets asap ...] [--smoke]
"""
import argparse

from vbpm.data import VALUES, load_crops, to_prob
from vbpm.fitting import (cv_out_of_fold, fit_vectorized, predict_m, score,
                          score_per_dataset, verify_vectorized)
from vbpm.metrics import majority_predict, peak_count_estimate
from vbpm.reducers import REDUCERS
from vbpm.stage0 import Stage0


def main(argv=None):
    """Fold-honest CV + per-dataset report over the real corpora, one run per reducer."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true", help="6 songs per fold, quick wiring check")
    ap.add_argument("--reducers", nargs="*", default=["meanmax", "peaks"],
                    choices=sorted(REDUCERS),
                    help="reducer variants (see vbpm/reducers.py), one CV run each")
    args = ap.parse_args(argv)

    print("loading crops (live fold-honest frontend pass)...", flush=True)
    crops, report = load_crops(datasets=args.datasets, device=args.device,
                               limit_per_fold=6 if args.smoke else None)
    print(f"usable: {report['usable']}  rejects: {report['rejects']}  "
          f"unmatched downbeats: {report['unmatched_downbeats']}")
    for dataset, class_counts in sorted(report["class_counts_by_dataset"].items()):
        counts = "  ".join(f"m={m}:{class_counts.get(m, 0)}" for m in VALUES)
        print(f"  {dataset:12s} {counts}")

    cv = [c for c in crops if c["fold"] is not None]
    test = [c for c in crops if c["fold"] is None]

    for name in args.reducers:
        reducer, s_dim = REDUCERS[name]
        print(f"\n######## reducer: {name} (s_dim={s_dim}) ########")
        verify_vectorized(crops, VALUES, reducer, s_dim)

        def fit_fn(train_crops):
            return fit_vectorized(Stage0(VALUES, reducer=reducer, s_dim=s_dim),
                                  train_crops, steps=args.steps, lr=args.lr)

        pooled_crops, pooled_preds, test_preds = cv_out_of_fold(
            cv, test, fit_fn, lambda model, cs: [predict_m(model, c["h"]) for c in cs])

        print("\n== pooled out-of-fold (CV datasets), per dataset ==")
        score_per_dataset(pooled_crops, pooled_preds, VALUES)
        if test:
            print("\n== test-only (gtzan), model trained on all CV crops ==")
            score("gtzan", test, test_preds, VALUES)

    print("\n######## baselines (held-out, deployable) ########")
    majority = majority_predict([c["m_true"] for c in cv], VALUES)
    score("majority", cv, [majority] * len(cv), VALUES)
    peak_preds = [peak_count_estimate(to_prob(c["h"]), VALUES) for c in cv]
    score("peak-count", cv, peak_preds, VALUES)
    if test:
        peak_test = [peak_count_estimate(to_prob(c["h"]), VALUES) for c in test]
        score("peak-gtzan", test, peak_test, VALUES)


if __name__ == "__main__":
    main()
