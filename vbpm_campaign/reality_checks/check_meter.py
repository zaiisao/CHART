#!/usr/bin/env python
"""
Reality check: METER assumption (ELBO_for_DBN.md sec 3, 5.1).

Paper model: m_t is a PER-FRAME Categorical latent with a full K x K transition
matrix p(m_t | m_{t-1}, phi_t, phi_{t-1}, h) that ALLOWS meter switches at every
bar boundary. The model spends parameters (a learned transition network) on
mid-song meter change.

Test whether that per-frame switching is warranted on real beat annotations:
  (1) Meter-class histogram: how many distinct beats-per-bar (bpb), how dominant is 4?
  (2) Within-song stability: how often does bpb actually CHANGE mid-song?
  (3) Is a per-FRAME meter latent warranted, or is meter per-SONG constant?

col1 (when present) = beat-index-within-bar, 1 == downbeat. bpb of a completed
bar = number of beats from one downbeat up to (not incl.) the next downbeat.
"""
import os, glob, collections
import numpy as np

ROOT = "/home/sogang/jaehoon/VBPM/dataset_store/beat_this_annotations"
DATASETS_WITH_METER = ["ballroom", "beatles", "asap", "hjdb", "gtzan",
                       "hainsworth", "rwc"]
DATASETS_NO_METER = ["smc"]


def bar_lengths_from_file(path):
    arr = np.loadtxt(path, ndmin=2)
    if arr.shape[1] < 2:
        return [], False
    idx = arr[:, 1].astype(int)
    db = np.where(idx == 1)[0]
    if len(db) < 2:
        return [], True
    return list(np.diff(db)), True


def analyze_dataset(name):
    files = sorted(glob.glob(os.path.join(ROOT, name, "annotations", "beats", "*.beats")))
    per_song, all_bar_lengths, n_no_col1 = [], [], 0
    for f in files:
        bl, has = bar_lengths_from_file(f)
        if not has:
            n_no_col1 += 1
            continue
        if len(bl) == 0:
            continue
        all_bar_lengths.extend(bl)
        changes = int(np.sum(np.array(bl[1:]) != np.array(bl[:-1]))) if len(bl) > 1 else 0
        song_mode = collections.Counter(bl).most_common(1)[0][0]
        per_song.append(dict(
            n_bars=len(bl), n_distinct=len(set(bl)),
            changes=changes, trans=len(bl) - 1, mode=song_mode,
            frac_modal=collections.Counter(bl)[song_mode] / len(bl)))
    return per_song, all_bar_lengths, n_no_col1, len(files)


def summarize(name, per_song, all_bar_lengths):
    n_songs = len(per_song)
    if n_songs == 0:
        return None
    hist = collections.Counter(all_bar_lengths)
    total_bars = sum(hist.values())
    song_modes = collections.Counter(s["mode"] for s in per_song)
    total_changes = sum(s["changes"] for s in per_song)
    total_trans = sum(s["trans"] for s in per_song)
    return dict(
        name=name, n_songs=n_songs, total_bars=total_bars,
        hist=dict(sorted(hist.items())), frac4_bars=hist.get(4, 0) / total_bars,
        song_modes=dict(sorted(song_modes.items())),
        frac_songs_mode4=song_modes.get(4, 0) / n_songs,
        songs_multi=sum(1 for s in per_song if s["n_distinct"] > 1),
        songs_multi_frac=sum(1 for s in per_song if s["n_distinct"] > 1) / n_songs,
        songs_multi_strict=sum(1 for s in per_song if s["frac_modal"] < 0.98),
        songs_multi_strict_frac=sum(1 for s in per_song if s["frac_modal"] < 0.98) / n_songs,
        bar_switch_rate=(total_changes / total_trans) if total_trans else float("nan"),
        total_changes=total_changes, total_trans=total_trans)


def main():
    print("=" * 78)
    print("METER reality check (ELBO_for_DBN.md 3, 5.1: per-frame Categorical")
    print("with full KxK transition matrix; switches allowed mid-song)")
    print("=" * 78)
    pooled_bl, pooled_ps = [], []
    for d in DATASETS_WITH_METER:
        per_song, bl, n_no_col1, n_files = analyze_dataset(d)
        s = summarize(d, per_song, bl)
        if s is None:
            print(f"\n[{d}] no col1 meter annotation in {n_files} files")
            continue
        pooled_bl.extend(bl); pooled_ps.extend(per_song)
        print(f"\n[{d}]  files={n_files}  songs_with_meter={s['n_songs']}  no_col1={n_no_col1}")
        print(f"   bpb histogram (over {s['total_bars']} completed bars): {s['hist']}")
        print(f"   frac of bars that are bpb==4 : {s['frac4_bars']:.3f}")
        print(f"   song-level modal meter hist  : {s['song_modes']}")
        print(f"   frac of songs whose meter==4 : {s['frac_songs_mode4']:.3f}")
        print(f"   songs with >1 distinct bpb   : {s['songs_multi']}/{s['n_songs']} ({s['songs_multi_frac']*100:.1f}%)")
        print(f"   songs 'really' multi-meter   : {s['songs_multi_strict']}/{s['n_songs']} ({s['songs_multi_strict_frac']*100:.1f}%)  [modal<98% of bars]")
        print(f"   bar-to-bar SWITCH RATE       : {s['bar_switch_rate']:.5f}  ({s['total_changes']} switches / {s['total_trans']} bar-transitions)")

    print("\n" + "=" * 78)
    print("POOLED across meter-annotated datasets")
    print("=" * 78)
    ps = summarize("POOLED", pooled_ps, pooled_bl)
    print(f"   songs={ps['n_songs']}  completed_bars={ps['total_bars']}")
    print(f"   bpb histogram: {ps['hist']}")
    print(f"   frac bpb==4  : {ps['frac4_bars']:.4f}")
    print(f"   distinct meters observed (bar lengths): {sorted(ps['hist'].keys())}")
    print(f"   song modal-meter hist: {ps['song_modes']}")
    print(f"   frac songs meter==4  : {ps['frac_songs_mode4']:.4f}")
    print(f"   songs with >1 distinct bpb : {ps['songs_multi']}/{ps['n_songs']} ({ps['songs_multi_frac']*100:.2f}%)")
    print(f"   songs 'really' multi-meter : {ps['songs_multi_strict']}/{ps['n_songs']} ({ps['songs_multi_strict_frac']*100:.2f}%)  [modal<98% of bars]")
    print(f"   POOLED bar-to-bar SWITCH RATE: {ps['bar_switch_rate']:.6f}  ({ps['total_changes']}/{ps['total_trans']})")
    print("\n   --- per-FRAME implication ---")
    print("   Meter can only change at a bar boundary; the bar-to-bar switch rate")
    print("   above is an UPPER BOUND on any per-frame switch probability.")
    print(f"   Upper-bound per-bar-boundary switch prob ~ {ps['bar_switch_rate']:.2e}")

    # ---- decisive statistic: sticky (per-song constant) vs iid per-bar draw ----
    print("\n" + "=" * 78)
    print("DECISIVE STAT: empirical switching vs a named alternative (iid per-bar)")
    print("=" * 78)
    hist = collections.Counter(pooled_bl)
    total_bars = sum(hist.values())
    p = {k: v / total_bars for k, v in hist.items()}
    null_switch = 1.0 - sum(pv * pv for pv in p.values())  # E[switch] if iid per-bar draw
    emp_switch = ps["bar_switch_rate"]
    print(f"   pooled bar marginal p(bpb): { {k: round(v,4) for k,v in sorted(p.items())} }")
    print(f"   iid-per-bar-draw null: expected switch rate = 1 - sum p_k^2 = {null_switch:.4f}")
    print(f"   EMPIRICAL bar-to-bar switch rate                          = {emp_switch:.6f}")
    print(f"   ratio null/empirical = {null_switch/emp_switch:.1f}x  "
          f"(real meter is ~{null_switch/emp_switch:.0f}x STICKIER than iid per-bar)")
    # LLR: per-song-constant model vs iid-per-bar model, on the bar sequences
    #   constant model: one draw per song from marginal, then held (0 nats/transition)
    #   iid model: every bar drawn from marginal (sum log p_k per transition)
    #   difference dominated by switches; report avg nats/bar-transition saved by 'constant'.
    ll_iid_trans = 0.0
    ll_const_trans = 0.0
    n_trans = 0
    for s_ps in pooled_ps:
        pass  # per-song bar sequences not retained; recompute below
    # recompute over raw sequences for LLR
    for d in DATASETS_WITH_METER:
        for f in sorted(glob.glob(os.path.join(ROOT, d, "annotations", "beats", "*.beats"))):
            arr = np.loadtxt(f, ndmin=2)
            if arr.shape[1] < 2:
                continue
            idx = arr[:, 1].astype(int)
            db = np.where(idx == 1)[0]
            if len(db) < 3:
                continue
            bl = list(np.diff(db))
            for i in range(1, len(bl)):
                n_trans += 1
                ll_iid_trans += np.log(p.get(bl[i], 1e-12))
                # constant model: P(same)=1-eps, P(switch)=eps*marginal; use empirical eps
                if bl[i] == bl[i - 1]:
                    ll_const_trans += np.log(1.0 - emp_switch + 1e-12)
                else:
                    ll_const_trans += np.log(emp_switch * p.get(bl[i], 1e-12) + 1e-12)
    print(f"   LLR per bar-transition: constant/sticky model beats iid by "
          f"{(ll_const_trans - ll_iid_trans)/max(n_trans,1):.4f} nats/transition "
          f"(n={n_trans}); positive => switching is over-modeled.")

    print("\n" + "=" * 78)
    print("SMC (target-style corpus): NO meter annotation")
    print("=" * 78)
    for d in DATASETS_NO_METER:
        files = sorted(glob.glob(os.path.join(ROOT, d, "annotations", "beats", "*.beats")))
        has_any = sum(1 for f in files if np.loadtxt(f, ndmin=2).shape[1] >= 2)
        print(f"   [{d}] {len(files)} files, {has_any} with a col1 -> meter UNOBSERVABLE from annotations.")


if __name__ == "__main__":
    main()
