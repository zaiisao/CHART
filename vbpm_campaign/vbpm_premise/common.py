"""Shared loader for the PREMISE audit (labels only -- no 30MB feats decompression)."""
from __future__ import annotations

import glob
import math
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final")

from vbpm.evaluate import _estimate_meter  # noqa: E402

CACHE = "/disk1/jaehoon/vbpm_mert_cache"
FPS = 50.0
TWO_PI = 2.0 * math.pi


def _T_of(path):
    z = zipfile.ZipFile(path)
    with z.open("feats.npy") as fh:
        ver = np.lib.format.read_magic(fh)
        shp, _, _ = np.lib.format._read_array_header(fh, ver)
    return int(shp[1])


def load_labels(split):
    out = []
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d = np.load(f, allow_pickle=True)
        rec = dict(stem=Path(f).stem, split=split, path=f, T=_T_of(f),
                   beats=np.sort(np.asarray(d["beats"], float)),
                   downs=np.sort(np.asarray(d["downs"], float)),
                   fps=float(d["fps"]), dataset=str(d["dataset"]), fold=int(d["fold"]))
        rec["meter"] = int(_estimate_meter(rec["beats"], rec["downs"]))
        out.append(rec)
    return out


def per_ds(rows, key, fn=np.mean):
    ds = {}
    for r in rows:
        ds.setdefault(r["dataset"], []).append(r[key])
    o = {k: (float(fn(v)), len(v)) for k, v in sorted(ds.items())}
    allv = [r[key] for r in rows]
    o["POOLED"] = (float(fn(allv)), len(allv))
    return o


def fmt_ds(d, prec=4):
    return "  ".join(f"{k}={v:.{prec}f}(n={n})" for k, (v, n) in d.items())
