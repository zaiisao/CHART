"""Loader shim for the full-corpus MERT cache (Stage 3).

Layout: /disk4/jaehoon/VBPM_cache/mert/corpus/<stem>.npz
  feats [T, k*768] fp16 at 50 fps (winning layers, concatenated in `layers` order)
plus manifest.json {stem: {dataset, layers, fps, n_frames}}.

  from mert_features import get, manifest
  get("hjdb_R_Yeah")  -> np.ndarray [T, D] float32
"""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

CORPUS = Path("/disk4/jaehoon/VBPM_cache/mert/corpus")


@lru_cache(maxsize=1)
def manifest():
    return json.load(open(CORPUS / "manifest.json"))


def get(stem):
    """[T, D] float32 MERT features for a song stem (50 fps unless noted in the manifest)."""
    return np.load(CORPUS / f"{stem}.npz")["feats"].astype(np.float32)
