"""Flaw probe: is the frozen head's activation on SMC informative about tempo at all?
Per-song: global autocorr best lag of beat channel vs true median IBI (octave-tolerant)."""
import sys, numpy as np
sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough')
from feats import _ac
from smc_data import build_smc, FPS
from data import build

def per_song(D, name):
    hits = 0; hits_oct = 0; n = 0; errs = []
    for d in D:
        a = d['act'][:, 0].astype(np.float64)
        L, V, _ = _ac(a, 10, 130)
        if L != L: continue
        ibi = float(np.median(d['I']))*FPS
        r = L/ibi
        errs.append(np.log2(max(r, 1e-6)))
        if abs(np.log2(r)) < np.log2(1.1): hits += 1
        if min(abs(np.log2(r)), abs(np.log2(r/2)), abs(np.log2(r*2)), abs(np.log2(r/3)), abs(np.log2(r*3))) < np.log2(1.1): hits_oct += 1
        n += 1
    print(f'{name}: n={n} AC-lag within 10% of IBI: {hits/n:.3f}  octave-tolerant: {hits_oct/n:.3f}  '
          f'median|log2 ratio| {np.median(np.abs(errs)):.3f}')

per_song(build_smc(), 'SMC')
import random
Dm = build('eval')
per_song(Dm, 'MAIN-eval')
