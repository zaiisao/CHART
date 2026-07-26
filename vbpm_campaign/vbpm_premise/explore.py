import sys, numpy as np, collections
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_final')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')
from emission import load_act, load_split, song_phase, METERS
from vbpm.evaluate import _estimate_meter
for sp in ('train','eval'):
    S = load_split(sp); A = load_act(sp)
    print(sp, 'songs', len(S), 'acts', len(A))
    ds = collections.Counter(s['dataset'] for s in S); print('  datasets', dict(ds))
    mm = collections.Counter(_estimate_meter(s['beats'], s['downs']) for s in S); print('  meters', dict(mm))
    s = S[0]; a = A.get(s['stem'])
    print('  ex stem', s['stem'], 'T', s['T'], 'act', None if a is None else a.shape, a.dtype if a is not None else '')
    print('  nbeats', [len(x['beats']) for x in S[:5]], 'ndowns', [len(x['downs']) for x in S[:5]])
    miss = [x['stem'] for x in S if A.get(x['stem']) is None]
    print('  missing acts', len(miss), miss[:3])
    lens = [(len(A[x['stem']]), x['T']) for x in S if A.get(x['stem']) is not None][:5]
    print('  (len(act), T)', lens)
    nb = np.array([len(x['beats']) for x in S]); print('  beats total', nb.sum(), 'min', nb.min())
