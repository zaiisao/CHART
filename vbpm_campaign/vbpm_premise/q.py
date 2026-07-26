import sys, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from data import build, FPS
for sp in ('train','eval'):
    D = build(sp)
    for ds in sorted(set(d['dataset'] for d in D)):
        e = np.concatenate([d['e'] for d in D if d['dataset']==ds])
        I = np.concatenate([d['I'] for d in D if d['dataset']==ds])
        print(sp, ds, 'n_e', len(e), 'frac|e|<1e-9', round(float(np.mean(np.abs(e)<1e-9)),4),
              'frac|e|<1e-4', round(float(np.mean(np.abs(e)<1e-4)),4),
              'sd', round(float(e.std()),4), 'mad', round(float(np.abs(e).mean()),5))
    # annotation time resolution
    b = np.concatenate([d['beats'] for d in D])
    r = b*1000 - np.round(b*1000)
    print('   beat-time ms residual absmax', float(np.abs(r).max()))
