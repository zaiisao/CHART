import sys, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from data import build
D = build('train')
for ds in ('ballroom','beatles','hainsworth'):
    sub=[d for d in D if d['dataset']==ds]
    b=sub[0]['beats']
    print(ds, sub[0]['stem'])
    print('  first beats', np.round(b[:8],6))
    print('  first IBIs ', np.round(np.diff(b[:9]),6))
    allb=np.concatenate([d['beats'] for d in sub])
    for g in (1000, 100, 44100/1024, 50):
        r = allb*g - np.round(allb*g)
        print(f'   grid {g:.2f}/s: frac|res|<1e-6 = {np.mean(np.abs(r)<1e-6):.4f}')
    # how many songs have perfectly constant IBI
    ncon = sum(1 for d in sub if np.std(d['I'])<1e-6)
    print('   songs w/ constant IBI:', ncon, '/', len(sub))
    e=np.concatenate([d['e'] for d in sub])
    ez=np.abs(e)<1e-9
    print('   zero-increment songs share: ', np.round(np.mean([np.mean(np.abs(d['e'])<1e-9) for d in sub]),3))
