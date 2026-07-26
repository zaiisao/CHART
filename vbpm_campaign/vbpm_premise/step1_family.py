import sys, numpy as np, math, json
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from core import *
Dtr = prep(build('train')); Dev = prep(build('eval'))
Gtr = gather(pairs(Dtr,'all'))
Gev2 = gather(pairs(Dev,'second'))
print('train pairs', Gtr['n'], 'eval-2ndhalf pairs', Gev2['n'], 'songs', len(set(Gev2['stem'])))
res={}
for fam,nu in [('gauss',0),('laplace',0),('t',2.0),('t',3.0),('t',5.0)]:
    for ou in (False,True):
        M = fit_rw(Gtr, fam, nu, ou)
        tr = score_rw(M,Gtr).mean(); ev = score_rw(M,Gev2).mean()
        nm=f"{fam}{'' if not nu else nu}{'_OU' if ou else '_RW'}"
        res[nm]=dict(train=float(tr), eval2=float(ev), th=[float(x) for x in M['th']])
        print(f"{nm:14s} train {tr:+.4f} evalHO {ev:+.4f} params {np.round(M['th'],4)}")
json.dump(res, open('/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise/step1.json','w'), indent=1)
