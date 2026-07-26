import sys, numpy as np, math, time
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from core import *
from feats import build_feats
t0=time.time()
Dev = prep(build('eval'))
G = gather(pairs(Dev,'all'))
X = build_feats(Dev, mode='full')
print('X', X.shape, 'G n', G['n'], 'time %.1fs'%(time.time()-t0))
e = G['u']-G['u_prev']
names=['c_gOff','c_gV','c_lOff','c_lV','c_mean','c_max','c_std','c_dmean','c_pkOff','c_pkFrac',
       'n_gOff','n_gV','n_lOff','n_lV','n_mean','n_max','n_std','n_dmean','n_pkOff','n_pkFrac',
       'la_logRatio','la_val','db_at_f','n_dbpk','act_at_f','u_prev','m2','m3','m4']
print('feature/target correlations with e = u_k - u_{k-1}  (n=%d)'%len(e))
for i,nm in enumerate(names):
    c=np.corrcoef(X[:,i], e)[0,1]
    if abs(c)>0.02: print(f'   {nm:12s} r={c:+.3f}')
# how good is the audio tempo estimate as an absolute predictor of u_k?
uhat_la = G['u_prev'] + X[:,20]
uhat_ln = G['u_prev'] + X[:,12]
for nm,uh in [('lookahead',uhat_la),('noncausal local-AC',uhat_ln),('causal local-AC',G['u_prev']+X[:,2])]:
    print(f'   {nm:20s} corr(u_hat,u)={np.corrcoef(uh,G["u"])[0,1]:+.4f}  MAE={np.abs(uh-G["u"]).mean():.4f}'
          f'  (persistence MAE={np.abs(G["u_prev"]-G["u"]).mean():.4f})')
