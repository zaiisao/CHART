"""Decompose the audio gain: is it AUDIO, or just LONGER LATENT MEMORY (audio-blind)?"""
import sys, numpy as np, math, json
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
def L(mode,sp):
    d=np.load(f'feat_{mode}_{sp}.npz',allow_pickle=True)
    G={k:d[k] for k in d.files if k!='X'}; G['n']=len(G['u']); return d['X'].astype(np.float64),G
def H(sp):
    d=np.load(f'hist_{sp}.npz',allow_pickle=True); return d['X'].astype(np.float64)
def half(G):
    from collections import defaultdict
    idx=defaultdict(list); m=np.zeros(G['n'],bool)
    for i,s in enumerate(G['stem']): idx[s].append(i)
    for s,ii in idx.items(): m[np.array(ii[len(ii)//2:])]=True
    return m
Xf_t,Gt=L('full','train'); Xf_e,Ge=L('full','eval')
Xc_t,_=L('causal','train'); Xc_e,_=L('causal','eval')
Hh_t,Hh_e=H('train'),H('eval'); M2=half(Ge)
et=Gt['u']-Gt['u_prev']; ee=(Ge['u']-Ge['u_prev'])[M2]
sets={'hist(audio-blind)':(Hh_t,Hh_e),
      'audio-causal':(Xc_t,Xc_e),
      'audio-full':(Xf_t,Xf_e),
      'hist+audio-causal':(np.hstack([Hh_t,Xc_t]),np.hstack([Hh_e,Xc_e])),
      'hist+audio-full':(np.hstack([Hh_t,Xf_t]),np.hstack([Hh_e,Xf_e]))}
print(f'held-out n={len(ee)}, train n={len(et)};  MEAN prediction of e (held-out R2)')
OUT={}
for nm,(A_,B_) in sets.items():
    m=GBR(max_iter=300,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=0).fit(A_,et)
    p=m.predict(B_[M2]); r2=1-((ee-p)**2).sum()/((ee-ee.mean())**2).sum()
    print(f'  {nm:22s} R2={r2:+.4f}  MAE={np.abs(ee-p).mean():.5f}  (persistence MAE {np.abs(ee).mean():.5f})')
    OUT[nm]=dict(R2=float(r2),mae=float(np.abs(ee-p).mean()))
print(f'\nSCALE prediction: log|e| (held-out R2)')
lt=np.log(np.abs(et)+1e-4); le=np.log(np.abs(ee)+1e-4)
for nm,(A_,B_) in sets.items():
    m=GBR(max_iter=300,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=0).fit(A_,lt)
    p=m.predict(B_[M2]); r2=1-((le-p)**2).sum()/((le-le.mean())**2).sum()
    print(f'  {nm:22s} R2={r2:+.4f}')
    OUT[nm+'_scale']=dict(R2=float(r2))
json.dump(OUT,open('step6_gbm.json','w'),indent=1)
