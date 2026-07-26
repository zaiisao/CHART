"""X2(b) extra arm: penult-512 (PCA-64) causal audio features vs history, GBM, same protocol."""
import sys, numpy as np, json
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.decomposition import PCA
from collections import defaultdict
T='/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough'
def L(path):
    d=np.load(path,allow_pickle=True)
    G={k:d[k] for k in d.files if k!='X'}; G['n']=len(G['u'])
    return d['X'].astype(np.float64),G
Xh_t,Gt=L(f'{T}/hist_asap_train.npz'); Xh_e,Ge=L(f'{T}/hist_asap_eval.npz')
Pt=np.load(f'{T}/pen_asap_train.npz')['X'].astype(np.float32)
Pe=np.load(f'{T}/pen_asap_eval.npz')['X'].astype(np.float32)
assert len(Pt)==Gt['n'] and len(Pe)==Ge['n'], (len(Pt),Gt['n'],len(Pe),Ge['n'])
rng=np.random.default_rng(0)
sub=rng.choice(len(Pt), 40000, replace=False)
pca=PCA(n_components=64, random_state=0).fit(Pt[sub])
print('PCA evr', float(pca.explained_variance_ratio_.sum()), flush=True)
Zt=pca.transform(Pt).astype(np.float64); Ze=pca.transform(Pe).astype(np.float64)
# mirror feats.py: audio arm also carries u_prev + meter one-hots
ex_t=np.stack([Gt['u_prev'],(Gt['meter']==2)*1.0,(Gt['meter']==3)*1.0,(Gt['meter']==4)*1.0],1)
ex_e=np.stack([Ge['u_prev'],(Ge['meter']==2)*1.0,(Ge['meter']==3)*1.0,(Ge['meter']==4)*1.0],1)
def half(G):
    idx=defaultdict(list); m=np.zeros(G['n'],bool)
    for i,s in enumerate(G['stem']): idx[s].append(i)
    for s,ii in idx.items(): m[np.array(ii[len(ii)//2:])]=True
    return m
M2=half(Ge)
et=Gt['u']-Gt['u_prev']; ee=(Ge['u']-Ge['u_prev'])[M2]
sets={'penult-causal(PCA64)':(np.hstack([Zt,ex_t]),np.hstack([Ze,ex_e])),
      'hist+penult-causal':(np.hstack([Xh_t,Zt]),np.hstack([Xh_e,Ze]))}
OUT={}
print(f'held-out n={len(ee)}; MEAN prediction R2')
for nm,(A_,B_) in sets.items():
    m=GBR(max_iter=300,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=0).fit(A_,et)
    p=m.predict(B_[M2]); r2=1-((ee-p)**2).sum()/((ee-ee.mean())**2).sum()
    print(f'  {nm:22s} R2={r2:+.4f}  MAE={np.abs(ee-p).mean():.5f}', flush=True)
    OUT[nm]=dict(R2=float(r2),mae=float(np.abs(ee-p).mean()))
lt=np.log(np.abs(et)+1e-4); le=np.log(np.abs(ee)+1e-4)
print('SCALE prediction R2')
for nm,(A_,B_) in sets.items():
    m=GBR(max_iter=300,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=0).fit(A_,lt)
    p=m.predict(B_[M2]); r2=1-((le-p)**2).sum()/((le-le.mean())**2).sum()
    print(f'  {nm:22s} R2={r2:+.4f}', flush=True)
    OUT[nm+'_scale']=dict(R2=float(r2))
json.dump(OUT, open(f'{T}/asap_step6b.json','w'), indent=1)
print('DONE')
