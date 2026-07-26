"""V1(a)+(b-ii): audit numerics on the CACHED features of step5/step6.
- alignment check: corr of the near-oracle look-ahead feature with the target
- zero-atom + per-dataset structure of the target
- R2 restricted to: nonzero increments, top-decile |e|, per-dataset
- stacking check: does audio add orthogonal signal on top of hist predictions?
"""
import sys, json, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from collections import defaultdict
P='/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise'
def L(mode,sp):
    d=np.load(f'{P}/feat_{mode}_{sp}.npz',allow_pickle=True)
    G={k:d[k] for k in d.files if k!='X'}; G['n']=len(G['u']); return d['X'].astype(np.float64),G
def H(sp):
    return np.load(f'{P}/hist_{sp}.npz',allow_pickle=True)['X'].astype(np.float64)
def half(G):
    idx=defaultdict(list); m=np.zeros(G['n'],bool)
    for i,s in enumerate(G['stem']): idx[s].append(i)
    for s,ii in idx.items(): m[np.array(ii[len(ii)//2:])]=True
    return m
def r2(y,p,m=None):
    if m is not None: y,p=y[m],p[m]
    if len(y)<10: return float('nan')
    return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()

Xf_t,Gt=L('full','train'); Xf_e,Ge=L('full','eval')
Xc_t,_=L('causal','train'); Xc_e,_=L('causal','eval')
Hh_t,Hh_e=H('train'),H('eval'); M2=half(Ge)
et=Gt['u']-Gt['u_prev']; ee_all=Ge['u']-Ge['u_prev']; ee=ee_all[M2]
ds_e=Ge['dataset'][M2]

print('=== target structure (held-out, n=%d) ==='%len(ee))
z=np.abs(ee)<1e-9
print('frac exactly-zero increment: %.4f'%z.mean())
for d in np.unique(ds_e):
    m=ds_e==d
    print(f'  {d:12s} n={m.sum():5d} frac_zero={np.abs(ee[m]).mean()<1e-9 if False else (np.abs(ee[m])<1e-9).mean():.4f} sd={ee[m].std():.4f} mad={np.abs(ee[m]-np.median(ee[m])).mean():.4f}')

print('\n=== alignment check: individual FULL features vs target (held-out corr) ===')
names={0:'c:globalAC-u_prev',2:'c:localAC-u_prev',8:'c:peakIntervalRatio',10:'n:globalAC-u_prev',12:'n:localAC-u_prev',18:'n:peakIntervalRatio',20:'LOOKAHEAD log((j-f)/Lp)',21:'lookahead peak height',25:'u_prev'}
Xe2=Xf_e[M2]
for c,nm in names.items():
    v=Xe2[:,c]
    print(f'  col{c:3d} {nm:26s} corr={np.corrcoef(v,ee)[0,1]:+.4f}   corr(-target expected for col20)')
# theoretical: if peak j were the true next beat, log((j-f)/Lp)= log(I_k/I_{k-1}) = -e_k
la=Xe2[:,20]; found=la!=0.0
print(f'  lookahead found frac={found.mean():.3f}; corr on found-only={np.corrcoef(la[found],ee[found])[0,1]:+.4f}; corr(-la,e)={np.corrcoef(-la[found],ee[found])[0,1]:+.4f}')

print('\n=== R2 by subset (models fit on FULL train targets, GBM same hyperparams) ===')
sets={'hist':(Hh_t,Hh_e),'audio-causal':(Xc_t,Xc_e),'audio-full':(Xf_t,Xf_e),
      'hist+aud-causal':(np.hstack([Hh_t,Xc_t]),np.hstack([Hh_e,Xc_e])),
      'hist+aud-full':(np.hstack([Hh_t,Xf_t]),np.hstack([Hh_e,Xf_e]))}
preds={}
q90=np.quantile(np.abs(ee),0.9)
mnz=~z; mtop=np.abs(ee)>=q90
res={}
for nm,(A_,B_) in sets.items():
    mdl=GBR(max_iter=300,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=0).fit(A_,et)
    p=mdl.predict(B_[M2]); preds[nm]=p
    row=dict(all=r2(ee,p),nonzero=r2(ee,p,mnz),top10=r2(ee,p,mtop))
    for d in np.unique(ds_e): row[d]=r2(ee,p,ds_e==d)
    res[nm]=row
    print(f'  {nm:16s} all={row["all"]:+.4f} nonzero={row["nonzero"]:+.4f} top10%|e|={row["top10"]:+.4f} | '+' '.join(f'{d}={row[d]:+.3f}' for d in np.unique(ds_e)))

print('\n=== stacking: ridge on [hist_pred, audio_pred] (fit on train-side preds via 5-fold-ish) ===')
# out-of-fold train preds by song groups
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
gt=Gt['stem']
oof={nm:np.zeros(len(et)) for nm in ('hist','audio-full','audio-causal')}
gkf=GroupKFold(n_splits=5)
for nm in oof:
    A_,B_=sets[nm]
    for tr,va in gkf.split(A_,et,groups=gt):
        m=GBR(max_iter=300,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=0).fit(A_[tr],et[tr])
        oof[nm][va]=m.predict(A_[va])
for anm in ('audio-full','audio-causal'):
    st=LinearRegression().fit(np.column_stack([oof['hist'],oof[anm]]),et)
    ps=st.predict(np.column_stack([preds['hist'],preds[anm]]))
    print(f'  stack hist+{anm}: coefs={st.coef_.round(3)} heldout R2={r2(ee,ps):+.4f} (hist alone {res["hist"]["all"]:+.4f})')
    res[f'stack_hist_{anm}']=dict(R2=float(r2(ee,ps)),coefs=[float(c) for c in st.coef_])

json.dump({k:{kk:(float(vv) if isinstance(vv,(int,float,np.floating)) else vv) for kk,vv in v.items()} if isinstance(v,dict) else v for k,v in res.items()},
          open('/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough/v1_audit.json','w'),indent=1)
print('\nsaved v1_audit.json')
