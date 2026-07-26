"""V1(a) deploy-fairness: history features rebuilt from JITTERED beat times
(iid Gaussian on each beat, sigma in {10,20,30} ms ~ realistic filter/peak timing error),
targets stay GROUND TRUTH. Audio features rebuilt with the SAME jittered anchors/u_prev.
Question: once history is only as good as a deployed estimate, does audio add?"""
import sys, json, copy, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from collections import defaultdict
from core import prep
from data import build, FPS, TWO_PI
from feats import song_feats
from histfeat import hist_feats
import math
rng=np.random.default_rng(0)

def jitter_song(d, sigma):
    dd=dict(d)  # shallow copy; replace tempo-history fields with jittered versions
    b=d['beats'].copy()
    if sigma>0:
        b=b+rng.normal(0,sigma,size=len(b))
        b=np.sort(b)
    I=np.diff(b); I=np.maximum(I,6e-3)
    A=math.log(TWO_PI/(d['meter']*FPS))
    dd['beats']=b; dd['I']=I; dd['u']=A-np.log(I)
    # keep CLEAN 'valid' so row indexing matches the clean targets
    return dd

def feats_for(D, sigma):
    Xa=[]; Xh=[]; y=[]; stems=[]
    for d in D:
        dj=jitter_song(d,sigma)
        X,ks=song_feats(dj,300,'full')
        if not len(ks): continue
        Xa.append(X)
        # history feats on jittered u, same ks
        n=len(dj['u']); u=dj['u']
        for k in ks:
            up=u[k-1]
            lag=lambda j: u[k-j] if k-j>=0 else up
            past=u[max(0,k-8):k]; past2=u[max(0,k-4):k]; allp=u[:k]
            de=np.diff(allp) if len(allp)>1 else np.array([0.0])
            f=[up, up-lag(2), lag(2)-lag(3), lag(3)-lag(4),
               float(np.mean(past))-up, float(np.mean(past2))-up,
               float(np.mean(np.abs(np.diff(past)))) if len(past)>2 else 0.0,
               float(np.mean(np.abs(de))), float(np.std(de)), float(np.median(np.abs(de))),
               float(np.mean(allp))-up, float(np.std(allp)), float(k),
               float(d['meter']==2),float(d['meter']==3),float(d['meter']==4)]
            Xh.append(f)
            y.append(d['u'][k]-d['u'][k-1])       # CLEAN target
            stems.append(d['stem'])
    return np.concatenate(Xa,0).astype(np.float64), np.asarray(Xh), np.asarray(y), np.asarray(stems)

def half_mask(stems):
    idx=defaultdict(list); m=np.zeros(len(stems),bool)
    for i,s in enumerate(stems): idx[s].append(i)
    for s,ii in idx.items(): m[np.array(ii[len(ii)//2:])]=True
    return m
def r2(y,p): return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
def gbm(): return GBR(max_iter=300,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=0)

Dt=prep(build('train')); De=prep(build('eval'))
OUT={}
for sigma in (0.0,0.01,0.02,0.03):
    Xa_t,Xh_t,yt,st_t=feats_for(Dt,sigma)
    Xa_e,Xh_e,ye,st_e=feats_for(De,sigma)
    M2=half_mask(st_e); yh=ye[M2]
    row={}
    preds={}
    for nm,(A_,B_) in {'hist':(Xh_t,Xh_e),'audio':(Xa_t,Xa_e),
                       'hist+audio':(np.hstack([Xh_t,Xa_t]),np.hstack([Xh_e,Xa_e]))}.items():
        m=gbm().fit(A_,yt); p=m.predict(B_[M2]); preds[nm]=p
        row[nm]=float(r2(yh,p))
    # OOF stack hist+audio
    gkf=GroupKFold(n_splits=5); oof={k:np.zeros(len(yt)) for k in ('hist','audio')}
    for nm,A_ in (('hist',Xh_t),('audio',Xa_t)):
        for tr,va in gkf.split(A_,yt,groups=st_t):
            oof[nm][va]=gbm().fit(A_[tr],yt[tr]).predict(A_[va])
    stk=LinearRegression().fit(np.column_stack([oof['hist'],oof['audio']]),yt)
    row['stack']=float(r2(yh,stk.predict(np.column_stack([preds['hist'],preds['audio']]))))
    row['stack_coefs']=[float(c) for c in stk.coef_]
    # recalibrated hist baseline (fair vs stack)
    cal=LinearRegression().fit(oof['hist'].reshape(-1,1),yt)
    row['hist_recal']=float(r2(yh,cal.predict(preds['hist'].reshape(-1,1))))
    OUT[f'sigma_{int(sigma*1000)}ms']=row
    print(f"sigma={sigma*1000:.0f}ms  hist={row['hist']:+.4f} hist_recal={row['hist_recal']:+.4f} audio={row['audio']:+.4f} "
          f"joint={row['hist+audio']:+.4f} stack={row['stack']:+.4f} coefs={np.round(row['stack_coefs'],3)}")
json.dump(OUT,open('/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough/v1_deployfair.json','w'),indent=1)
print('saved')
