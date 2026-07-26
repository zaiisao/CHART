"""LL-level STACK: mu = u_prev + c + w1*mhist + w2*maud ; s = exp(a + b1*ghist + b2*gaud).
OOF-calibrated on train, scored held-out with exact discretized t2 mass. Compared to
hist-only calibrated model (same machinery, w2=b2=0) so the audio increment is isolated."""
import sys, json, math, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
from collections import defaultdict
from core import logmass
P='/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise'
def L(mode,sp):
    d=np.load(f'{P}/feat_{mode}_{sp}.npz',allow_pickle=True)
    G={k:d[k] for k in d.files if k!='X'}; G['n']=len(G['u']); return d['X'].astype(np.float64),G
def H(sp): return np.load(f'{P}/hist_{sp}.npz',allow_pickle=True)['X'].astype(np.float64)
def half(G):
    idx=defaultdict(list); m=np.zeros(G['n'],bool)
    for i,s in enumerate(G['stem']): idx[s].append(i)
    for s,ii in idx.items(): m[np.array(ii[len(ii)//2:])]=True
    return m
def gbm(): return GBR(max_iter=300,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=0)
NU=2.0
Xf_t,Gt=L('full','train'); Xf_e,Ge=L('full','eval')
Xc_t,_=L('causal','train'); Xc_e,_=L('causal','eval')
Hh_t,Hh_e=H('train'),H('eval'); M2=half(Ge)
et=Gt['u']-Gt['u_prev']; lt=np.log(np.abs(et)+1e-4)
up_t,lo_t,hi_t=Gt['u_prev'],Gt['ulo'],Gt['uhi']
up_e,lo_e,hi_e=Ge['u_prev'][M2],Ge['ulo'][M2],Ge['uhi'][M2]
gkf=GroupKFold(n_splits=5); gs=Gt['stem']
def oof_and_eval(A_,B_):
    mo=np.zeros(len(et)); go=np.zeros(len(et))
    for tr,va in gkf.split(A_,et,groups=gs):
        mo[va]=gbm().fit(A_[tr],et[tr]).predict(A_[va])
        go[va]=gbm().fit(A_[tr],lt[tr]).predict(A_[va])
    me=gbm().fit(A_,et).predict(B_[M2]); ge=gbm().fit(A_,lt).predict(B_[M2])
    return mo,go,me,ge
mh,gh,mhE,ghE=oof_and_eval(Hh_t,Hh_e)
OUT={}
for anm,(At,Ae) in (('audio-full',(Xf_t,Xf_e)),('audio-causal',(Xc_t,Xc_e))):
    ma,ga,maE,gaE=oof_and_eval(At,Ae)
    def nll_h(th):
        c,a,b1,w1=th
        return -logmass(lo_t,hi_t,up_t+c+w1*mh,np.exp(a+b1*gh),'t',NU).mean()
    r=minimize(nll_h,[0.0,math.log(0.02),0.5,1.0],method='Nelder-Mead',options=dict(maxiter=8000,xatol=1e-8,fatol=1e-10))
    c,a,b1,w1=r.x
    LLh=float(logmass(lo_e,hi_e,up_e+c+w1*mhE,np.exp(a+b1*ghE),'t',NU).mean())
    def nll_s(th):
        c,a,b1,b2,w1,w2=th
        return -logmass(lo_t,hi_t,up_t+c+w1*mh+w2*ma,np.exp(a+b1*gh+b2*ga),'t',NU).mean()
    r2_=minimize(nll_s,[c,a,b1,0.0,w1,0.3],method='Nelder-Mead',options=dict(maxiter=12000,xatol=1e-8,fatol=1e-10))
    c2,a2,b12,b22,w12,w22=r2_.x
    LLs=float(logmass(lo_e,hi_e,up_e+c2+w12*mhE+w22*maE,np.exp(a2+b12*ghE+b22*gaE),'t',NU).mean())
    print(f'{anm}: hist-only calibrated LL={LLh:.4f}; +audio stack LL={LLs:.4f}  d_audio={LLs-LLh:+.4f}  (w2={w22:.3f} b2={b22:.3f})')
    OUT[anm]=dict(LL_hist=LLh,LL_stack=LLs,d_audio=LLs-LLh,w2=float(w22),b2=float(b22),w1=float(w12),b1=float(b12))
json.dump(OUT,open('/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough/v1_hetero2.json','w'),indent=1)
print('saved')
