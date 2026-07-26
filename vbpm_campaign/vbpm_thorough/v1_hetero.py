"""V1(b-iii): heteroscedastic Student-t(nu=2) transition -- audio predicts WHEN tempo is
volatile (scale), even if not the increment mean. Exact discretized 1-ms-bin log-mass,
same protocol as step1 (train songs -> fit; eval-song second halves -> score).
Models: mu = u_prev + c;  s = exp(a + b*ghat(features)), ghat = GBM log|e| predictor.
Also mean+scale: mu = u_prev + mhat(features) with mhat = GBM mean predictor (OOF-calibrated shrink).
"""
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
NU=2.0
def gbm(): return GBR(max_iter=300,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=0)

Xf_t,Gt=L('full','train'); Xf_e,Ge=L('full','eval')
Xc_t,_=L('causal','train'); Xc_e,_=L('causal','eval')
Hh_t,Hh_e=H('train'),H('eval'); M2=half(Ge)
et=Gt['u']-Gt['u_prev']; lt=np.log(np.abs(et)+1e-4)
up_t,lo_t,hi_t=Gt['u_prev'],Gt['ulo'],Gt['uhi']
up_e,lo_e,hi_e=Ge['u_prev'][M2],Ge['ulo'][M2],Ge['uhi'][M2]

# ---------- baseline fixed-scale t2 RW
def fit_base():
    def nll(th):
        c,ls=th; return -logmass(lo_t,hi_t,up_t+c,math.exp(ls),'t',NU).mean()
    r=minimize(nll,[0.0,math.log(0.02)],method='Nelder-Mead',options=dict(maxiter=4000,xatol=1e-8,fatol=1e-10))
    return r.x
thb=fit_base()
LLb_tr=logmass(lo_t,hi_t,up_t+thb[0],math.exp(thb[1]),'t',NU).mean()
LLb=logmass(lo_e,hi_e,up_e+thb[0],math.exp(thb[1]),'t',NU).mean()
print(f'baseline t2 fixed-scale: train {LLb_tr:.4f}  heldout {LLb:.4f} (step1 said -4.176)')
OUT={'baseline':dict(train=float(LLb_tr),heldout=float(LLb))}

sets={'hist':(Hh_t,Hh_e),'audio-causal':(Xc_t,Xc_e),'audio-full':(Xf_t,Xf_e),
      'hist+audio-full':(np.hstack([Hh_t,Xf_t]),np.hstack([Hh_e,Xf_e]))}
gkf=GroupKFold(n_splits=5); gsong=Gt['stem']
for nm,(A_,B_) in sets.items():
    # OOF scale-feature on train (honest calibration), full-train model for eval
    g_oof=np.zeros(len(et)); m_oof=np.zeros(len(et))
    for tr,va in gkf.split(A_,lt,groups=gsong):
        g_oof[va]=gbm().fit(A_[tr],lt[tr]).predict(A_[va])
        m_oof[va]=gbm().fit(A_[tr],et[tr]).predict(A_[va])
    g_ev=gbm().fit(A_,lt).predict(B_[M2])
    m_ev=gbm().fit(A_,et).predict(B_[M2])
    # SCALE-ONLY: mu=u_prev+c, s=exp(a+b*g)
    def nll_s(th):
        c,a,b=th; return -logmass(lo_t,hi_t,up_t+c,np.exp(a+b*g_oof),'t',NU).mean()
    r=minimize(nll_s,[thb[0],thb[1],0.0],method='Nelder-Mead',options=dict(maxiter=6000,xatol=1e-8,fatol=1e-10))
    c,a,b=r.x
    LLs=logmass(lo_e,hi_e,up_e+c,np.exp(a+b*g_ev),'t',NU).mean()
    # MEAN+SCALE: mu=u_prev+c+w*mhat
    def nll_ms(th):
        c,a,b,w=th; return -logmass(lo_t,hi_t,up_t+c+w*m_oof,np.exp(a+b*g_oof),'t',NU).mean()
    r2_=minimize(nll_ms,[c,a,b,0.5],method='Nelder-Mead',options=dict(maxiter=8000,xatol=1e-8,fatol=1e-10))
    c2,a2,b2,w2=r2_.x
    LLms=logmass(lo_e,hi_e,up_e+c2+w2*m_ev,np.exp(a2+b2*g_ev),'t',NU).mean()
    # MEAN-ONLY (fixed scale refit)
    def nll_m(th):
        c,ls,w=th; return -logmass(lo_t,hi_t,up_t+c+w*m_oof,math.exp(ls),'t',NU).mean()
    r3_=minimize(nll_m,[thb[0],thb[1],0.5],method='Nelder-Mead',options=dict(maxiter=6000,xatol=1e-8,fatol=1e-10))
    LLm=logmass(lo_e,hi_e,up_e+r3_.x[0]+r3_.x[2]*m_ev,math.exp(r3_.x[1]),'t',NU).mean()
    print(f'{nm:16s} scale-only {LLs:.4f} (d={LLs-LLb:+.4f})  mean-only {LLm:.4f} (d={LLm-LLb:+.4f})  '
          f'mean+scale {LLms:.4f} (d={LLms-LLb:+.4f})  [b={b:.3f} w={w2:.3f}]')
    OUT[nm]=dict(scale=float(LLs),mean=float(LLm),mean_scale=float(LLms),
                 d_scale=float(LLs-LLb),d_mean=float(LLm-LLb),d_mean_scale=float(LLms-LLb),
                 b=float(b),w=float(w2))
json.dump(OUT,open('/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough/v1_hetero.json','w'),indent=1)
print('saved')
