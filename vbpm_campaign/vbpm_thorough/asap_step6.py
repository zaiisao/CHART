"""X2(b) KEY: history-alone vs audio-causal vs combined prediction of per-beat
tempo increments on ASAP, held-out by piece (eval pieces) x 2nd-half-of-song.
Mirror of vbpm_premise/step6_gbm.py + change-point formulation.
Also runs the change-point formulation on the STEADY corpus files for reference."""
import sys, numpy as np, json
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.ensemble import HistGradientBoostingClassifier as GBC
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

def L(path):
    d=np.load(path,allow_pickle=True)
    G={k:d[k] for k in d.files if k!='X'}; G['n']=len(G['u'])
    return d['X'].astype(np.float64),G
def half(G):
    idx=defaultdict(list); m=np.zeros(G['n'],bool)
    for i,s in enumerate(G['stem']): idx[s].append(i)
    for s,ii in idx.items(): m[np.array(ii[len(ii)//2:])]=True
    return m

def run(tag, hist_tr, hist_ev, cz_tr, cz_ev, fu_tr, fu_ev, out):
    Xh_t,Gt=L(hist_tr); Xh_e,Ge=L(hist_ev)
    Xc_t,_=L(cz_tr);    Xc_e,_=L(cz_ev)
    Xf_t,_=L(fu_tr);    Xf_e,_=L(fu_ev)
    M2=half(Ge)
    et=Gt['u']-Gt['u_prev']; ee=(Ge['u']-Ge['u_prev'])[M2]
    sets={'hist(audio-blind)':(Xh_t,Xh_e),
          'audio-causal':(Xc_t,Xc_e),
          'audio-full':(Xf_t,Xf_e),
          'hist+audio-causal':(np.hstack([Xh_t,Xc_t]),np.hstack([Xh_e,Xc_e])),
          'hist+audio-full':(np.hstack([Xh_t,Xf_t]),np.hstack([Xh_e,Xf_e]))}
    print(f'=== {tag}: held-out n={len(ee)}, train n={len(et)}', flush=True)
    O={}
    print('MEAN prediction of e (held-out R2)')
    for nm,(A_,B_) in sets.items():
        m=GBR(max_iter=300,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=0).fit(A_,et)
        p=m.predict(B_[M2]); r2=1-((ee-p)**2).sum()/((ee-ee.mean())**2).sum()
        print(f'  {nm:22s} R2={r2:+.4f}  MAE={np.abs(ee-p).mean():.5f}  (persistence MAE {np.abs(ee).mean():.5f})', flush=True)
        O[nm]=dict(R2=float(r2),mae=float(np.abs(ee-p).mean()))
    print('SCALE prediction: log|e| (held-out R2)')
    lt=np.log(np.abs(et)+1e-4); le=np.log(np.abs(ee)+1e-4)
    for nm,(A_,B_) in sets.items():
        m=GBR(max_iter=300,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=0).fit(A_,lt)
        p=m.predict(B_[M2]); r2=1-((le-p)**2).sum()/((le-le.mean())**2).sum()
        print(f'  {nm:22s} R2={r2:+.4f}', flush=True)
        O[nm+'_scale']=dict(R2=float(r2))
    print('CHANGE-POINT: |e|>0.1 (held-out ROC-AUC / AP)')
    yt=(np.abs(et)>0.1).astype(int); ye=(np.abs(ee)>0.1).astype(int)
    print(f'  base rate train {yt.mean():.4f} eval {ye.mean():.4f}')
    for nm,(A_,B_) in sets.items():
        if yt.sum()<50 or ye.sum()<20 or ye.mean() in (0.0,1.0):
            print(f'  {nm:22s} SKIP (too few change-points)'); continue
        m=GBC(max_iter=300,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=0).fit(A_,yt)
        p=m.predict_proba(B_[M2])[:,1]
        auc=roc_auc_score(ye,p); ap=average_precision_score(ye,p)
        print(f'  {nm:22s} AUC={auc:.4f}  AP={ap:.4f}  (base {ye.mean():.4f})', flush=True)
        O[nm+'_cp']=dict(auc=float(auc),ap=float(ap),base=float(ye.mean()))
    out[tag]=O

OUT={}
T='/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough'
P='/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise'
run('ASAP', f'{T}/hist_asap_train.npz', f'{T}/hist_asap_eval.npz',
    f'{T}/feat_causal_asap_train.npz', f'{T}/feat_causal_asap_eval.npz',
    f'{T}/feat_full_asap_train.npz', f'{T}/feat_full_asap_eval.npz', OUT)
run('STEADY', f'{P}/hist_train.npz', f'{P}/hist_eval.npz',
    f'{P}/feat_causal_train.npz', f'{P}/feat_causal_eval.npz',
    f'{P}/feat_full_train.npz', f'{P}/feat_full_eval.npz', OUT)
json.dump(OUT, open(f'{T}/asap_step6.json','w'), indent=1)
print('DONE')
