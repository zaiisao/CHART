"""Interpretable corollary: how much of the one-step tempo increment is PREDICTABLE from audio,
and what is that worth in milliseconds of next-beat placement (beat-F tolerance is +-70 ms)."""
import sys, math, numpy as np, json
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
def load(mode,sp):
    d=np.load(f'feat_{mode}_{sp}.npz',allow_pickle=True)
    G={k:d[k] for k in d.files if k!='X'}; G['n']=len(G['u']); return d['X'].astype(np.float64),G
def half(G):
    from collections import defaultdict
    idx=defaultdict(list); m=np.zeros(G['n'],bool)
    for i,s in enumerate(G['stem']): idx[s].append(i)
    for s,ii in idx.items(): m[np.array(ii[len(ii)//2:])]=True
    return m
OUT={}
for mode in ('full','causal'):
    Xt,Gt=load(mode,'train'); Xe,Ge=load(mode,'eval'); M2=half(Ge)
    et=Gt['u']-Gt['u_prev']; ee=(Ge['u']-Ge['u_prev'])[M2]
    Xe2=Xe[M2]
    print(f'--- {mode}: train n={len(et)}  held-out n={len(ee)}')
    for nm,mdl in [('ridge',RidgeCV(alphas=np.logspace(-2,4,20))),
                   ('HistGBR',HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,
                                                            max_depth=4,l2_regularization=1.0,random_state=0))]:
        mdl.fit(Xt,et); p=mdl.predict(Xe2)
        ss=1-((ee-p)**2).sum()/((ee-ee.mean())**2).sum()
        print(f'   {nm:8s} held-out R2 of increment = {ss:+.4f}   corr={np.corrcoef(p,ee)[0,1]:+.4f}'
              f'   MAE {np.abs(ee-p).mean():.5f} vs persistence {np.abs(ee).mean():.5f}')
        OUT[f'{mode}_{nm}']=dict(R2=float(ss),corr=float(np.corrcoef(p,ee)[0,1]),
                                 mae=float(np.abs(ee-p).mean()),mae_persist=float(np.abs(ee).mean()))
        if mode=='full' and nm=='HistGBR': best_p=p
    # ms interpretation: next-beat placement error
    Iprev=np.exp(-(Ge['u_prev']))[M2]; Inext=np.exp(-(Ge['u']))[M2]     # up to the same const -> ratio ok
    Ip=(2*math.pi/(Ge['meter'][M2]*50.0))/np.exp(Ge['u_prev'][M2])
    In=(2*math.pi/(Ge['meter'][M2]*50.0))/np.exp(Ge['u'][M2])
    err_persist=np.abs(In-Ip)*1000
    print(f'   next-beat placement |error| ms: persistence mean {err_persist.mean():.2f} median {np.median(err_persist):.2f} p95 {np.percentile(err_persist,95):.2f}')
    OUT[f'{mode}_ms_persist']=dict(mean=float(err_persist.mean()),median=float(np.median(err_persist)),p95=float(np.percentile(err_persist,95)))
Xt,Gt=load('full','train'); Xe,Ge=load('full','eval'); M2=half(Ge)
Ip=(2*math.pi/(Ge['meter'][M2]*50.0))/np.exp(Ge['u_prev'][M2]); In=(2*math.pi/(Ge['meter'][M2]*50.0))/np.exp(Ge['u'][M2])
Iaud=Ip*np.exp(-best_p)
print(f'\naudio-corrected next-beat |error| ms: mean {np.abs(In-Iaud).mean()*1000:.2f} '
      f'median {np.median(np.abs(In-Iaud))*1000:.2f}  (persistence {np.abs(In-Ip).mean()*1000:.2f} / {np.median(np.abs(In-Ip))*1000:.2f})')
OUT['ms_audio']=dict(mean=float(np.abs(In-Iaud).mean()*1000),median=float(np.median(np.abs(In-Iaud))*1000))
# per-song scale heterogeneity: is there real per-song variation in |e| to exploit?
from collections import defaultdict
g=defaultdict(list)
for s,x in zip(Ge['stem'],Ge['u']-Ge['u_prev']): g[s].append(x)
mads=np.array([np.mean(np.abs(np.array(v))) for v in g.values()])
print(f'\nper-song mean|e|: n_songs={len(mads)} median {np.median(mads):.4f} IQR [{np.percentile(mads,25):.4f},{np.percentile(mads,75):.4f}] min {mads.min():.4f} max {mads.max():.4f}  ratio p90/p10 {np.percentile(mads,90)/max(np.percentile(mads,10),1e-9):.1f}x')
OUT['per_song_mad']=dict(n=len(mads),median=float(np.median(mads)),p10=float(np.percentile(mads,10)),p90=float(np.percentile(mads,90)))
json.dump(OUT,open('step5.json','w'),indent=1)
