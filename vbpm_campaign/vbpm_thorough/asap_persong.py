"""X2(c+): decompose the ASAP room in nats/beat:
steady-global -> ASAP-global (corpus rescale) -> per-song scale (fit on 1st half,
score 2nd half; a HISTORY-accessible latent) -> per-song mean+scale."""
import sys, math, json, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough')
from core import prep, pairs, gather, logmass
from asap_data import build_asap
from scipy.optimize import minimize
from collections import defaultdict

steady=json.load(open('/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise/step1.json'))
asapfit=json.load(open('asap_step1.json'))
NU=2.0
c_st,ls_st=steady['t2.0_RW']['th']; c_as,ls_as=asapfit['t2.0_RW']['th']
Dev=prep(build_asap('eval'))
G1=gather(pairs(Dev,'first')); G2=gather(pairs(Dev,'second'))
def score(G,c,ls,mask=None):
    ll=logmass(G['ulo'],G['uhi'],G['u_prev']+c,math.exp(ls),'t',NU)
    return ll if mask is None else ll[mask]
ll_st=score(G2,c_st,ls_st); ll_as=score(G2,c_as,ls_as)
# per-song: fit on first-half pairs, score second half
i1=defaultdict(list); i2=defaultdict(list)
for i,s in enumerate(G1['stem']): i1[s].append(i)
for i,s in enumerate(G2['stem']): i2[s].append(i)
ll_scale=np.zeros(G2['n']); ll_ms=np.zeros(G2['n'])
for s,ii2 in i2.items():
    ii1=np.array(i1[s]); ii2=np.array(ii2)
    sub=lambda G,ii:{k:v[ii] for k,v in G.items() if isinstance(v,np.ndarray)}
    g1=sub(G1,ii1)
    f1=lambda th: -logmass(g1['ulo'],g1['uhi'],g1['u_prev']+c_as,math.exp(th[0]),'t',NU).mean()
    r1=minimize(f1,[ls_as],method='Nelder-Mead',options=dict(maxiter=800))
    f2=lambda th: -logmass(g1['ulo'],g1['uhi'],g1['u_prev']+th[0],math.exp(th[1]),'t',NU).mean()
    r2=minimize(f2,[c_as,ls_as],method='Nelder-Mead',options=dict(maxiter=1500))
    g2=sub(G2,ii2)
    ll_scale[ii2]=logmass(g2['ulo'],g2['uhi'],g2['u_prev']+c_as,math.exp(r1.x[0]),'t',NU)
    ll_ms[ii2]  =logmass(g2['ulo'],g2['uhi'],g2['u_prev']+r2.x[0],math.exp(r2.x[1]),'t',NU)
print(f"t2-RW nats/beat on ASAP eval 2nd half (n={G2['n']}):")
print(f"  steady-global           {ll_st.mean():+.4f}")
print(f"  ASAP-global             {ll_as.mean():+.4f}   (+{ll_as.mean()-ll_st.mean():.4f} corpus rescale)")
print(f"  per-song scale (1st half){ll_scale.mean():+.4f}   (+{ll_scale.mean()-ll_as.mean():.4f} per-song scale)")
print(f"  per-song mean+scale     {ll_ms.mean():+.4f}   (+{ll_ms.mean()-ll_scale.mean():.4f} per-song mean)")
json.dump(dict(steady=float(ll_st.mean()),asap_global=float(ll_as.mean()),
               persong_scale=float(ll_scale.mean()),persong_meanscale=float(ll_ms.mean()),
               n=int(G2['n'])), open('asap_persong.json','w'), indent=1)
print('DONE')
