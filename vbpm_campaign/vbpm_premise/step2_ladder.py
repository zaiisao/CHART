import sys, numpy as np, math, json
from scipy.optimize import minimize_scalar, minimize
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from core import *

Dtr = prep(build('train')); Dev = prep(build('eval'))
Ptr, Pev1, Pev2 = pairs(Dtr,'all'), pairs(Dev,'first'), pairs(Dev,'second')
Gtr, Gev1, Gev2 = gather(Ptr), gather(Pev1), gather(Pev2)
FPB = float(np.mean([25.7]))  # filled below
fpb = np.mean(np.concatenate([d['I'][d['valid']]*FPS for d in Dev]))
print(f"train pairs {Gtr['n']}  eval-1st {Gev1['n']}  eval-2nd(HELD-OUT) {Gev2['n']}  "
      f"songs {len(set(Gev2['stem']))}  mean frames/beat {fpb:.2f}")

def per_dataset(ll, G, name, base=None):
    rows={}
    for ds in sorted(set(G['dataset'])):
        m = G['dataset']==ds
        rows[ds]=dict(n=int(m.sum()), ll=float(ll[m].mean()))
        if base is not None: rows[ds]['gain']=float((ll-base)[m].mean())
    rows['POOLED']=dict(n=int(len(ll)), ll=float(ll.mean()))
    if base is not None: rows['POOLED']['gain']=float((ll-base).mean())
    return rows

def show(name, ll, base=None):
    s=f"{name:38s} HO {ll.mean():+.4f}"
    if base is not None: s+=f"  gain {(ll-base).mean():+.4f}"
    s+=" | "+" ".join(f"{ds}:{ll[G_ds==ds].mean():+.4f}"+(f"({(ll-base)[G_ds==ds].mean():+.4f})" if base is not None else "")
                      for ds in DS)
    print(s)
G_ds=Gev2['dataset']; DS=sorted(set(G_ds))

# free-nu student-t fit -------------------------------------------------------
def fit_t_freenu(G):
    def nll(th):
        c, ls, lnu = th
        return -logmass(G['ulo'],G['uhi'],G['u_prev']+c, math.exp(ls),'t',math.exp(lnu)+0.2).mean()
    r=minimize(nll,[0.,math.log(0.03),math.log(2.0)],method='Nelder-Mead',
               options=dict(maxiter=6000,xatol=1e-7,fatol=1e-9))
    c,ls,lnu=r.x; return dict(c=c,s=math.exp(ls),nu=math.exp(lnu)+0.2)
def sc_t(M,G,cshift=None,sscale=None):
    c = M['c'] if cshift is None else cshift
    s = M['s'] if sscale is None else sscale
    return logmass(G['ulo'],G['uhi'],G['u_prev']+c, s,'t',M['nu'])

OUT={}
# ---------------- (a) GLOBAL FIXED -------------------------------------------
A = fit_t_freenu(Gtr)
print(f"\n(a) GLOBAL fixed physical law: student-t  c={A['c']:+.5f} s={A['s']:.5f} nu={A['nu']:.2f}"
      f"   [fit on {Gtr['n']} train beats]")
ll_a = sc_t(A,Gev2); show('(a) GLOBAL fixed', ll_a)
OUT['a']=per_dataset(ll_a,Gev2,'a')
# gaussian reference (what a naive physical prior uses)
Mg = fit_rw(Gtr,'gauss'); ll_g = score_rw(Mg,Gev2); show('    [ref] global GAUSSIAN RW', ll_g, ll_a)

# ---------------- (b) PER-METER ----------------------------------------------
ll_b = np.zeros(Gev2['n'])
for m in sorted(set(Gtr['meter'])):
    tr = {k:(v[Gtr['meter']==m] if isinstance(v,np.ndarray) else v) for k,v in Gtr.items() if k!='n'}
    if (Gtr['meter']==m).sum() < 200:
        Mm = A
    else:
        tr['n']=int((Gtr['meter']==m).sum()); Mm = fit_t_freenu(tr)
    sel = Gev2['meter']==m
    if sel.sum()==0: continue
    ev = {k:(v[sel] if isinstance(v,np.ndarray) else v) for k,v in Gev2.items() if k!='n'}
    ll_b[sel] = sc_t(Mm,ev)
    print(f"    meter {m}: n_tr={(Gtr['meter']==m).sum():5d} n_ho={sel.sum():5d}  s={Mm['s']:.5f} nu={Mm['nu']:.2f}")
show('(b) PER-METER fixed', ll_b, ll_a); OUT['b']=per_dataset(ll_b,Gev2,'b',ll_a)

# ---------------- (b2) PER-BEAT-IN-BAR (position-conditioned fixed law) -------
ll_b2 = np.zeros(Gev2['n'])
for m in sorted(set(Gtr['meter'])):
  for j in range(max(m,1)):
    trm=(Gtr['meter']==m)&(Gtr['bib']==j); evm=(Gev2['meter']==m)&(Gev2['bib']==j)
    if evm.sum()==0: continue
    if trm.sum()<150: Mm=A
    else:
        tr={k:(v[trm] if isinstance(v,np.ndarray) else v) for k,v in Gtr.items() if k!='n'}; tr['n']=int(trm.sum())
        Mm=fit_t_freenu(tr)
    ev={k:(v[evm] if isinstance(v,np.ndarray) else v) for k,v in Gev2.items() if k!='n'}
    ll_b2[evm]=sc_t(Mm,ev)
rest = ll_b2==0
if rest.any():
    ev={k:(v[rest] if isinstance(v,np.ndarray) else v) for k,v in Gev2.items() if k!='n'}; ll_b2[rest]=sc_t(A,ev)
show('(b2) PER-METER x BEAT-IN-BAR fixed', ll_b2, ll_a); OUT['b2']=per_dataset(ll_b2,Gev2,'b2',ll_a)

# ---------------- (c) PER-SONG (fit 1st half of EVAL song, score 2nd) --------
from collections import defaultdict
idx1=defaultdict(list); idx2=defaultdict(list)
for i,st in enumerate(Gev1['stem']): idx1[st].append(i)
for i,st in enumerate(Gev2['stem']): idx2[st].append(i)
ll_c = np.zeros(Gev2['n']); ll_c_s = np.zeros(Gev2['n']); ll_c_mle=np.zeros(Gev2['n'])
nsong=0
for st,ii in idx2.items():
    ii=np.array(ii); jj=np.array(idx1.get(st,[]))
    ev={k:(v[ii] if isinstance(v,np.ndarray) else v) for k,v in Gev2.items() if k!='n'}
    if len(jj)<4:
        ll_c[ii]=sc_t(A,ev); ll_c_s[ii]=sc_t(A,ev); ll_c_mle[ii]=sc_t(A,ev); continue
    nsong+=1
    tr={k:(v[jj] if isinstance(v,np.ndarray) else v) for k,v in Gev1.items() if k!='n'}; tr['n']=len(jj)
    # c1: per-song SCALE only (shrunk to global by an empirical-Bayes weight fit on TRAIN songs)
    f=lambda ls: -logmass(tr['ulo'],tr['uhi'],tr['u_prev']+A['c'],math.exp(ls),'t',A['nu']).mean()
    r=minimize_scalar(f,bounds=(math.log(1e-4),math.log(1.0)),method='bounded')
    s_song=math.exp(r.x)
    ll_c_s[ii]=sc_t(A,ev,sscale=s_song)
    # c2: per-song scale + shift (full 2-param refit)
    g=lambda th: -logmass(tr['ulo'],tr['uhi'],tr['u_prev']+th[0],math.exp(th[1]),'t',A['nu']).mean()
    r2=minimize(g,[A['c'],math.log(s_song)],method='Nelder-Mead',options=dict(maxiter=2000))
    ll_c_mle[ii]=sc_t(A,ev,cshift=r2.x[0],sscale=math.exp(r2.x[1]))
    # c3: shrunk scale (sqrt-blend) -- more honest, less overfit
    w=len(jj)/(len(jj)+8.0); s_bl=math.exp(w*math.log(s_song)+(1-w)*math.log(A['s']))
    ll_c[ii]=sc_t(A,ev,sscale=s_bl)
print(f"    per-song fits: {nsong}/{len(idx2)} songs had >=4 first-half beats; "
      f"median first-half n={int(np.median([len(v) for v in idx1.values()]))}")
show('(c) PER-SONG scale (shrunk)', ll_c, ll_a);      OUT['c_shrunk']=per_dataset(ll_c,Gev2,'c',ll_a)
show('(c) PER-SONG scale (raw MLE)', ll_c_s, ll_a);   OUT['c_scale']=per_dataset(ll_c_s,Gev2,'c',ll_a)
show('(c) PER-SONG scale+shift (raw MLE)', ll_c_mle, ll_a); OUT['c_full']=per_dataset(ll_c_mle,Gev2,'c',ll_a)

np.savez('/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise/step2.npz',
         ll_a=ll_a, ll_b=ll_b, ll_b2=ll_b2, ll_c=ll_c, ll_c_s=ll_c_s, ll_c_mle=ll_c_mle,
         A_c=A['c'], A_s=A['s'], A_nu=A['nu'], fpb=fpb)
json.dump(OUT, open('/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise/step2.json','w'), indent=1)
print('\nnats/FRAME (divide by frames-per-beat %.2f):'%fpb)
for nm,ll in [('a',ll_a),('b',ll_b),('b2',ll_b2),('c_shrunk',ll_c),('c_full',ll_c_mle)]:
    print(f"   {nm:10s} {ll.mean()/fpb:+.5f}   gain/frame {(ll-ll_a).mean()/fpb:+.6f}")
