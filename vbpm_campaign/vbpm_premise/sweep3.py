"""PREMISE 4, direct lambda simulation.

Physical-prior anchoring  L = ELBO - lambda*KL(p_psi || p_physical)  pulls the deployed
transition kernel along the log-linear (geometric) bridge between the learned/sloppy kernel
and the physical kernel:
        g_w(u) ∝ q_sloppy(u)^(1-w) * p_physical(u)^w ,     w = lambda/(1+lambda)
(u = the pointer-increment offset around the deterministic advance exp(lt)).
w=0 is the pure sloppy kernel, w=1 the pure physical kernel; lambda=1 -> w=0.5.

Two sloppy endpoints, both calibrated to the LEARNED prior's measured behaviour
(cell_learned_sup: beat_F 0.5920, frac_neg 0.5028, jitter/advance 153):
  qU  : uniform on the circle  -- FAMILY-matched (the learned prior is a wrapped Cauchy whose
        concentration has collapsed); reproduces frac_neg ~ 0.50.
  qR  : the physical step size with its SIGN flipped w.p. 0.5 -- SCORE-matched
        (beat_F 0.617 at the learned prior's own operating point vs its 0.592).
"""
import sys, json, math, time, argparse
import numpy as np
from multiprocessing import Pool

sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_final')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from emission import PhaseEmission, load_act, load_split, METERS, TWO_PI, FPS, _estimate_meter
import pf_corrupt as PFC
from run_exp2 import score_traj

OUT='/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise'
OPS={'pub': dict(K=600, alpha=0.25), 'cell': dict(K=300, alpha=1.0)}
BASE=dict(sigma_lt=0.05, sigma_phi=0.03, p_switch=0.005, noise='gauss', fps=FPS)
S_PHYS=0.03; STEP=0.0832                 # measured mean bar advance, rad/frame (eval fold)
NG=200001
UG=np.linspace(-math.pi, math.pi, NG)[:-1]
DU=UG[1]-UG[0]

def wn(mu,s,nw=6):
    d=np.zeros_like(UG)
    for k in range(-nw,nw+1): d+=np.exp(-0.5*((UG-mu+k*TWO_PI)/s)**2)/(s*math.sqrt(TWO_PI))
    return d
def norm(d): return d/(d.sum()*DU)

P_PHYS=norm(wn(0.0,S_PHYS))
Q={'qU': norm(np.full_like(UG,1.0/TWO_PI)),
   'qR': norm(0.5*wn(STEP,S_PHYS)+0.5*wn(-STEP,S_PHYS))}   # offset around +step -> +/- step

def bridge(qname,w):
    q=Q[qname]
    lg=(1-w)*np.log(np.maximum(q,1e-300))+w*np.log(np.maximum(P_PHYS,1e-300))
    g=np.exp(lg-lg.max()); g=norm(g)
    cdf=np.cumsum(g)*DU; cdf/=cdf[-1]
    kl=float(np.sum(g*np.log(np.maximum(g,1e-300)/np.maximum(P_PHYS,1e-300)))*DU)
    return (UG,cdf), kl

_G={}
def init():
    tr=load_split('train'); at=load_act('train')
    ev=load_split('eval');  ae=load_act('eval')
    emis=PhaseEmission(bins_per_beat=24,likelihood='gauss',smooth=0.0).fit(tr,at,phase_mode='downbeat')
    prior=np.zeros(5)
    for s in tr:
        m=_estimate_meter(s['beats'],s['downs'])
        if m in METERS: prior[m]+=1
    _G.update(emis=emis,prior=prior,ev=ev,ae=ae)

def one(job):
    ci,samp,op,i=job
    emis,prior,ev,ae=_G['emis'],_G['prior'],_G['ev'],_G['ae']
    s=ev[i]; act=ae.get(s['stem'])
    if act is None: return None
    T=min(len(act),s['T'])
    ref=s['beats'][s['beats']<T/FPS]; dref=s['downs'][s['downs']<T/FPS]
    if len(ref)<3: return None
    m_gt=_estimate_meter(s['beats'],s['downs'])
    LL=emis.padded_table(act[:T])
    kw=dict(BASE); kw.update(OPS[op])
    out=PFC.particle_filter(LL,emis.nb,meter_prior=prior,seed=1234+i,mode='blend',
                            blend_sampler=samp,**kw)
    m_pf=int(np.bincount(out['meter_path']).argmax())
    base=dict(stem=s['stem'],dataset=s['dataset'],T=T,n_true=len(ref),n_true_db=len(dref),
              ess=out['ess'],obs_contrast=float('nan'),meter_ok=float(m_pf==m_gt),
              logZpf=out['logZ_per_frame'])
    return ci,op,{'path':{**base,**score_traj(out['phase_path'],m_gt,ref,dref,T)},
                  'pf_meter_path':{**base,**score_traj(out['phase_path'],m_pf,ref,dref,T)}}

def agg(rows,keys=('beat_F','db_F','blind_best','blind_db_best','frac_neg','mean_adv',
                   'jitter','jitter_over_adv','ess','meter_ok','logZpf')):
    d={}
    for k in keys:
        v=[r[k] for r in rows if isinstance(r.get(k),float) and not math.isnan(r[k])]
        d[k]=float(np.mean(v)) if v else float('nan')
    ne=sum(r['n_est'] for r in rows); nt=sum(r['n_true'] for r in rows)
    d['n_ratio']=ne/max(nt,1); d['n_songs']=len(rows)
    d['margin']=d['beat_F']-d['blind_best']; d['margin_db']=d['db_F']-d['blind_db_best']
    return d

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tag',default='lam'); a=ap.parse_args()
    lams=[0.0,0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0,10.0,100.0,float('inf')]
    cfgs=[]
    for qn in ('qU','qR'):
        for lam in lams:
            w=1.0 if math.isinf(lam) else lam/(1.0+lam)
            samp,kl=bridge(qn,w)
            cfgs.append(dict(name=f'{qn}_lam{lam}',q=qn,lam=lam,w=w,kl=kl,samp=samp))
    ops=list(OPS)
    jobs=[(ci,c['samp'],op,i) for ci,c in enumerate(cfgs) for op in ops for i in range(79)]
    print(f'{len(cfgs)} bridge points x {len(ops)} ops x 79 songs = {len(jobs)} PF runs',flush=True)
    t0=time.time()
    store={(ci,op):{'path':[],'pf_meter_path':[]} for ci in range(len(cfgs)) for op in ops}
    with Pool(48,initializer=init) as pool:
        for n,res in enumerate(pool.imap_unordered(one,jobs,chunksize=1)):
            if res is None: continue
            ci,op,r=res
            for k in store[(ci,op)]: store[(ci,op)][k].append(r[k])
            if (n+1)%500==0: print(f'  {n+1}/{len(jobs)} {time.time()-t0:.0f}s',flush=True)
    res=[]
    for ci,c in enumerate(cfgs):
        for op in ops:
            e=dict(name=c['name'],q=c['q'],lam=c['lam'],w=c['w'],kl_to_phys=c['kl'],op=op)
            for k in store[(ci,op)]:
                e[k]=agg(store[(ci,op)][k])
                e[k+'_by_ds']={ds:agg([r for r in store[(ci,op)][k] if r['dataset']==ds])
                               for ds in sorted({r['dataset'] for r in store[(ci,op)][k]})}
            e['per_song_beatF']={r['stem']:r['beat_F'] for r in store[(ci,op)]['pf_meter_path']}
            e['per_song_logZpf']={r['stem']:r['logZpf'] for r in store[(ci,op)]['pf_meter_path']}
            res.append(e)
            d=e['pf_meter_path']
            print(f"[{op:4s}][{c['q']}] lam={c['lam']:<8} w={c['w']:.4f} KL={c['kl']:9.3f} "
                  f"| beatF={d['beat_F']:.4f} blind={d['blind_best']:.4f} MARG={d['margin']:+.4f} "
                  f"db={d['db_F']:.4f} fneg={d['frac_neg']:.3f} logZ/fr={d['logZpf']:+.4f} "
                  f"penalty=lam*KL={ (0.0 if c['lam']==0 else (float('inf') if math.isinf(c['lam']) else c['lam']*c['kl'])):.4g} "
                  f"nr={d['n_ratio']:.2f}",flush=True)
    json.dump(res,open(f'{OUT}/sweep3_{a.tag}.json','w'),indent=1,default=float)
    print('DONE',time.time()-t0,flush=True)

if __name__=='__main__': main()
