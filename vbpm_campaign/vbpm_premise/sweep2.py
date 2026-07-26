"""PREMISE 4 main run: beat_F AND PF log-evidence vs transition sloppiness, at TWO matched
operating points.  Fit on 147 TRAIN songs, score on 79 EVAL (fold-0) songs.

  op='pub'  : alpha=0.25 K=600  -- the published physical operating point (FINAL_eval,
              pf_meter_path beat_F 0.7505, frac_neg 0.012).
  op='cell' : alpha=1.00 K=300  -- the operating point at which the LEARNED VAE transition
              was measured (cell_learned_sup: 0.5920 / frac_neg 0.503) and its matched
              physical control (cell_simple_sup: 0.7647 / frac_neg 0.012).
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
    ci,cfg,op,i=job
    emis,prior,ev,ae=_G['emis'],_G['prior'],_G['ev'],_G['ae']
    s=ev[i]; act=ae.get(s['stem'])
    if act is None: return None
    T=min(len(act),s['T'])
    ref=s['beats'][s['beats']<T/FPS]; dref=s['downs'][s['downs']<T/FPS]
    if len(ref)<3: return None
    m_gt=_estimate_meter(s['beats'],s['downs'])
    LL=emis.padded_table(act[:T])
    kw=dict(BASE); kw.update(OPS[op]); kw.update(cfg['pf'])
    out=PFC.particle_filter(LL,emis.nb,meter_prior=prior,seed=1234+i,**kw)
    m_pf=int(np.bincount(out['meter_path']).argmax())
    base=dict(stem=s['stem'],dataset=s['dataset'],T=T,n_true=len(ref),n_true_db=len(dref),
              ess=out['ess'],obs_contrast=float('nan'),
              meter_ok=float(m_pf==m_gt), logZpf=out['logZ_per_frame'])
    return ci,op,{'path':{**base,**score_traj(out['phase_path'],m_gt,ref,dref,T)},
                  'pf_meter_path':{**base,**score_traj(out['phase_path'],m_pf,ref,dref,T)},
                  'mean':{**base,**score_traj(out['phase_mean'],m_gt,ref,dref,T)}}

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
    ap=argparse.ArgumentParser(); ap.add_argument('--tag',default='main2'); a=ap.parse_args()
    cfgs=[]
    def add(n,ax,x,**pf): cfgs.append(dict(name=n,axis=ax,x=x,pf=pf))
    for p in [0.0,0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]:
        add(f'rev_p{p}','rev_p',p,mode='rev',p_rev=p)
    for r in [1.0,0.9995,0.999,0.995,0.99,0.98,0.95,0.90,0.80,0.60,0.30,0.0]:
        add(f'cauchy_rho{r}','cauchy_rho',r,mode='cauchy',rho_phase=r)
    for sp in [0.03,0.06,0.10,0.15,0.20,0.30,0.40,0.80,1.50,3.00]:
        add(f'gauss_sp{sp}','gauss_sphi',sp,mode='none',sigma_phi=sp)
    for sl in [0.05,0.10,0.20,0.40,0.80,1.60]:
        add(f'lt_s{sl}','sigma_lt',sl,mode='none',sigma_lt=sl)
    ops=list(OPS)
    jobs=[(ci,c,op,i) for ci,c in enumerate(cfgs) for op in ops for i in range(79)]
    print(f'{len(cfgs)} configs x {len(ops)} ops x 79 songs = {len(jobs)} PF runs',flush=True)
    t0=time.time()
    store={(ci,op):{'path':[],'pf_meter_path':[],'mean':[]} for ci in range(len(cfgs)) for op in ops}
    with Pool(48,initializer=init) as pool:
        for n,res in enumerate(pool.imap_unordered(one,jobs,chunksize=1)):
            if res is None: continue
            ci,op,r=res
            for k in store[(ci,op)]: store[(ci,op)][k].append(r[k])
            if (n+1)%500==0: print(f'  {n+1}/{len(jobs)} {time.time()-t0:.0f}s',flush=True)
    res=[]
    for ci,c in enumerate(cfgs):
        for op in ops:
            e=dict(name=c['name'],axis=c['axis'],x=c['x'],op=op)
            for k in store[(ci,op)]:
                e[k]=agg(store[(ci,op)][k])
                e[k+'_by_ds']={ds:agg([r for r in store[(ci,op)][k] if r['dataset']==ds])
                               for ds in sorted({r['dataset'] for r in store[(ci,op)][k]})}
            e['per_song_beatF']={r['stem']:r['beat_F'] for r in store[(ci,op)]['pf_meter_path']}
            e['per_song_logZpf']={r['stem']:r['logZpf'] for r in store[(ci,op)]['pf_meter_path']}
            res.append(e)
            d=e['pf_meter_path']
            print(f"[{op:4s}][{c['name']:16s}] beatF={d['beat_F']:.4f} blind={d['blind_best']:.4f} "
                  f"MARG={d['margin']:+.4f} db={d['db_F']:.4f} path={e['path']['beat_F']:.4f} "
                  f"fneg={d['frac_neg']:.3f} adv={d['mean_adv']:.4f} j/a={d['jitter_over_adv']:.2f} "
                  f"logZ/fr={d['logZpf']:+.4f} nr={d['n_ratio']:.2f} ESS={d['ess']:.0f} "
                  f"met={d['meter_ok']:.2f}",flush=True)
    json.dump(res,open(f'{OUT}/sweep2_{a.tag}.json','w'),indent=1,default=float)
    print('DONE',time.time()-t0,flush=True)

if __name__=='__main__': main()
