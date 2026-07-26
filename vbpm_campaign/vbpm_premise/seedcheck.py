"""PF Monte-Carlo stability of the response curve: 4 independent seed offsets at key points."""
import sys, json, math, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_final')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from emission import PhaseEmission, load_act, load_split, METERS, TWO_PI, FPS, _estimate_meter
import pf_corrupt as PFC
from run_exp2 import score_traj
OPS={'pub':dict(K=600,alpha=0.25),'cell':dict(K=300,alpha=1.0)}
BASE=dict(sigma_lt=0.05,sigma_phi=0.03,p_switch=0.005,noise='gauss',fps=FPS)
_G={}
def init():
    tr=load_split('train'); at=load_act('train'); ev=load_split('eval'); ae=load_act('eval')
    emis=PhaseEmission(bins_per_beat=24,likelihood='gauss',smooth=0.0).fit(tr,at,phase_mode='downbeat')
    prior=np.zeros(5)
    for s in tr:
        m=_estimate_meter(s['beats'],s['downs'])
        if m in METERS: prior[m]+=1
    _G.update(emis=emis,prior=prior,ev=ev,ae=ae)
def one(job):
    key,pf,op,sd,i=job
    emis,prior,ev,ae=_G['emis'],_G['prior'],_G['ev'],_G['ae']
    s=ev[i]; act=ae.get(s['stem']); T=min(len(act),s['T'])
    ref=s['beats'][s['beats']<T/FPS]; dref=s['downs'][s['downs']<T/FPS]
    if len(ref)<3: return None
    m_gt=_estimate_meter(s['beats'],s['downs']); LL=emis.padded_table(act[:T])
    kw=dict(BASE); kw.update(OPS[op]); kw.update(pf)
    out=PFC.particle_filter(LL,emis.nb,meter_prior=prior,seed=sd*100000+i,**kw)
    m_pf=int(np.bincount(out['meter_path']).argmax())
    r=score_traj(out['phase_path'],m_pf,ref,dref,T)
    return key,op,sd,r['beat_F'],out['logZ_per_frame']
pts=[('rev0.0',dict(mode='rev',p_rev=0.0)),('rev0.15',dict(mode='rev',p_rev=0.15)),
     ('rev0.30',dict(mode='rev',p_rev=0.30)),('rev0.50',dict(mode='rev',p_rev=0.50)),
     ('cau0.99',dict(mode='cauchy',rho_phase=0.99)),('cau0.0',dict(mode='cauchy',rho_phase=0.0))]
jobs=[(k,pf,op,sd,i) for k,pf in pts for op in OPS for sd in range(1,5) for i in range(79)]
print(len(jobs),'runs',flush=True)
acc={}
with Pool(48,initializer=init) as pool:
    for res in pool.imap_unordered(one,jobs,chunksize=1):
        if res is None: continue
        k,op,sd,f,lz=res
        acc.setdefault((k,op,sd),[]).append((f,lz))
print(f"{'point':>9} {'op':>5} | " + ' '.join(f'seed{d}' for d in range(1,5)) + "   mean+-sd   logZ mean+-sd")
for k,_ in pts:
    for op in OPS:
        fs=[float(np.mean([x[0] for x in acc[(k,op,sd)]])) for sd in range(1,5)]
        lz=[float(np.mean([x[1] for x in acc[(k,op,sd)]])) for sd in range(1,5)]
        print(f"{k:>9} {op:>5} | " + ' '.join(f'{v:.4f}' for v in fs)
              + f"   {np.mean(fs):.4f}+-{np.std(fs):.4f}   {np.mean(lz):.4f}+-{np.std(lz):.4f}")
