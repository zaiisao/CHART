"""Regression: corrupted PF with mode='none' must be BIT-IDENTICAL to vbpm_final/pf.py."""
import sys, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_final')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from emission import PhaseEmission, load_act, load_split, METERS, TWO_PI, FPS, _estimate_meter
import pf as PF_ORIG
import pf_corrupt as PF_NEW

tr=load_split('train'); at=load_act('train')
ev=load_split('eval');  ae=load_act('eval')
emis=PhaseEmission(bins_per_beat=24,likelihood='gauss',smooth=0.0).fit(tr,at,phase_mode='downbeat')
prior=np.zeros(5)
for s in tr:
    m=_estimate_meter(s['beats'],s['downs'])
    if m in METERS: prior[m]+=1
ok=0
for i,s in enumerate(ev[:6]):
    act=ae[s['stem']]; T=min(len(act),s['T'])
    LL=emis.padded_table(act[:T])
    kw=dict(nb=emis.nb,K=600,alpha=0.25,sigma_lt=0.05,sigma_phi=0.03,p_switch=0.005,
            meter_prior=prior,fps=FPS,seed=1234+i,noise='gauss')
    a=PF_ORIG.particle_filter(LL,**kw)
    b=PF_NEW.particle_filter(LL,mode='none',**kw)
    for k in ('phase_path','phase_mean','phase_map','meter_path'):
        assert np.array_equal(np.asarray(a[k]),np.asarray(b[k])), (s['stem'],k)
    assert a['ess']==b['ess'] and a['n_resample']==b['n_resample']
    ok+=1
print("BIT-IDENTICAL on %d/%d songs (phase_path, phase_mean, phase_map, meter_path, ess, n_resample)"%(ok,6))
