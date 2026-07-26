"""What makes the log_tempo side-channel free? Measure prior concentrations/scales
and the actual likelihood surfaces (phase vs log_tempo)."""
import sys, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms')
import variant_b as VB
from vbpm.distributions import TWO_PI
from audit_common import load_split, ideal_barphase, FPS
from common import targets
from vbpm.evaluate import _estimate_meter
ARMS='/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms'; DEV='cuda:0'

ck=torch.load(f'{ARMS}/arm_i_ii_bern.pt',map_location=DEV)
model=VB.BarPointerVAE_B(h_dim=2,hidden=128,num_meters=4,obs_dim=2,obs_type='bern').to(DEV)
model.load_state_dict(ck['model']); model.eval()

songs=load_split('eval',cap=12)
d=np.load(f'{ARMS}/act_eval.npz',allow_pickle=True)
with torch.no_grad():
  RHO=[];SLV=[];SDV=[];AR=[]
  for s in songs:
    a=np.clip(np.asarray(d[s['stem']+'|act'],np.float32),1e-4,1-1e-4)
    h=torch.from_numpy(a).unsqueeze(0).to(DEV)
    ctx=model.encode_prior(h)[0]
    RHO.append(model.prior_phase_conc(ctx).cpu().numpy())
    SLV.append(model.prior_level_scale(ctx).cpu().numpy())
    SDV.append(model.prior_dev_scale(ctx).cpu().numpy())
    AR.append(model.prior_dev_coef(ctx).cpu().numpy())
  for nm,v in (('prior_phase_rho (wrapped-Cauchy conc; 0=UNIFORM,1=delta)',RHO),
               ('prior_level_sigma',SLV),('prior_dev_sigma',SDV),('prior_dev_ar',AR)):
    x=np.concatenate(v); print(f'  {nm}: mean {x.mean():.4f} median {np.median(x):.4f} p5 {np.percentile(x,5):.4f} p95 {np.percentile(x,95):.4f} max {x.max():.4f}')
  print(f'  level_ar={float(model.level_ar()):.4f}  tempo_dof={float(model.tempo_dof()):.3f}  level_offset={float(model.level_offset):.3f}')

  # posterior phase concentration (from the ELBO recursion, arbitrary z_prev -- probe head directly)
  s=songs[0]; a=np.clip(np.asarray(d[s['stem']+'|act'],np.float32),1e-4,1-1e-4)
  T=a.shape[0]; h=torch.from_numpy(a).unsqueeze(0).to(DEV)
  b,db=targets(s['beats'],s['downs'],0,T)
  pc=model.encode_posterior(h,torch.from_numpy(b).unsqueeze(0).to(DEV))[0]
  zp=model.z_features(F.one_hot(torch.tensor([3]*T,device=DEV),4).float(),
                      torch.rand(T,device=DEV)*TWO_PI, torch.full((T,),-2.77,device=DEV))
  q=model.unpack(model.post_head(torch.cat([pc,zp],-1)))
  print(f'  POSTERIOR phase_rho: mean {float(q[2].mean()):.4f} max {float(q[2].max()):.4f}   '
        f'level_sigma mean {float(q[4].mean()):.3f}  dev_sigma mean {float(q[6].mean()):.3f}')

  # ---- likelihood SURFACES on real eval songs ----
  print('\n  EMISSION LOG-LIK SURFACE (mean per frame over song), meter=4')
  for s in songs[:6]:
    a=np.clip(np.asarray(d[s['stem']+'|act'],np.float32),1e-4,1-1e-4); T=a.shape[0]
    o=torch.from_numpy(a).to(DEV)
    ref=s['beats']; dref=s['downs']
    phi=ideal_barphase(dref,T,FPS,mode='extrap')
    if phi is None: continue
    m=_estimate_meter(ref,dref)
    mt=F.one_hot(torch.tensor([m-1]*T,device=DEV),4).float()
    lt0=math.log(TWO_PI/(float(np.median(np.diff(dref)))*FPS))
    ph=torch.from_numpy(phi).float().to(DEV)
    def L(p,l): return float(model.obs_logp(model.z_features(mt,p,torch.full((T,),l,device=DEV)),o).mean())
    lt_true=L(ph,lt0)
    offs=[L((ph+TWO_PI*k/12)%TWO_PI,lt0) for k in range(1,12)]
    # tempo sweep at fixed (true) phase
    sweep={l:L(ph,l) for l in (-8.,-5.,-3.5,lt0,-2.,0.,2.,4.,6.)}
    best=max(sweep,key=sweep.get)
    print(f'   {s["stem"][:38]:38s} ll@true_phi={lt_true:.4f} spread_over_11_phase_offsets={max(offs)-min(offs):.5f} '
          f'contrast={math.exp(lt_true-np.mean(offs)):.5f} | tempo-sweep argmax={best:+.2f} ll {sweep[best]:.4f} '
          f'(at true lt {lt0:.2f}: {sweep[lt0]:.4f}) SPREAD={max(sweep.values())-min(sweep.values()):.4f}')
