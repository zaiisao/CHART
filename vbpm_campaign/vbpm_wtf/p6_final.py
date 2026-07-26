import sys, math, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms')
import variant_b as VB
from audit_common import load_split
ARMS='/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms'; DEV='cuda:0'
TWO_PI=2*math.pi

print('--- (a) arm_ii LayerMerge gradient ---')
import arm_ii as A2
torch.manual_seed(0)
merge=A2.LayerMerge().to(DEV)
model=VB.BarPointerVAE_B(h_dim=2,hidden=128,num_meters=4,obs_dim=2,obs_type='bern').to(DEV)
params=list(merge.parameters())+list(model.parameters())
opt=torch.optim.AdamW(params,lr=3e-4)
f=torch.randn(2,13,64,768,device=DEV); o=torch.rand(2,64,2,device=DEV)
b=(torch.rand(2,64,device=DEV)<0.04).float(); d=(torch.rand(2,64,device=DEV)<0.01).float()
w0=merge.layer_logits.detach().clone()
loss,_=VB.elbo_b(model,o,b,d,o,temperature=1.0,beta=1.0)
loss.backward()
print('   merge.layer_logits.grad =',merge.layer_logits.grad)
opt.step()
print('   layer_logits changed after opt.step()?',bool((merge.layer_logits.detach()-w0).abs().max()>0),
      ' max|delta| =',float((merge.layer_logits.detach()-w0).abs().max()))
print('   model params receiving grad:',sum(1 for p in model.parameters() if p.grad is not None),'/',sum(1 for _ in model.parameters()))
nog=[n for n,p in model.named_parameters() if p.grad is None]
print('   model params with NO grad:',nog)

print('\n--- (c) train vs deploy observation identity ---')
ev=load_split('eval',cap=3)
c=A2.build_obs_cache(ev,f'{ARMS}/act_eval.npz','head_bern')
raw=np.load(f'{ARMS}/act_eval.npz',allow_pickle=True)
s=ev[0]; a=np.asarray(raw[s['stem']+'|act'],np.float32)
print('   build_obs_cache == clip(raw,1e-4,1-1e-4):',np.allclose(c[s['stem']],np.clip(a,1e-4,1-1e-4)),
      ' range',c[s['stem']].min(),c[s['stem']].max())
print('   train path: elbo_b(model, h=o, ..., obs=o) ; deploy: particle_filter(model, h=obs, obs=obs) -> SAME tensor, no z-scoring for bern. OK')

print('\n--- (d) beta schedule ---')
for st in (1,300,599,600,601,1200):
    print(f'   step {st:5d}  beta={min(1.0,st/600):.3f}  temp={1.0+(0.3-1.0)*min(st/1200,1.0):.3f}')

print('\n--- exact obs_contrast values from the shipped JSON ---')
for tag in ('ii_bern','i_bern'):
    J=json.load(open(f'{ARMS}/arm_i_{tag}.json'))
    rows=J['rows']['K300_a1.0']
    v=np.array([r['obs_contrast'] for r in rows if not math.isnan(r.get('obs_contrast',float('nan')))])
    print(f'   {tag}: n={len(v)} mean {v.mean():.6f} min {v.min():.6f} max {v.max():.6f}')

print('\n--- emission as an AMPLITUDE code: p(o=1|z) vs log_tempo ---')
ck=torch.load(f'{ARMS}/arm_i_ii_bern.pt',map_location=DEV)
m=VB.BarPointerVAE_B(h_dim=2,hidden=128,num_meters=4,obs_dim=2,obs_type='bern').to(DEV); m.load_state_dict(ck['model']); m.eval()
with torch.no_grad():
    for lt in (-10,-8,-6,-4,-2.77,-2,0,2,4,6):
        ph=torch.rand(4096,device=DEV)*TWO_PI
        z=m.z_features(F.one_hot(torch.tensor([3]*4096,device=DEV),4).float(),ph,torch.full((4096,),float(lt),device=DEV))
        p=torch.sigmoid(m.h_dec(z))
        print(f'   log_tempo={lt:+6.2f} (phidot={math.exp(lt):9.4f} rad/fr) -> p(beat_act)={float(p[:,0].mean()):.4f} p(db_act)={float(p[:,1].mean()):.4f}  (std over phi: {float(p[:,0].std()):.5f})')
    print('   REAL activation channel means: beat 0.1095  downbeat 0.0357')
