import sys, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms')
import variant_b as VB
from audit_common import load_split, FPS
DEV='cuda:0'; ARMS='/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms'; TWO_PI=2*math.pi
HDIM={'i_bern':768,'i_gauss':768,'ii_bern':2}
class LM(nn.Module):
    def __init__(s,n=13):
        super().__init__(); s.layer_logits=nn.Parameter(torch.zeros(n))
    def forward(s,f): return torch.einsum('l,bltf->btf',torch.softmax(s.layer_logits,0),f)

@torch.no_grad()
def main():
    A=np.load(f'{ARMS}/act_eval.npz',allow_pickle=True)
    ev=load_split('eval',with_feats=False)
    for tag in ['i_bern','ii_bern','i_gauss']:
        ck=torch.load(f'{ARMS}/arm_i_{tag}.pt',map_location=DEV)
        ot='bern' if ck['obs']=='head_bern' else 'gauss'
        m=VB.BarPointerVAE_B(h_dim=HDIM[tag],hidden=128,num_meters=4,obs_dim=2,obs_type=ot).to(DEV)
        m.load_state_dict(ck['model']); m.eval()
        torch.manual_seed(0)
        m0=VB.BarPointerVAE_B(h_dim=HDIM[tag],hidden=128,num_meters=4,obs_dim=2,obs_type=ot).to(DEV)
        print(f'\n===== {tag} =====')
        # (i) how far did the first layers move from a fresh init?
        for nm in ['h_dec','decoder']:
            W=getattr(m,nm)[0].weight.detach(); W0=getattr(m0,nm)[0].weight.detach()
            print(f'  {nm:8s} |W|_2={W.norm():.3f}  fresh-init |W0|_2={W0.norm():.3f}  '
                  f'per-col |W| cos/sin/logT = {W[:,0].norm():.3f}/{W[:,1].norm():.3f}/{W[:,2].norm():.3f}')
        # (ii) prior + posterior phase concentration rho on real audio
        rows=[]
        for s in ev[:20]:
            T=min(s['T'],3000)
            if HDIM[tag]==2:
                a=np.clip(np.asarray(A[s['stem']+'|act'],np.float32),1e-4,1-1e-4)
                h=torch.from_numpy(a[:T]).unsqueeze(0).to(DEV)
            else:
                d=np.load(s['path'],allow_pickle=True)
                f=torch.from_numpy(np.asarray(d['feats'][:,:T,:],np.float32)).unsqueeze(0).to(DEV)
                mg=LM().to(DEV); mg.load_state_dict(ck['merge']); h=mg(f)
            ctx=m.encode_prior(h)[0]
            rho=m.prior_phase_conc(ctx); s_lv=m.prior_level_scale(ctx); s_dv=m.prior_dev_scale(ctx)
            rows.append((float(rho.mean()),float(rho.max()),float(s_lv.mean()),float(s_dv.mean())))
        r=np.array(rows)
        print(f'  PRIOR phase rho: mean={r[:,0].mean():.5f} max={r[:,1].max():.5f}   '
              f'(rho=0 => wrapped Cauchy == UNIFORM on the circle)')
        print(f'  prior level sigma={r[:,2].mean():.4f}  dev sigma={r[:,3].mean():.4f}  '
              f'level_ar={float(m.level_ar()):.4f} dof={float(m.tempo_dof()):.3f}')
        # (iii) obs emission vs the best CONSTANT predictor, on eval
        tot=[]; totc=[]; mrate=[]
        for s in ev[:20]:
            T=min(s['T'],3000)
            a=np.clip(np.asarray(A[s['stem']+'|act'],np.float32),1e-4,1-1e-4)[:T]
            o=torch.from_numpy(a if ot=='bern' else np.log(a/(1-a))).to(DEV)
            G=torch.rand(T,device=DEV)*TWO_PI
            mt=F.one_hot(torch.full((T,),3,device=DEV,dtype=torch.long),4).float()
            z=m.z_features(mt,G,torch.full((T,),-2.66,device=DEV))
            tot.append(float(m.obs_logp(z,o).mean()))
            if ot=='bern':
                p=o.mean(0,keepdim=True).expand(T,-1)
                totc.append(float(-F.binary_cross_entropy(p,o,reduction='none').sum(-1).mean()))
            else:
                mu=o.mean(0,keepdim=True); sd=o.std(0,keepdim=True)
                totc.append(float((-0.5*((o-mu)/sd)**2-torch.log(sd)-0.5*math.log(TWO_PI)).sum(-1).mean()))
            mrate.append(float(a.mean()))
        print(f'  obs log-lik/frame: trained h_dec={np.mean(tot):+.4f}   '
              f'best-CONSTANT (per-song marginal)={np.mean(totc):+.4f}   '
              f'gap={np.mean(tot)-np.mean(totc):+.4f} nats/frame')
        # (iv) how far can a phase sweep move p(beat) / p(obs) in PROBABILITY?
        G=torch.linspace(0,TWO_PI,721,device=DEV)
        mt=F.one_hot(torch.full((721,),3,device=DEV,dtype=torch.long),4).float()
        z=m.z_features(mt,G,torch.full((721,),-2.66,device=DEV))
        pb=torch.sigmoid(m.decoder(z)); po=torch.sigmoid(m.h_dec(z)) if ot=='bern' else m.h_dec(z)
        print(f'  phase sweep p(beat) range=[{float(pb[:,0].min()):.4f},{float(pb[:,0].max()):.4f}]  '
              f'p(db) range=[{float(pb[:,1].min()):.4f},{float(pb[:,1].max()):.4f}]')
        print(f'  phase sweep obs ch0 range=[{float(po[:,0].min()):.4f},{float(po[:,0].max()):.4f}] '
              f'ch1=[{float(po[:,1].min()):.4f},{float(po[:,1].max()):.4f}]   '
              f'true act mean={np.mean(mrate):.4f}')
main()
