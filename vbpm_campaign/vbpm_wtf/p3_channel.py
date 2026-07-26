"""Which z_feat channel carries the beat information? Replay the exact ELBO posterior
recursion with the TRAINED model, then ablate channels of Z and re-score rec_beat/rec_obs."""
import sys, math, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms')
import variant_b as VB
from variant_b import _stationary_dev_sigma
from vbpm.distributions import (TWO_PI, gumbel_softmax, sample_wrapped_cauchy, sample_student_t)
from audit_common import load_split, FPS
from common import targets
ARMS='/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms'
DEV='cuda:0'

@torch.no_grad()
def posterior_Z(model,h,b,temperature=0.3):
    B,T,_=h.shape
    post_ctx=model.encode_posterior(h,b); prior_ctx=model.encode_prior(h)
    dof=model.tempo_dof()
    z0=model.z0.unsqueeze(0).expand(B,-1)
    q=model.unpack(model.post_head(torch.cat([post_ctx[:,0],z0],-1)))
    q_m,q_ph_mu,q_ph_rho,q_lv_mu,q_lv_s,q_dv_mu,q_dv_s=q
    meter=gumbel_softmax(q_m,temperature); phi=sample_wrapped_cauchy(q_ph_mu,q_ph_rho)
    level=sample_student_t(dof,q_lv_mu,q_lv_s); dev=q_dv_mu+q_dv_s*torch.randn_like(q_dv_mu)
    lt=level+dev
    Zs=[model.z_features(meter,phi,lt)]; a_lv=model.level_ar(); anchor=level
    mp,pp,lp,dp,ltp=meter,phi,level,dev,lt
    crossings=[torch.zeros_like(phi)]
    for t in range(1,T):
        zpf=model.z_features(mp,pp,ltp)
        q_m,q_ph_mu,q_ph_rho,q_lv_mu,q_lv_s,q_dv_mu,q_dv_s=model.unpack(
            model.post_head(torch.cat([post_ctx[:,t],zpf],-1)))
        adv=pp+torch.exp(ltp.clamp(-12,6)); cross=(adv>=TWO_PI).to(h.dtype)
        phi=sample_wrapped_cauchy(q_ph_mu,q_ph_rho)
        level=sample_student_t(dof,q_lv_mu,q_lv_s); dev=q_dv_mu+q_dv_s*torch.randn_like(q_dv_mu)
        lt=level+dev
        meter=torch.where(cross.unsqueeze(-1)>0.5,gumbel_softmax(q_m,temperature),mp)
        Zs.append(model.z_features(meter,phi,lt)); crossings.append(cross)
        mp,pp,lp,dp,ltp=meter,phi,level,dev,lt
    return torch.stack(Zs,1), torch.stack(crossings,1)

def bce(logits,tgt): return F.binary_cross_entropy_with_logits(logits,tgt,reduction='none').sum(1).mean()

def run(tag,h_from_obs):
    ck=torch.load(f'{ARMS}/arm_i_{tag}.pt',map_location=DEV)
    hd=2 if h_from_obs else 768
    model=VB.BarPointerVAE_B(h_dim=hd,hidden=128,num_meters=4,obs_dim=2,obs_type='bern').to(DEV)
    model.load_state_dict(ck['model']); model.eval()
    merge=None
    if not h_from_obs:
        import arm_i as A
        merge=A.LayerMerge().to(DEV); merge.load_state_dict(ck['merge']); merge.eval()
    print('='*78); print('MODEL',tag,'h=',('activation' if h_from_obs else 'MERT'))
    for split in ['train','eval']:
        songs=load_split(split,with_feats=True,cap=24)
        d=np.load(f'{ARMS}/act_{split}.npz',allow_pickle=True)
        rng=np.random.default_rng(0); torch.manual_seed(0)
        FE,B_,D_,O_=[],[],[],[]
        for s in songs[:16]:
            T=s['feats'].shape[1]
            if T<=256: continue
            st=int(rng.integers(0,T-256))
            FE.append(torch.from_numpy(s['feats'][:,st:st+256,:].astype(np.float32)))
            b,dd=targets(s['beats'],s['downs'],st,256)
            B_.append(torch.from_numpy(b)); D_.append(torch.from_numpy(dd))
            a=np.clip(np.asarray(d[s['stem']+'|act'],np.float32)[st:st+256],1e-4,1-1e-4)
            O_.append(torch.from_numpy(a))
        f=torch.stack(FE).to(DEV); b=torch.stack(B_).to(DEV); db=torch.stack(D_).to(DEV); o=torch.stack(O_).to(DEV)
        h = o if h_from_obs else merge(f)
        Z,cross=posterior_Z(model,h,b)
        Bn,T,_=Z.shape
        lt=Z[...,2]; phi=torch.atan2(Z[...,1],Z[...,0])%TWO_PI
        def sc(Zx,name):
            lg=model.decoder(Zx); rb=bce(lg[...,0],b); rd=bce(lg[...,1],db)
            ol=model.obs_logp(Zx.reshape(-1,7),o.reshape(-1,2)).reshape(Bn,T).sum(1).mean()
            print(f'   {name:34s} rec_b={float(rb):7.2f} rec_db={float(rd):6.2f} rec_obs={float(-ol):7.2f}')
            return float(rb),float(rd),float(-ol)
        print(f'  --- {split} (16 crops x 256) ---')
        print(f'   posterior log_tempo: mean {float(lt.mean()):.3f} std-over-time(per crop, avg) {float(lt.std(1).mean()):.3f} '
              f'min {float(lt.min()):.2f} max {float(lt.max()):.2f} | exp() mean {float(lt.exp().mean()):.4f} '
              f'(true phidot~0.063 rad/fr) ; frames with exp(lt)>pi: {float((lt.exp()>math.pi).float().mean()):.3f}')
        dphi=((phi[:,1:]-phi[:,:-1]+math.pi)%TWO_PI)-math.pi
        print(f'   posterior phase: frac_neg_incr {float((dphi<0).float().mean()):.3f} mean_incr {float(dphi.mean()):.4f} '
              f'crossings/crop {float(cross.sum(1).mean()):.1f} (true bars/crop ~ {256/ (4*24.4):.1f})')
        # point-biserial correlations with the beat target
        for nm,ch in (('cos_phi',Z[...,0]),('sin_phi',Z[...,1]),('log_tempo',Z[...,2])):
            x=ch.reshape(-1).float(); y=b.reshape(-1).float()
            r=float(((x-x.mean())*(y-y.mean())).mean()/(x.std()*y.std()+1e-9))
            print(f'   corr({nm}, beat_target) = {r:+.4f}')
        base=sc(Z,'FULL posterior Z')
        Zm=Z.clone(); Zm[...,2]=Z[...,2].mean(1,keepdim=True)          # freeze log_tempo per crop
        sc(Zm,'log_tempo -> per-crop MEAN')
        Zp=Z.clone(); rp=torch.rand(Bn,T,device=DEV)*TWO_PI
        Zp[...,0]=rp.cos(); Zp[...,1]=rp.sin()                          # destroy phase
        sc(Zp,'phase -> RANDOM')
        Zpm=Zp.clone(); Zpm[...,2]=Z[...,2].mean(1,keepdim=True)
        sc(Zpm,'phase RANDOM + log_tempo MEAN')
        Zmt=Z.clone(); Zmt[...,3:]=Z[...,3:].mean(1,keepdim=True)
        sc(Zmt,'meter -> per-crop MEAN')
        Zc=Z.mean(1,keepdim=True).expand_as(Z).contiguous()
        sc(Zc,'ALL z -> per-crop MEAN (constant)')
        # oracle bar phase substituted in
        del f,h,Z
        torch.cuda.empty_cache()

run('ii_bern',True)
run('i_bern',False)
