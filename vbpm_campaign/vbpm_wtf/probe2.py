"""PROBE 1 continued: WHERE does rec_beat=23 come from, if phase is worthless?
Ablate each z_feat channel on real posterior rollouts."""
import sys, math, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms')
import variant_b as VB
from vbpm.distributions import TWO_PI, gumbel_softmax, sample_wrapped_cauchy, sample_student_t
from audit_common import load_split, FPS
from common import targets
DEV='cuda:0'; ARMS='/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms'
HDIM={'i_bern':768,'i_gauss':768,'ii_bern':2}

class LayerMerge(nn.Module):
    def __init__(s,n=13):
        super().__init__(); s.layer_logits=nn.Parameter(torch.zeros(n))
    def forward(s,f): return torch.einsum('l,bltf->btf',torch.softmax(s.layer_logits,0),f)

def load(tag):
    ck=torch.load(f'{ARMS}/arm_i_{tag}.pt',map_location=DEV)
    ot='bern' if ck['obs']=='head_bern' else 'gauss'
    m=VB.BarPointerVAE_B(h_dim=HDIM[tag],hidden=ck['config']['hidden'],num_meters=4,
                         obs_dim=2,obs_type=ot).to(DEV)
    m.load_state_dict(ck['model']); m.eval()
    mg=LayerMerge().to(DEV); mg.load_state_dict(ck['merge']); mg.eval()
    return mg,m,ck

@torch.no_grad()
def rollout(model,h,b,temperature=0.3):
    """posterior rollout exactly as elbo_b, returning per-frame meter/phi/log_tempo."""
    B,T,_=h.shape
    post=model.encode_posterior(h,b); pri=model.encode_prior(h); dof=model.tempo_dof()
    z0=model.z0.unsqueeze(0).expand(B,-1)
    q=model.unpack(model.post_head(torch.cat([post[:,0],z0],-1)))
    meter=gumbel_softmax(q[0],temperature); phi=sample_wrapped_cauchy(q[1],q[2])
    level=sample_student_t(dof,q[3],q[4]); dev=q[5]+q[6]*torch.randn_like(q[5])
    lt=level+dev; a_lv=model.level_ar(); anchor=level
    M,P,L=[meter],[phi],[lt]
    mp,pp,lp,dp,ltp=meter,phi,level,dev,lt
    for t in range(1,T):
        zf=model.z_features(mp,pp,ltp)
        q=model.unpack(model.post_head(torch.cat([post[:,t],zf],-1)))
        adv=pp+torch.exp(ltp.clamp(-12,6)); cross=(adv>=TWO_PI).to(h.dtype)
        phi=sample_wrapped_cauchy(q[1],q[2]); level=sample_student_t(dof,q[3],q[4])
        dev=q[5]+q[6]*torch.randn_like(q[5]); lt=level+dev
        draw=gumbel_softmax(q[0],temperature)
        meter=torch.where(cross.unsqueeze(-1)>0.5,draw,mp)
        M.append(meter);P.append(phi);L.append(lt)
        mp,pp,lp,dp,ltp=meter,phi,level,dev,lt
    return torch.stack(M,1),torch.stack(P,1),torch.stack(L,1)

def bce(logit,tgt): return F.binary_cross_entropy_with_logits(logit,tgt,reduction='none').sum(1)

@torch.no_grad()
def main():
    tr=load_split('train',with_feats=True,cap=40)
    rng=np.random.default_rng(0)
    for tag in ['i_bern','ii_bern']:
        mg,model,ck=load(tag)
        print(f'\n{"="*70}\n{tag}\n{"="*70}')
        # sample a fixed batch of crops
        torch.manual_seed(0); rng=np.random.default_rng(0)
        fe,bb,dd=[],[],[]
        while len(fe)<16:
            s=tr[rng.integers(len(tr))]; T=s['feats'].shape[1]
            if T<=256: continue
            st=int(rng.integers(0,T-256))
            fe.append(torch.from_numpy(s['feats'][:,st:st+256,:].astype(np.float32)))
            b,d=targets(s['beats'],s['downs'],st,256)
            bb.append(torch.from_numpy(b)); dd.append(torch.from_numpy(d))
        f=torch.stack(fe).to(DEV); b=torch.stack(bb).to(DEV); d=torch.stack(dd).to(DEV)
        h=mg(f)
        if HDIM[tag]==2:
            # arm ii feeds the ACTIVATION as h -- reconstruct it from the act cache is complex;
            # instead use the *saved* pipeline: arm_ii builds h from the act cache.
            print('  [arm ii: h = 2-ch activation, rebuilt below]')
            import numpy as _np
            A=_np.load(f'{ARMS}/act_train.npz',allow_pickle=True)
            # recompute the same crops
            hs=[]; rng2=np.random.default_rng(0); k=0
            rng2=np.random.default_rng(0)
            tmp=[]
            while len(tmp)<16:
                s=tr[rng2.integers(len(tr))]; T=s['feats'].shape[1]
                if T<=256: continue
                st=int(rng2.integers(0,T-256))
                a=_np.clip(_np.asarray(A[s['stem']+'|act'],_np.float32),1e-4,1-1e-4)
                tmp.append(torch.from_numpy(a[st:st+256]))
            h=torch.stack(tmp).to(DEV)
        M,P,L=rollout(model,h,b)
        base_b=float(bce(torch.full_like(b,-100.)*0+torch.logit(b.mean().clamp(1e-6,1-1e-6)),b).mean())
        base_d=float(bce(torch.full_like(d,0.)+torch.logit(d.mean().clamp(1e-6,1-1e-6)),d).mean())
        print(f'  base-rate BCE/crop: beat={base_b:.2f}  db={base_d:.2f}   '
              f'(beat rate={float(b.mean()):.4f})')
        print(f'  posterior log_tempo: mean={float(L.mean()):+.3f} std={float(L.std()):.3f} '
              f'min={float(L.min()):+.2f} max={float(L.max()):+.2f}  '
              f'|dlt| mean={float((L[:,1:]-L[:,:-1]).abs().mean()):.4f}')
        print(f'  physical log_tempo would be ~ -2.66 (=> phidot {math.exp(-2.66):.4f} rad/frame)')
        dphi=((P[:,1:]-P[:,:-1]+math.pi)%TWO_PI)-math.pi
        print(f'  posterior dphi/frame: mean={float(dphi.mean()):+.4f} std={float(dphi.std()):.4f} '
              f'frac_neg={float((dphi<0).float().mean()):.3f}')
        variants={}
        Lmed=L.median(dim=1,keepdim=True).values.expand_as(L)
        Pconst=torch.zeros_like(P)
        Pshuf=P[:,torch.randperm(P.shape[1],device=DEV)]
        Mconst=F.one_hot(torch.full(M.shape[:2],3,device=DEV,dtype=torch.long),4).float()
        variants['FULL             ']=(M,P,L)
        variants['phase->0         ']=(M,Pconst,L)
        variants['phase shuffled   ']=(M,Pshuf,L)
        variants['logT->per-crop med']=(M,P,Lmed)
        variants['logT->-2.66      ']=(M,P,torch.full_like(L,-2.66))
        variants['meter->4         ']=(Mconst,P,L)
        variants['logT med + phase0']=(M,Pconst,Lmed)
        for k,(m_,p_,l_) in variants.items():
            z=model.z_features(m_.reshape(-1,4),p_.reshape(-1),l_.reshape(-1)).reshape(*P.shape,-1)
            lg=model.decoder(z)
            rb=float(bce(lg[...,0],b).mean()); rd=float(bce(lg[...,1],d).mean())
            ol=model.obs_logp(z.reshape(-1,7),h.reshape(-1,h.shape[-1])[:,:2]*0+0.0) if False else None
            print(f'   {k}  rec_b={rb:7.2f}  rec_db={rd:7.2f}')
main()
