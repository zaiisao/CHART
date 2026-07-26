import sys, json, math, numpy as np, torch
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms')
import variant_b as VB
ARMS='/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms'
NAMES=['cos_phi','sin_phi','log_tempo','m1','m2','m3','m4']
for tag,hdim in [('ii_bern',2),('i_bern',768)]:
    ck=torch.load(f'{ARMS}/arm_i_{tag}.pt',map_location='cpu')
    sd=ck['model']
    print('='*70); print(tag,'obs=',ck['obs'],'cfg keys',{k:ck['config'][k] for k in ('steps','warmup','obs_w','lr','frames','bs')})
    for name,key in [('h_dec (obs emission)','h_dec.0.weight'),('decoder (beat/db)','decoder.0.weight')]:
        W=sd[key]                      # [hidden, z_feat_dim]
        cn=W.abs().mean(0)
        l2=W.norm(dim=0)
        print(f'  {name}: shape {tuple(W.shape)}')
        print('     mean|w| per input dim: '+'  '.join(f'{n}={v:.4f}' for n,v in zip(NAMES,cn.tolist())))
        print('     L2   per input dim: '+'  '.join(f'{n}={v:.4f}' for n,v in zip(NAMES,l2.tolist())))
    # effective sensitivity: how much does output logit change per unit input change
    for name,k0,k2 in [('h_dec','h_dec.0.weight','h_dec.2.weight'),('decoder','decoder.0.weight','decoder.2.weight')]:
        W0=sd[k0]; b0=sd[k0.replace('weight','bias')]; W2=sd[k2]; b2=sd[k2.replace('weight','bias')]
        # sample z_feat on a realistic manifold: phi uniform, log_tempo ~ N(-2.77,0.2), meter=4
        g=torch.Generator().manual_seed(0)
        N=20000
        phi=torch.rand(N,generator=g)*2*math.pi
        lt=-2.77+0.2*torch.randn(N,generator=g)
        m=torch.zeros(N,4); m[:,3]=1.
        def fwd(phi,lt,m):
            z=torch.cat([torch.cos(phi)[:,None],torch.sin(phi)[:,None],lt[:,None],m],1)
            return (torch.tanh(z@W0.T+b0))@W2.T+b2
        base=fwd(phi,lt,m)
        # vary phi only
        p2=(phi+math.pi)%(2*math.pi)
        d_phi=(fwd(p2,lt,m)-base).abs().mean(0)
        # vary log_tempo only by 1 sd of what the posterior actually uses (probe with +-0.5)
        d_lt=(fwd(phi,lt+0.5,m)-base).abs().mean(0)
        d_m=(fwd(phi,lt,torch.eye(4)[torch.tensor([2]*N)])-base).abs().mean(0)
        print(f'  {name} output logit |delta| : phi->phi+pi {d_phi.tolist()} | logtempo+0.5 {d_lt.tolist()} | meter4->3 {d_m.tolist()}')
        print(f'  {name} output logit std over phi at fixed lt: {base.std(0).tolist()}  mean {base.mean(0).tolist()}')
