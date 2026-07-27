"""Vectorized (parallel-in-time) rollout via Picard iteration + exactness test vs the loop.
The only sequential coupling is innov_head(ctx_t, z_{t-1}); innovations are bounded (|mu|<=s_phi
=0.05) so the map is near-contractive. Iterate: trajectory -> all heads in parallel -> cumsum.
Meter carry is handled by cummax-over-crossings (crossings are determined by the trajectory)."""
import sys, math, time
import torch, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
from innovq import TWO_PI, R0, B_SLT0, DEV_SIGMA, DOF, T_SCALE

def rollout_vec(model, h, b, n_picard=3, temperature=0.3):
    """DETERMINISTIC (sample=False) path only -- what placement/probe use."""
    Bn,T,_=h.shape; dev=h.device; K=model.K
    ctx=model.encode_posterior(h,b)
    init=model.init_head(torch.cat([ctx.mean(1),ctx[:,0]],-1))
    mu_phi1=torch.atan2(init[:,K+1],init[:,K])%TWO_PI
    mu_l1=init[:,K+3]+model.level_offset
    meter0=F.softmax(init[:,:K]/max(temperature,1e-6),-1)
    # initial guess: zero innovations (pure prior recursion) -- itself a cumsum
    phi=torch.empty(Bn,T,device=dev); lt=torch.empty(Bn,T,device=dev)
    lev=mu_l1.unsqueeze(1).expand(-1,T).clone(); lt=lev.clone()
    steps=torch.exp(lt.clamp(-12.,6.))
    phi=((mu_phi1.unsqueeze(1).double()+torch.cumsum(F.pad(steps[:,:-1],(1,0)).double(),1))%TWO_PI).float()
    meter=meter0.unsqueeze(1).expand(-1,T,-1)
    for _ in range(n_picard):
        zf=model.z_features(meter.reshape(-1,K),phi.reshape(-1),lt.reshape(-1)).reshape(Bn,T,-1)
        out=model.innov_head(torch.cat([ctx,zf],-1))             # ALL t at once
        mu_eps=torch.tanh(out[...,0])*model.s_phi
        mu_lt=torch.tanh(out[...,2])*model.s_lt
        # rebuild: level = mu_l1 + cumsum(mu_lt shifted); phi = mu_phi1 + cumsum(step + eps)
        lev=(mu_l1.unsqueeze(1).double()+torch.cumsum(F.pad(mu_lt[:,1:],(1,0)).double(),1)).float()
        lt=lev
        steps=torch.exp(lt.clamp(-12.,6.))
        inc=F.pad(steps[:,:-1],(1,0))+F.pad(mu_eps[:,1:],(1,0))
        phi=((mu_phi1.unsqueeze(1).double()+torch.cumsum(inc.double(),1))%TWO_PI).float()
        # meter: resample at crossings (advance>=2pi), carry otherwise -> cummax of crossing idx
        adv=phi[:,:-1]+steps[:,:-1]
        cross=F.pad((adv>=TWO_PI).float(),(1,0))
        meter=meter0.unsqueeze(1).expand(-1,T,-1)   # deterministic: m_draw == meter0 every draw
    return dict(phi=phi,lt=lt,mu_eps=mu_eps)

if __name__=="__main__":
    dev="cuda:0"; torch.manual_seed(0)
    D=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=1500,dev=dev)
    m=IQ.InnovQ().to(dev)
    import glob
    ck=glob.glob("innovq_pf_sm101_s0.pt")
    if ck: m.load_state_dict(torch.load(ck[0],map_location=dev,weights_only=False).get("model"),strict=False)
    m.eval()
    h,b=D["h"][:3],D["b"][:3]
    t0=time.time(); ref=IQ.rollout(m,h,b,sample=False); t_loop=time.time()-t0
    for npic in (1,2,3,5):
        t0=time.time(); v=rollout_vec(m,h,b,n_picard=npic); t_vec=time.time()-t0
        dphi=float(torch.abs(torch.angle(torch.exp(1j*(v["phi"]-ref["phi"])))).max())
        dlt=float((v["lt"]-ref["lt"]).abs().max())
        print(f"picard={npic}: max|dphi|={dphi:.3e} rad  max|dlt|={dlt:.3e} | loop {t_loop*1000:.0f}ms vec {t_vec*1000:.0f}ms  speedup {t_loop/t_vec:.1f}x")
