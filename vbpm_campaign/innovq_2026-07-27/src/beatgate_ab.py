"""A/B: per-frame tempo random walk (current) vs BEAT-GATED tempo (the doc's own
between-beats-constant condition, which our model deliberately departed from).

  "Note that this formulation restricts tempo changes to occur ONLY at beat boundaries;
   between beats, tempo is held constant."   -- ELBO_for_DBN.md, describing Krebs 2015
  "OUR MODEL adopts a continuous Log-Normal random walk (a deliberate departure ...)"

Gated: lev_t = lev_{t-1} + gate_t * eps_lt_t, gate = 1 only when the BEAT index advances.
Beats-per-bar comes from the meter latent (class k -> k+1 beats), never hardcoded.
Both arms: free q (no encoder), initialized AT the true tempo, identical objective,
identical frozen phase-sensitive decoder, fresh noise per step.
"""
import sys, math, torch, torch.nn.functional as F, numpy as np
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P, innovq as IQ
from wire_test import free_traj, free_heads
from rollout_vec_s import draw_noise
from faithful.distributions import kl_categorical
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi; K=4; FPS=50.0
GP=0.06; P.PHYS["gamma_phase"]=GP; IQ.RHO_P=math.exp(-GP); IQ.GP1=1-IQ.RHO_P; IQ.R0=IQ.softplus_inv(IQ.GP1)

def free_traj_gated(m_logits,mu_phi1,rho1,mu_l1,s_l1,mu_eps,sq,mu_lt,s_lv,noise,Pi,*,
                    sample=True,temperature=0.3,passes=2):
    """free_traj with the level update BEAT-GATED. Everything else identical."""
    B,Kk=m_logits.shape; Tn=mu_eps.shape[1]+1; dv=m_logits.device
    dof=torch.tensor(IQ.DOF,device=dv)
    if sample:
        phi1=(mu_phi1+(-torch.log(rho1))*torch.tan(math.pi*(noise["u"][:,0]-0.5)))%TWO_PI
        lev1=mu_l1+s_l1*noise["tstd"][:,0]
        eps=mu_eps+(-torch.log1p(-sq))*torch.tan(math.pi*(noise["u"][:,1:]-0.5))
        eps_lt=mu_lt+s_lv*noise["tstd"][:,1:]
        dvv=IQ.DEV_SIGMA*noise["nrm"]
        m_draw=F.softmax((m_logits.unsqueeze(1)+noise["gum"])/temperature,-1)
    else:
        phi1,lev1,eps,eps_lt=mu_phi1,mu_l1,mu_eps,mu_lt
        dvv=torch.zeros(B,Tn,device=dv); m_draw=F.softmax(m_logits/max(temperature,1e-6),-1).unsqueeze(1).expand(-1,Tn,-1)
    bpb=(F.softmax(m_logits,-1)*torch.arange(1,Kk+1,device=dv).float()).sum(-1)   # meter latent -> beats/bar
    gate=torch.ones_like(eps_lt)
    for _ in range(passes):
        lev=(lev1.unsqueeze(1).double()+torch.cumsum(F.pad(gate*eps_lt,(1,0)).double(),1)).float()
        lt=lev+dvv
        steps=torch.exp(lt.clamp(-12.,6.))
        inc=F.pad(steps[:,:-1],(1,0))+F.pad(eps,(1,0))
        unwrapped=(phi1.unsqueeze(1).double()+torch.cumsum(inc.double(),1))
        be=(unwrapped*bpb.unsqueeze(1).double()/TWO_PI)                    # beats elapsed
        gate=(torch.floor(be[:,1:])>torch.floor(be[:,:-1])).float().detach()
    phi=(unwrapped%TWO_PI).float()
    adv=phi[:,:-1]+steps[:,:-1]
    cross=F.pad((adv>=TWO_PI).float(),(1,0)); cf=cross.clone(); cf[:,0]=1.0
    ar=torch.arange(Tn,device=dv)
    last=torch.cummax((ar.unsqueeze(0)*cf).long(),1).values
    meter=torch.gather(m_draw,1,last.unsqueeze(-1).expand(-1,-1,Kk))
    mprev=torch.cat([meter[:,:1],meter[:,:-1]],1)
    Z=torch.cat([torch.cos(phi).unsqueeze(-1),torch.sin(phi).unsqueeze(-1),
                 lt.clamp(-12.,6.).unsqueeze(-1),meter],-1)
    kl_p=(IQ.kl_wrapped_cauchy(mu_phi1,rho1,torch.full_like(mu_phi1,math.pi),torch.full_like(mu_phi1,1e-6)).double()
          +IQ.kl_phase_innov(mu_eps.reshape(-1),sq.reshape(-1)).reshape(B,-1).sum(1))
    # only GATED level innovations are used, so only they are charged KL
    kl_l=(P.kl_t_mc(dof,mu_l1,s_l1,dof,torch.full((B,),IQ.INIT_LV_MU,device=dv),
                    torch.full((B,),IQ.INIT_LV_S,device=dv),lev[:,0])
          +(gate*P.kl_t_mc(dof,mu_lt,s_lv,dof,torch.zeros_like(mu_lt),
                           torch.full_like(mu_lt,IQ.T_SCALE),eps_lt)).sum(1))
    lqr=torch.log_softmax(m_logits,-1); lq=lqr.unsqueeze(1).expand(-1,Tn-1,-1)
    lp=torch.log(mprev[:,1:]@Pi+1e-9)
    kl_m=(kl_categorical(lqr,torch.full((B,Kk),-math.log(Kk),device=dv))
          +(cross[:,1:]*kl_categorical(lq,lp)).sum(1))
    return dict(Z=Z,phi=phi,lt=lt,kl_p=kl_p.float(),kl_l=kl_l,kl_m=kl_m,
                kl_dv=torch.zeros(B,device=dv),n_cross=1.0+cross[:,1:].sum(1),
                gate_rate=float(gate.mean()))

if __name__ == "__main__":
    tr=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev); N=tr["b"].shape[0]
    sel=torch.arange(48,device=dev); B=len(sel)
    h,b,db,obs=tr["h"][sel],tr["b"][sel],tr["db"][sel],tr["obs"][sel]
    phi_true,lt_frame=tr["phi"][sel],tr["lt"][sel]; lt_true=lt_frame.mean(1)
    mo=F.one_hot(tr["m"].long().clamp(0,K-1),K).float()
    if mo.dim()==2: mo=mo.unsqueeze(1).expand(-1,T,-1)
    Zor=torch.cat([torch.cos(tr["phi"]).unsqueeze(-1),torch.sin(tr["phi"]).unsqueeze(-1),
                   tr["lt"].unsqueeze(-1),mo],-1)
    d0,h0=P.new_decoders(dev); dec,hdec=IQ.Cut(d0),IQ.Cut(h0)
    od=torch.optim.Adam(list(d0.parameters())+list(h0.parameters()),lr=3e-3)
    for s in range(700):
        od.zero_grad(); ss=torch.randperm(N,device=dev)[:16]
        a,b_,c_=P.recon_terms(dec,hdec,Zor[ss],tr["b"][ss],tr["db"][ss],tr["obs"][ss],recon="bce")
        (a+b_+c_).mean().backward(); od.step()
    with torch.no_grad():
        g=float(sum(P.recon_terms(dec,hdec,Zor[:32],tr["b"][:32],tr["db"][:32],tr["obs"][:32],recon="bce")[i].mean() for i in (0,1)))
        scr=(tr["phi"][:32]+0.05*torch.arange(T,device=dev).float())%TWO_PI
        Zb=torch.cat([torch.cos(scr).unsqueeze(-1),torch.sin(scr).unsqueeze(-1),tr["lt"][:32].unsqueeze(-1),mo[:32]],-1)
        bd=float(sum(P.recon_terms(dec,hdec,Zb,tr["b"][:32],tr["db"][:32],tr["obs"][:32],recon="bce")[i].mean() for i in (0,1)))
    print(f"[gate] decoder phase sensitivity {100*(bd-g)/abs(g):+.1f}%")
    for q in list(d0.parameters())+list(h0.parameters()): q.requires_grad_(False)
    Pi=IQ.InnovQ().to(dev).Pi_phys
    def run(gated,steps=400):
        torch.manual_seed(0)
        pars=dict(mp1=torch.zeros(B,device=dev,requires_grad=True),
                  r1r=torch.zeros(B,device=dev,requires_grad=True),
                  ml1=lt_true.clone().detach().requires_grad_(True),
                  sl1r=torch.full((B,),IQ.softplus_inv(0.25),device=dev,requires_grad=True),
                  mlog=torch.zeros(B,K,device=dev,requires_grad=True),
                  inn=torch.zeros(B,T-1,4,device=dev,requires_grad=True))
        fn=free_traj_gated if gated else free_traj
        def roll(nz,sample): return fn(**free_heads(pars,0.05,0.0025,0.9,0.05),noise=nz,Pi=Pi,sample=sample)
        def stat():
            with torch.no_grad():
                ro=roll(draw_noise(B,T,K,dev,IQ.DOF),False)
                corr=float(torch.abs(torch.exp(1j*(ro["phi"]-phi_true)).mean(1)).mean())
                mae=100*float((ro["lt"].mean(1)-lt_true).abs().mean())
                wob=100*float(((ro["lt"].max(1).values-ro["lt"].min(1).values).exp()-1).mean())
                rb,rd,roo=P.recon_terms(dec,hdec,ro["Z"],b,db,obs,recon="bce")
                L=float((rb+rd+roo+ro["kl_p"]+ro["kl_l"]+ro["kl_m"]).mean())
            return corr,mae,wob,L,ro.get("gate_rate",1.0)
        s0=stat()
        opt=torch.optim.Adam(list(pars.values()),lr=3e-3)
        for s in range(steps):
            opt.zero_grad()
            ro=roll(draw_noise(B,T,K,dev,IQ.DOF),True)
            rb,rd,roo=P.recon_terms(dec,hdec,ro["Z"],b,db,obs,recon="bce")
            ((rb+rd+roo+ro["kl_p"]+ro["kl_l"]+ro["kl_m"]).mean()).backward(); opt.step()
        return s0,stat()
    def run_ret(gated,steps=400):
        torch.manual_seed(0)
        pars=dict(mp1=torch.zeros(B,device=dev,requires_grad=True),
                  r1r=torch.zeros(B,device=dev,requires_grad=True),
                  ml1=lt_true.clone().detach().requires_grad_(True),
                  sl1r=torch.full((B,),IQ.softplus_inv(0.25),device=dev,requires_grad=True),
                  mlog=torch.zeros(B,K,device=dev,requires_grad=True),
                  inn=torch.zeros(B,T-1,4,device=dev,requires_grad=True))
        fn=free_traj_gated if gated else free_traj
        opt=torch.optim.Adam(list(pars.values()),lr=3e-3)
        for s in range(steps):
            opt.zero_grad()
            ro=fn(**free_heads(pars,0.05,0.0025,0.9,0.05),noise=draw_noise(B,T,K,dev,IQ.DOF),Pi=Pi,sample=True)
            rb,rd,roo=P.recon_terms(dec,hdec,ro["Z"],b,db,obs,recon="bce")
            ((rb+rd+roo+ro["kl_p"]+ro["kl_l"]+ro["kl_m"]).mean()).backward(); opt.step()
        return pars,fn
    def corr_of(pars,fn,zero_innov=False,true_level=False,true_offset=False):
        with torch.no_grad():
            hd=free_heads(pars,0.05,0.0025,0.9,0.05)
            if zero_innov: hd["mu_eps"]=torch.zeros_like(hd["mu_eps"])
            if true_level: hd["mu_l1"]=lt_true.clone(); hd["mu_lt"]=torch.zeros_like(hd["mu_lt"])
            if true_offset: hd["mu_phi1"]=phi_true[:,0].clone()
            ro=fn(**hd,noise=draw_noise(B,T,K,dev,IQ.DOF),Pi=Pi,sample=False)
            return float(torch.abs(torch.exp(1j*(ro["phi"]-phi_true)).mean(1)).mean())
    print("\n=== WHAT causes the residual fall? (BEAT-GATED arm, substitutions) ===")
    pg,fg=run_ret(True)
    print(f"  D  as trained                              corr {corr_of(pg,fg):.4f}")
    print(f"  C  + zero phase innovations                corr {corr_of(pg,fg,zero_innov=True):.4f}")
    print(f"  B  + TRUE tempo (keep its innovations)     corr {corr_of(pg,fg,true_level=True):.4f}")
    print(f"  A  + TRUE tempo AND zero innovations       corr {corr_of(pg,fg,true_level=True,zero_innov=True):.4f}")
    print(f"  A' + also true phase offset                corr {corr_of(pg,fg,true_level=True,zero_innov=True,true_offset=True):.4f}")
    for nm,gt in (("per-frame (current)",False),("BEAT-GATED (doc)",True)):
        (c0,m0,w0,L0,gr0),(c1,m1,w1,L1,gr1)=run(gt)
        print(f"\n  {nm}")
        print(f"    gate rate (frames where tempo may change): {100*gr1:.2f}%")
        print(f"    corr    {c0:.4f} -> {c1:.4f}")
        print(f"    lvlMAE  {m0:6.2f}% -> {m1:6.2f}%")
        print(f"    tempo wobble within crop {w0:.1f}% -> {w1:.1f}%")
        print(f"    ELBO    {L0:9.1f} -> {L1:9.1f}  ({L1-L0:+.1f})")
