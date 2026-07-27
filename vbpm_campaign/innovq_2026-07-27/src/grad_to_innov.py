"""Does ANY learning signal reach innov_head? Measure it directly.
 (1) grad norm on innov_head vs init_head vs encoder, under each recon loss
 (2) d(recon)/d(mu_eps) directly -- does recon WANT nonzero innovations?
 (3) does the recon actually improve if we hand it ORACLE innovations?
 (4) what does the decoder emit (explains F=0.000)?"""
import sys, math, numpy as np, torch, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P, innovq as IQ
dev="cuda:0"; torch.manual_seed(0); T=1500
P.PHYS["gamma_phase"]=0.06; P.PHYS["dev_sigma"]=0.02
D=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=T,dev=dev)
idx=torch.arange(8,device=dev)
def grp(m):
    return dict(innov=list(m.innov_head.parameters()), init=list(m.init_head.parameters()),
                enc=[p for n,p in m.named_parameters() if "innov_head" not in n and "init_head" not in n])
print("(1) gradient norms by module, beta=1")
for r in ("bce","cramer"):
    m=IQ.InnovQ().to(dev); m.train(); d0,h0=P.new_decoders(dev); dec,hdec=IQ.Cut(d0),IQ.Cut(h0)
    torch.manual_seed(0)
    loss,info,_=IQ.elbo_innovq(m,D,dec,hdec,idx=idx,beta=1.0,recon=r,sample=True)
    loss.backward(); g=grp(m)
    nrm=lambda ps: float(torch.cat([p.grad.flatten() for p in ps if p.grad is not None]).norm())
    print(f"   {r:7s} innov_head {nrm(g['innov']):11.4e} | init_head {nrm(g['init']):11.4e} | encoder {nrm(g['enc']):11.4e}")
print()
print("(2)+(3) does recon WANT innovations? inject a constant phase-innovation bias and watch recon")
m=IQ.InnovQ().to(dev); m.eval()
d0,h0=P.new_decoders(dev); dec,hdec=IQ.Cut(d0),IQ.Cut(h0)
for r in ("bce","cramer"):
    row=[]
    for bias in (0.0, 0.005, 0.02, 0.05):
        tot=0.0
        for rep in range(3):
            torch.manual_seed(rep)
            ro=IQ.rollout(m,D["h"][idx],D["b"][idx],sample=True)
            # perturb phase by a constant drift, rebuild Z, score recon only
            phi2=(ro["phi"]+bias*torch.arange(T,device=dev).float())%(2*math.pi)
            Z2=torch.cat([torch.cos(phi2).unsqueeze(-1),torch.sin(phi2).unsqueeze(-1),
                          ro["Z"][...,2:3],ro["Z"][...,3:]],-1)
            rb,rd,_=P.recon_terms(dec,hdec,Z2,D["b"][idx],D["db"][idx],D["obs"][idx],recon=r)
            tot+=float((rb+rd).mean())
        row.append(tot/3)
    print(f"   {r:7s} recon vs injected drift: "+"  ".join(f"{b}:{v:9.1f}" for b,v in zip((0.0,0.005,0.02,0.05),row)))
print()
print("(4) decoder output stats (why F=0.000?)")
for tag in ("t_cramer_tg","t_cramer_plain","t_bce_plain"):
    try:
        ck=torch.load(f"{P.OUT}/innovq_{tag}.pt",map_location=dev,weights_only=False)
    except Exception as e:
        print(f"   {tag}: no checkpoint ({type(e).__name__})"); continue
    dd,hh=P.new_decoders(dev); dd.load_state_dict(ck["dec"]); hh.load_state_dict(ck["hdec"])
    mm=IQ.InnovQ().to(dev)
    try: mm.load_state_dict(ck["model"],strict=False)
    except Exception: pass
    mm.eval()
    with torch.no_grad():
        ro=IQ.rollout(mm,D["h"][idx],D["b"][idx],sample=False)
        lg=IQ.Cut(dd)(ro["Z"]); pb=torch.sigmoid(lg[...,0])
    print(f"   {tag:16s} p_beat: mean {float(pb.mean()):.4f} max {float(pb.max()):.4f} "
          f"std {float(pb.std()):.4f} | frames>0.5: {int((pb>0.5).sum())}")
