"""What does a USEFUL phase innovation cost in KL, and what does it BUY in recon?
If price >> benefit, zero innovations is the CORRECT optimum and the prior is misspecified."""
import sys, math, numpy as np, torch, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P, innovq as IQ
dev="cuda:0"; T=1500; TWO_PI=2*math.pi
print(f"RHO_P = {IQ.RHO_P:.8f}  ->  gamma_p = {-math.log(IQ.RHO_P):.3e} rad/frame")
print(f"s_phi (posterior bound) = 0.05 rad  ->  {0.05/-math.log(IQ.RHO_P):.0f}x the prior scale")
print(f"physical microtiming ~15ms @ ibi 24 frames (50fps) = {2*math.pi/4/24*0.75:.4f} rad\n")
sq=torch.full((1,),float(1-IQ.RHO_P),device=dev)
print("KL price per frame, and over a 1500-frame crop:")
for mu in (0.0,1e-4,5e-4,0.001,0.005,0.01,0.05):
    k=float(IQ.kl_phase_innov(torch.full((1,),mu,device=dev),sq))
    print(f"   mu_eps={mu:7.4f} rad  KL/frame {k:10.4f} nats   x1500 = {k*1500:12.1f} nats")
# recon benefit: frozen oracle-fit decoder, how many nats does correct phase buy?
D=P.build_crops(P.load_songs("train"),n_per_song=2,seed=0,crop=T,dev=dev)
N=D["b"].shape[0]; K=4
mo=F.one_hot(D["m"].long().clamp(0,K-1),K).float()
if mo.dim()==2: mo=mo.unsqueeze(1).expand(-1,T,-1)
Z=lambda phi: torch.cat([torch.cos(phi).unsqueeze(-1),torch.sin(phi).unsqueeze(-1),D["lt"].unsqueeze(-1),mo],-1)
d0,h0=P.new_decoders(dev); dec,hdec=IQ.Cut(d0),IQ.Cut(h0)
opt=torch.optim.Adam(list(d0.parameters())+list(h0.parameters()),lr=3e-3)
Zor=Z(D["phi"])
for s in range(600):
    opt.zero_grad(); sel=torch.randperm(N,device=dev)[:16]
    a,b_,c=P.recon_terms(dec,hdec,Zor[sel],D["b"][sel],D["db"][sel],D["obs"][sel],recon="bce")
    (a+b_+c).mean().backward(); opt.step()
with torch.no_grad():
    good=float(sum(P.recon_terms(dec,hdec,Zor[:32],D["b"][:32],D["db"][:32],D["obs"][:32],recon="bce")[i].mean() for i in (0,1)))
    bad=float(sum(P.recon_terms(dec,hdec,Z((D["phi"][:32]+1.0)%TWO_PI),D["b"][:32],D["db"][:32],D["obs"][:32],recon="bce")[i].mean() for i in (0,1)))
print(f"\nrecon BENEFIT of correct vs 1-rad-off phase (frozen oracle-fit read-out): {bad-good:.1f} nats/crop")
print(f"KL PRICE of mu=0.005 rad sustained over the crop: {float(IQ.kl_phase_innov(torch.full((1,),0.005,device=dev),sq))*1500:.1f} nats/crop")
