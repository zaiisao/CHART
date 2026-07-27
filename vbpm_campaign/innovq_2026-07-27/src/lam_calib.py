"""Find lambda that keeps BOTH properties:
   (i) decoder phase sensitivity (BCE gives +40%, Cramer +1.3%)  -> want HIGH
   (ii) far-field tempo slope    (BCE ~0,        Cramer 89250)   -> want HIGH
lam=0.03 was calibrated on Cramer's CONVERGED magnitude and swamps BCE 26x at init."""
import sys, math, numpy as np, torch, torch.nn.functional as F
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P, innovq as IQ
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi
D=P.build_crops(P.load_songs("train"),n_per_song=2,seed=0,crop=T,dev=dev)
N=D["b"].shape[0]; K=4
mo=F.one_hot(D["m"].long().clamp(0,K-1),K).float()
if mo.dim()==2: mo=mo.unsqueeze(1).expand(-1,T,-1)
def Z(phi,lt): return torch.cat([torch.cos(phi).unsqueeze(-1),torch.sin(phi).unsqueeze(-1),lt.unsqueeze(-1),mo[:phi.shape[0]]],-1)
Zor=Z(D["phi"],D["lt"])
print(f"{'lambda':>9} {'sensitivity':>12} {'tempo far-slope':>16}")
for lam in (0.0, 0.0003, 0.001, 0.003, 0.01, 0.03):
    P.CRAMER_LAM=lam
    rec = "bce" if lam==0.0 else "hybrid"
    torch.manual_seed(0)
    d0,h0=P.new_decoders(dev); dec,hdec=IQ.Cut(d0),IQ.Cut(h0)
    opt=torch.optim.Adam(list(d0.parameters())+list(h0.parameters()),lr=3e-3)
    for s in range(600):
        opt.zero_grad(); sel=torch.randperm(N,device=dev)[:16]
        a,b_,c=P.recon_terms(dec,hdec,Zor[sel],D["b"][sel],D["db"][sel],D["obs"][sel],recon=rec)
        (a+b_+c).mean().backward(); opt.step()
    with torch.no_grad():
        g=float(sum(P.recon_terms(dec,hdec,Zor[:32],D["b"][:32],D["db"][:32],D["obs"][:32],recon=rec)[i].mean() for i in (0,1)))
        sc=(D["phi"][:32]+0.05*torch.arange(T,device=dev).float())%TWO_PI
        bd=float(sum(P.recon_terms(dec,hdec,Z(sc,D["lt"][:32]),D["b"][:32],D["db"][:32],D["obs"][:32],recon=rec)[i].mean() for i in (0,1)))
        sens=100*(bd-g)/abs(g)
        # far-field tempo slope: recon at 0.5x vs 0.7x tempo (both far from the answer)
        r=[]
        for m_ in (0.5,0.7):
            lt2=D["lt"][:32]+math.log(m_)
            inc=F.pad(torch.exp(lt2.clamp(-12,6))[:,:-1],(1,0))
            ph=(D["phi"][:32,:1]+torch.cumsum(inc,1))%TWO_PI
            r.append(float(sum(P.recon_terms(dec,hdec,Z(ph,lt2),D["b"][:32],D["db"][:32],D["obs"][:32],recon=rec)[i].mean() for i in (0,1))))
        slope=abs(r[1]-r[0])/0.2
    print(f"{lam:9.4f} {sens:11.2f}% {slope:16.1f}")
