"""Concrete example: ORACLE z vs the z the ELBO walks to, for real crops.
Shows tempo (BPM) and where each puts its BAR BOUNDARIES vs the true downbeats."""
import sys, math, torch, torch.nn.functional as F, numpy as np
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P, innovq as IQ
from wire_test import free_rollout
from rollout_vec_s import draw_noise
dev="cuda:0"; torch.manual_seed(0); T=1500; TWO_PI=2*math.pi; K=4; FPS=50.0
GP=0.06; P.PHYS["gamma_phase"]=GP; IQ.RHO_P=math.exp(-GP); IQ.GP1=1-IQ.RHO_P; IQ.R0=IQ.softplus_inv(IQ.GP1)
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
for q in list(d0.parameters())+list(h0.parameters()): q.requires_grad_(False)
Pi=IQ.InnovQ().to(dev).Pi_phys
pars=dict(mp1=torch.zeros(B,device=dev,requires_grad=True),
          r1r=torch.zeros(B,device=dev,requires_grad=True),
          ml1=lt_true.clone().detach().requires_grad_(True),
          sl1r=torch.full((B,),IQ.softplus_inv(0.25),device=dev,requires_grad=True),
          mlog=torch.zeros(B,K,device=dev,requires_grad=True),
          inn=torch.zeros(B,T-1,4,device=dev,requires_grad=True))
def snap():
    with torch.no_grad():
        return free_rollout(pars,draw_noise(B,T,K,dev,IQ.DOF),Pi,s_phi=0.05,s_lt=0.0025,rho1_max=0.9,sample=False)
ro0=snap()
opt=torch.optim.Adam(list(pars.values()),lr=3e-3)
for s in range(400):
    opt.zero_grad()
    ro=free_rollout(pars,draw_noise(B,T,K,dev,IQ.DOF),Pi,s_phi=0.05,s_lt=0.0025,rho1_max=0.9,sample=True)
    rb,rd,roo=P.recon_terms(dec,hdec,ro["Z"],b,db,obs,recon="bce")
    ((rb+rd+roo+ro["kl_p"]+ro["kl_l"]+ro["kl_m"]).mean()).backward(); opt.step()
ro1=snap()
def bpm(lt): return 3000.0*4.0*np.exp(lt)/TWO_PI      # beats/min, 4 beats per bar
def crossings(phi):
    p=phi.cpu().numpy(); return np.where(np.diff(p)<-math.pi)[0]+1   # wrap points = bar starts
print("\n=== DOES THE DECODER GO SILENT? downbeat channel output ===")
for tag,ro in (("ORACLE",ro0),("DESTROYED",ro1)):
    with torch.no_grad():
        lg=dec(ro["Z"]); pdb=torch.sigmoid(lg[...,1]); pb=torch.sigmoid(lg[...,0])
        rb,rd,_=P.recon_terms(dec,hdec,ro["Z"],b,db,obs,recon="bce")
    tgt_rate=float(db.mean())
    print(f"  {tag:10s} p_downbeat: mean {float(pdb.mean()):.5f}  max {float(pdb.max()):.4f}  "
          f"frames>0.5 {int((pdb>0.5).sum()):5d}  | p_beat mean {float(pb.mean()):.4f} "
          f"| recon_db {float(rd.mean()):7.2f}")
print(f"  target downbeat rate {tgt_rate:.5f} ({tgt_rate*100:.2f}% of frames)")
print(f"  -> a decoder that NEVER fires scores BCE = -log(1-p) summed = near 0 on 99%+ zeros")
for i in (0,1,2):
    dbt=torch.nonzero(db[i]>0.5).squeeze(-1).cpu().numpy()
    lt0=ro0["lt"][i].cpu().numpy(); lt1=ro1["lt"][i].cpu().numpy()
    c0,c1=crossings(ro0["phi"][i]),crossings(ro1["phi"][i])
    ct=crossings(phi_true[i])
    def hits(c,t,tol=7):
        return sum(1 for x in t if len(c) and np.min(np.abs(c-x))<=tol)
    print(f"\n=== crop {i} | {len(dbt)} true downbeats, true bars start at frames {ct[:6]}... ===")
    print(f"  ORACLE   tempo {bpm(lt0).mean():6.1f} BPM  (min {bpm(lt0).min():.1f} max {bpm(lt0).max():.1f}, "
          f"wobble +-{100*(bpm(lt0).max()-bpm(lt0).min())/bpm(lt0).mean()/2:.1f}%)")
    print(f"           bar starts: {c0[:8]}  ({len(c0)} bars)  hits {hits(c0,dbt)}/{len(dbt)} downbeats")
    print(f"  DESTROYED tempo {bpm(lt1).mean():6.1f} BPM  (min {bpm(lt1).min():.1f} max {bpm(lt1).max():.1f}, "
          f"wobble +-{100*(bpm(lt1).max()-bpm(lt1).min())/bpm(lt1).mean()/2:.1f}%)")
    print(f"           bar starts: {c1[:8]}  ({len(c1)} bars)  hits {hits(c1,dbt)}/{len(dbt)} downbeats")
    print(f"  true downbeats:      {dbt[:8]}")
    ib0=np.diff(c0) if len(c0)>1 else np.array([0]); ib1=np.diff(c1) if len(c1)>1 else np.array([0])
    ibt=np.diff(ct) if len(ct)>1 else np.array([0])
    print(f"  bar LENGTHS (frames): true {ibt[:6]} (sd {ibt.std():.1f})")
    print(f"                      oracle {ib0[:6]} (sd {ib0.std():.1f})")
    print(f"                   destroyed {ib1[:6]} (sd {ib1.std():.1f})")
