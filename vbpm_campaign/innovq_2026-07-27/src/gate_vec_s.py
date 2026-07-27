import sys, math, time, torch
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq","/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem"):
    sys.path.insert(0,p)
import pm_common as P, innovq as IQ
from rollout_vec import rollout_vec
from rollout_vec_s import rollout_loop_noise, rollout_vec_s, draw_noise
dev="cuda:0"; torch.manual_seed(0)
P.PHYS["gamma_phase"]=0.06; P.PHYS["dev_sigma"]=0.02
D=P.build_crops(P.load_songs("train"),n_per_song=1,seed=0,crop=1500,dev=dev)
m=IQ.InnovQ().to(dev); m.eval()
h,b=D["h"][:6],D["b"][:6]; Bn,T,_=h.shape
def dphi(a,c): return float(torch.abs(torch.angle(torch.exp(1j*(a-c)))).max())
# --- 0. the noise-consuming loop must reproduce the ORIGINAL loop (sample=False)
nz=draw_noise(Bn,T,m.K,dev,IQ.DOF)
with torch.no_grad():
    ref0=IQ.rollout(m,h,b,sample=False)
    rep0=rollout_loop_noise(m,h,b,nz,sample=False)
print(f"0) noise-loop == IQ.rollout (det): dphi {dphi(rep0['phi'],ref0['phi']):.2e}  "
      f"dlt {float((rep0['lt']-ref0['lt']).abs().max()):.2e}  "
      f"dKL {float((rep0['kl_p']-ref0['kl_p']).abs().max()):.2e}")
# --- 1. off-by-one: OLD vec vs NEW vec, deterministic
with torch.no_grad():
    old=rollout_vec(m,h,b,n_picard=4)
    new=rollout_vec_s(m,h,b,nz,sample=False,n_picard=4)
print(f"1) OFF-BY-ONE   old rollout_vec residual vs loop: dphi {dphi(old['phi'],ref0['phi']):.3e}")
print(f"                new rollout_vec_s residual vs loop: dphi {dphi(new['phi'],ref0['phi']):.3e}  "
      f"dlt {float((new['lt']-ref0['lt']).abs().max()):.3e}")
# --- 1b. does the residual shrink with Picard depth? (=> convergence, not bias)
for npic in (2,4,8,16):
    with torch.no_grad(): v=rollout_vec_s(m,h,b,nz,sample=False,n_picard=npic)
    print(f"   picard={npic:2d}  det dphi {dphi(v['phi'],ref0['phi']):.3e}")
# --- 2. SAMPLED equivalence, same noise
with torch.no_grad():
    ls=rollout_loop_noise(m,h,b,nz,sample=True)
    vs=rollout_vec_s(m,h,b,nz,sample=True,n_picard=8)
    for npic in (2,4,8,16):
        vv=rollout_vec_s(m,h,b,nz,sample=True,n_picard=npic)
        print(f"   picard={npic:2d}  sampled dphi {dphi(vv['phi'],ls['phi']):.3e}  "
              f"dkl_p {float((vv['kl_p']-ls['kl_p']).abs().max()):.3e}")
print(f"2) SAMPLED  dphi {dphi(vs['phi'],ls['phi']):.3e}  dlt {float((vs['lt']-ls['lt']).abs().max()):.3e}")
for k in ("kl_p","kl_l","kl_m","n_cross"):
    print(f"     {k:8s} loop {float(ls[k].mean()):12.4f}  vec {float(vs[k].mean()):12.4f}  "
          f"absdiff {float((ls[k]-vs[k]).abs().max()):.3e}")
print(f"     Z        maxdiff {float((ls['Z']-vs['Z']).abs().max()):.3e}")
# --- 3. gradients
def gvec(fn):
    m.zero_grad(set_to_none=True)
    r=fn(); (r["phi"].sin().sum()+r["lt"].sum()+r["kl_p"].sum()+r["kl_l"].sum()+r["kl_m"].sum()).backward()
    return torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
m.train()   # cudnn RNN backward requires training mode
ga=gvec(lambda: rollout_loop_noise(m,h,b,nz,sample=True))
gb=gvec(lambda: rollout_vec_s(m,h,b,nz,sample=True,n_picard=8))
cos=float(torch.nn.functional.cosine_similarity(ga,gb,dim=0))
print(f"3) GRADIENT cosine {cos:.6f}  norm ratio {float(gb.norm()/ga.norm()):.4f}")
# --- 4. speed
torch.cuda.synchronize(); t0=time.time()
for _ in range(3): rollout_loop_noise(m,h,b,nz,sample=True)
torch.cuda.synchronize(); tl=(time.time()-t0)/3
t0=time.time()
for _ in range(3): rollout_vec_s(m,h,b,nz,sample=True,n_picard=4)
torch.cuda.synchronize(); tv=(time.time()-t0)/3
print(f"4) SPEED loop {tl*1000:.0f}ms  vec {tv*1000:.0f}ms  speedup {tl/tv:.1f}x")
ok = dphi(vs['phi'],ls['phi'])<1e-4 and cos>0.999
print(f"\nGATE {'PASSED' if ok else 'FAILED'}")
