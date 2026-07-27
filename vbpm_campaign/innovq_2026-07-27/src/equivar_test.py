"""TEST: (A) is the init head roll-INVARIANT by construction (ctx.mean pooling)?
(B) does a self-supervised roll-EQUIVARIANCE loss (no labels) make the encoder listen?
Metric: roll response = how far the predicted phase moves when audio is rolled k frames.
Target = k*phidot (what a listening model must do). Metronome -> 0."""
import sys, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
dev="cuda:0"; torch.manual_seed(0); rng=np.random.default_rng(0)
TWO_PI=2*math.pi
D=P.build_crops(P.load_songs("train"),n_per_song=2,seed=0,dev=dev)
N=D["h"].shape[0]

def roll_response(model, idx, k=64):
    """median |Δφ(t=0)| when audio+beats rolled by k frames. Listening model: ≈ k*phidot."""
    with torch.no_grad():
        h,b=D["h"][idx],D["b"][idx]
        r0=IQ.rollout(model,h,b,sample=False)
        r1=IQ.rollout(model,torch.roll(h,k,1),torch.roll(b,k,1),sample=False)
        d=torch.abs(torch.angle(torch.exp(1j*(r1["phi"][:,0]-r0["phi"][:,0]))))
        expect=float((k*torch.exp(r0["lt"][:,0])).median())
        return float(d.median()), expect

# ---------- (A) diagnose the current init head ----------
m=IQ.InnovQ().to(dev)
ck=torch.load("innovq_pf_sm101_s0.pt",map_location=dev,weights_only=False)
m.load_state_dict(ck.get("model",ck),strict=False); m.eval()
idx=torch.arange(min(48,N),device=dev)
resp,exp_=roll_response(m,idx)
print(f"(A) TRAINED innovq: roll-64 response {resp:.4f} rad | a listening model needs ~{exp_:.4f} | ratio {resp/max(exp_,1e-9):.3f}")
# is it the mean-pooling? feed ctx with pooling removed vs kept
with torch.no_grad():
    ctx=m.encode_posterior(D["h"][idx],D["b"][idx])
    ctx_r=m.encode_posterior(torch.roll(D["h"][idx],64,1),torch.roll(D["b"][idx],64,1))
    pooled_shift=float((ctx.mean(1)-ctx_r.mean(1)).abs().mean())
    local_shift=float((ctx[:,0]-ctx_r[:,0]).abs().mean())
print(f"    ctx.mean(1) changes by {pooled_shift:.5f} under roll (invariant if ~0) | ctx[:,0] changes by {local_shift:.5f}")

# ---------- (B) self-supervised equivariance training (NO labels) ----------
print("\n(B) training with roll-equivariance loss ONLY (no beat labels, no teacher):")
m2=IQ.InnovQ().to(dev)
m2.load_state_dict(ck.get("model",ck),strict=False); m2.train()
opt=torch.optim.AdamW(m2.parameters(),lr=3e-4)
for s in range(1,201):
    i=torch.tensor(rng.integers(0,N,12),device=dev,dtype=torch.long)
    k=int(rng.integers(16,96))
    h,b=D["h"][i],D["b"][i]
    r0=IQ.rollout(m2,h,b,sample=False)
    r1=IQ.rollout(m2,torch.roll(h,k,1),torch.roll(b,k,1),sample=False)
    tgt=(k*torch.exp(r0["lt"][:,0].detach()))            # required phase shift
    d=wrap_=torch.angle(torch.exp(1j*(r1["phi"][:,0]-r0["phi"][:,0])))
    L=((1-torch.cos(d-tgt))).mean()                       # equivariance: shift must match
    opt.zero_grad(); L.backward()
    torch.nn.utils.clip_grad_norm_(m2.parameters(),5.0); opt.step()
    if s%50==0:
        m2.eval(); r,e=roll_response(m2,idx); m2.train()
        print(f"  s{s:3d} L_equi={float(L):.4f} | roll response {r:.4f} vs needed {e:.4f} (ratio {r/max(e,1e-9):.2f})")
print("DONE_EQUIVAR")
