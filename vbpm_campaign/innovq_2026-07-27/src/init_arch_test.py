"""Does the init head's roll-INVARIANCE come from mean-pooling? Compare how much the head's
phase output moves under roll when fed (a) [mean-pool, local] as now vs (b) [local, local].
No training: pure forward-pass sensitivity."""
import sys, math
import numpy as np, torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P, innovq as IQ
dev="cuda:0"; torch.manual_seed(0); TWO_PI=2*math.pi
D=P.build_crops(P.load_songs("train"),n_per_song=2,seed=0,dev=dev)
m=IQ.InnovQ().to(dev)
ck=torch.load("innovq_pf_sm101_s0.pt",map_location=dev,weights_only=False)
m.load_state_dict(ck.get("model",ck),strict=False); m.eval()
idx=torch.arange(48,device=dev); K=m.K
def phase_from(head_in):
    v=m.init_head(head_in)
    return torch.atan2(v[:,K+1],v[:,K])%TWO_PI
with torch.no_grad():
    h,b=D["h"][idx],D["b"][idx]
    for k in (32,64,128):
        c0=m.encode_posterior(h,b); c1=m.encode_posterior(torch.roll(h,k,1),torch.roll(b,k,1))
        # (a) current: [mean-pool, ctx0]
        pa0=phase_from(torch.cat([c0.mean(1),c0[:,0]],-1)); pa1=phase_from(torch.cat([c1.mean(1),c1[:,0]],-1))
        da=float(torch.abs(torch.angle(torch.exp(1j*(pa1-pa0)))).median())
        # (b) local-only: [ctx0, ctx0]  (same head, pooling replaced by local)
        pb0=phase_from(torch.cat([c0[:,0],c0[:,0]],-1)); pb1=phase_from(torch.cat([c1[:,0],c1[:,0]],-1))
        db=float(torch.abs(torch.angle(torch.exp(1j*(pb1-pb0)))).median())
        # (c) local window mean (first 32 frames) -- position-preserving summary
        pc0=phase_from(torch.cat([c0[:,:32].mean(1),c0[:,0]],-1)); pc1=phase_from(torch.cat([c1[:,:32].mean(1),c1[:,0]],-1))
        dc=float(torch.abs(torch.angle(torch.exp(1j*(pc1-pc0)))).median())
        need=float((k*torch.exp(D["lt"][idx][:,0])).median())
        print(f"roll k={k:3d} | needed {need:.3f} rad || (a) pooled+local {da:.4f} | (b) local-only {db:.4f} | (c) window32+local {dc:.4f}")
print("\n(a)=current architecture. Higher response = more position-aware.")
