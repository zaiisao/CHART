"""3-MIN TEST: is the innovq posterior "aligned but shifted" or "right tempo, blind"?
Cut each eval song into consecutive crops; per crop measure its circular-mean phase error
(its offset). SHIFTED -> offsets agree within a song (low circular sd). BLIND -> offsets
scatter uniformly. Also roll-null: shift inputs 128 frames; a true aligned model's offsets
must move with the roll."""
import sys, math, glob
import numpy as np, torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_postmortem")
import pm_common as P
import innovq as IQ
dev="cuda:0"; torch.manual_seed(0)
TWO_PI=2*math.pi
def circ_sd(a):
    R=np.abs(np.exp(1j*np.asarray(a)).mean()); return float(np.sqrt(-2*np.log(max(R,1e-9)))), float(R)

# rebuild eval crops but keep song identity: 4 consecutive 256-frame crops per song
songs=P.load_songs("eval")
CK=sorted(glob.glob("innovq_pf_sm101_s*.pt"))+sorted(glob.glob("innovq_label_s*.pt"))
print("checkpoints:",[c.split('/')[-1] for c in CK])
for ckp in CK[:2]:
    ck=torch.load(ckp,map_location=dev,weights_only=False)
    model=IQ.InnovQ().to(dev)
    model.load_state_dict(ck.get("model",ck),strict=False); model.eval()
    per_song=[]; rolled=[]
    with torch.no_grad():
        for s in songs[:25]:
            act=torch.tensor(np.asarray(s["act"],np.float32),device=dev)
            T=act.shape[0]
            if T< 4*256+300: continue
            offs=[];offs_r=[]
            for k in range(4):
                a=k*256; b=a+256
                h=act[a:b].unsqueeze(0)
                bt=torch.zeros(1,256,device=dev)
                out=IQ.rollout(model,h,bt,sample=False)
                phi=out["phi"] if "phi" in out else torch.atan2(out["Z"][...,1],out["Z"][...,0])%(2*math.pi)
                tphi=torch.tensor(P.oracle_z(s["beats"],s["downs"],T)[0][a:b],device=dev,dtype=torch.float32)
                offs.append(float(torch.angle(torch.exp(1j*(phi[0]-tphi)).mean())))
                hr=torch.roll(act,128,0)[a:b].unsqueeze(0)
                outr=IQ.rollout(model,hr,bt,sample=False)
                phir=outr["phi"] if "phi" in outr else torch.atan2(outr["Z"][...,1],outr["Z"][...,0])%(2*math.pi)
                offs_r.append(float(torch.angle(torch.exp(1j*(phir[0]-tphi)).mean())))
            sd,R=circ_sd(offs); per_song.append((sd,R))
            rolled.append(float(np.abs(np.angle(np.exp(1j*(np.array(offs_r)-np.array(offs))).mean()))))
    if not per_song: print(ckp,"no usable songs"); continue
    sds=np.array([p[0] for p in per_song]); Rs=np.array([p[1] for p in per_song])
    print(f"\n{ckp}: {len(per_song)} songs x4 crops")
    print(f"  within-song offset circ-SD: median {np.median(sds):.3f} rad   (SHIFTED if <0.6, BLIND if >1.5)")
    print(f"  within-song offset concentration R: median {np.median(Rs):.3f}  (1=identical offsets, 0=scatter)")
    print(f"  roll-128 offset response: median |delta| {np.median(rolled):.3f} rad  (aligned model must respond ~1.0+)")
