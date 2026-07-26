"""Is the emission ARCHITECTURE capable of phase-contrast, and how much is phase worth?
Fit the SAME 7->128->2 head on ORACLE z_feat (ideal bar phase, true tempo, true meter)."""
import sys, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms')
import variant_b as VB
from audit_common import load_split, ideal_barphase, FPS
from vbpm.evaluate import _estimate_meter
from common import targets
TWO_PI=2*math.pi; ARMS='/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms'; DEV='cuda:0'

def build(split,cap):
    songs=load_split(split,cap=cap); d=np.load(f'{ARMS}/act_{split}.npz',allow_pickle=True)
    Z=[];O=[];PH=[];BT=[]
    for s in songs:
        T=s['T']; dref=s['downs']; ref=s['beats']
        if len(dref)<3: continue
        phi=ideal_barphase(dref,T,FPS,'extrap')
        if phi is None: continue
        m=_estimate_meter(ref,dref)
        lt=math.log(TWO_PI/max(float(np.median(np.diff(dref)))*FPS,1e-6))
        a=np.clip(np.asarray(d[s['stem']+'|act'],np.float32),1e-4,1-1e-4)
        oh=np.zeros((T,4),np.float32); oh[:,m-1]=1
        z=np.concatenate([np.cos(phi)[:,None],np.sin(phi)[:,None],np.full((T,1),lt),oh],1).astype(np.float32)
        b,db=targets(ref,dref,0,T)
        Z.append(z);O.append(a);PH.append(phi);BT.append(np.stack([b,db],1))
    return (torch.from_numpy(np.concatenate(Z)).to(DEV),torch.from_numpy(np.concatenate(O)).to(DEV),
            np.concatenate(PH),torch.from_numpy(np.concatenate(BT)).to(DEV))

Ztr,Otr,Ptr,Btr=build('train',60); Zev,Oev,Pev,Bev=build('eval',40)
print(f'train frames {len(Ztr)}  eval frames {len(Zev)}')

def bce_mean(pred,tgt): return float(F.binary_cross_entropy_with_logits(pred,tgt,reduction='none').sum(-1).mean())
# floors, per FRAME (x256 -> per crop)
floor=bce_mean(torch.logit(Oev.clamp(1e-4,1-1e-4)),Oev)
const=bce_mean(torch.logit(Otr.mean(0,keepdim=True).clamp(1e-4,1-1e-4)).expand_as(Oev),Oev)
print(f'OBS BCE per frame  (x256 = per crop):  perfect-pred FLOOR {floor:.4f} ({floor*256:.1f})   '
      f'constant-mean {const:.4f} ({const*256:.1f})   headroom {(const-floor)*256:.1f} nats/crop')
# phase-bin lookup ceiling (32 bins of TRUE bar phase, meter-4 songs pooled)
nb=32; bt=torch.from_numpy((Ptr/TWO_PI*nb).astype(int)%nb).to(DEV); be=torch.from_numpy((Pev/TWO_PI*nb).astype(int)%nb).to(DEV)
tab=torch.stack([Otr[bt==k].mean(0) for k in range(nb)]).clamp(1e-4,1-1e-4)
lut=bce_mean(torch.logit(tab)[be],Oev)
print(f'  phase-BIN lookup (oracle bar phase, 32 bins): {lut:.4f} ({lut*256:.1f})  -> phase is worth {(const-lut)*256:.1f} nats/crop of the {(const-floor)*256:.1f} available')

torch.manual_seed(0)
head=nn.Sequential(nn.Linear(7,128),nn.Tanh(),nn.Linear(128,2)).to(DEV)
opt=torch.optim.AdamW(head.parameters(),lr=3e-3)
for i in range(3000):
    idx=torch.randint(0,len(Ztr),(8192,),device=DEV)
    loss=F.binary_cross_entropy_with_logits(head(Ztr[idx]),Otr[idx],reduction='none').sum(-1).mean()
    opt.zero_grad();loss.backward();opt.step()
head.eval()
with torch.no_grad():
    ev=bce_mean(head(Zev),Oev)
    print(f'  SAME 7->128->2 head fit on ORACLE z: {ev:.4f} ({ev*256:.1f})')
    # obs_contrast of this oracle-fit head, exact same estimator as arm_i.py
    def ll(z,o): return float(-F.binary_cross_entropy_with_logits(head(z),o,reduction='none').sum(-1).mean())
    base=ll(Zev,Oev); offs=[]
    for k in range(1,12):
        p=(torch.from_numpy(Pev).float().to(DEV)+TWO_PI*k/12)%TWO_PI
        z2=Zev.clone(); z2[:,0]=p.cos(); z2[:,1]=p.sin(); offs.append(ll(z2,Oev))
    print(f'  ORACLE-FIT head obs_contrast = {math.exp(base-np.mean(offs)):.4f}   (trained VBPM emission: 1.000)')
    ck=torch.load(f'{ARMS}/arm_i_ii_bern.pt',map_location=DEV)
    m=VB.BarPointerVAE_B(h_dim=2,hidden=128,num_meters=4,obs_dim=2,obs_type='bern').to(DEV); m.load_state_dict(ck['model']); m.eval()
    vb=float(-m.obs_logp(Zev,Oev).mean())
    print(f'  TRAINED VBPM emission on the SAME oracle z: {vb:.4f} ({vb*256:.1f})  <-- worse than constant-mean {const*256:.1f}? {vb>const}')
