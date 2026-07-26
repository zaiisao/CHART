import sys, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms')
import variant_b as VB
from probe2 import load, rollout, LayerMerge
from audit_common import load_split
from common import targets
DEV='cuda:0'; ARMS='/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms'; TWO_PI=2*math.pi

@torch.no_grad()
def main():
    tr=load_split('train',with_feats=True,cap=40)
    for tag in ['i_bern']:
        mg,model,ck=load(tag)
        # decoder response to log_tempo alone
        G=torch.linspace(-13,13,521,device=DEV)
        mt=F.one_hot(torch.full((521,),3,device=DEV,dtype=torch.long),4).float()
        z=model.z_features(mt,torch.zeros(521,device=DEV),G)
        p=torch.sigmoid(model.decoder(z)); q=torch.sigmoid(model.h_dec(z))
        print(f'{tag}: LOG-TEMPO sweep (-13..13, phase=0, m=4)')
        print(f'  p(beat) range=[{float(p[:,0].min()):.4f},{float(p[:,0].max()):.4f}]  '
              f'p(db) range=[{float(p[:,1].min()):.4f},{float(p[:,1].max()):.4f}]')
        print(f'  p(obs0) range=[{float(q[:,0].min()):.4f},{float(q[:,0].max()):.4f}]  '
              f'p(obs1)=[{float(q[:,1].min()):.4f},{float(q[:,1].max()):.4f}]')
        # is log_tempo a Morse wire? beat vs non-beat frames
        torch.manual_seed(0); rng=np.random.default_rng(0)
        fe,bb,dd=[],[],[]
        while len(fe)<16:
            s=tr[rng.integers(len(tr))]; T=s['feats'].shape[1]
            if T<=256: continue
            st=int(rng.integers(0,T-256))
            fe.append(torch.from_numpy(s['feats'][:,st:st+256,:].astype(np.float32)))
            b,d=targets(s['beats'],s['downs'],st,256)
            bb.append(torch.from_numpy(b)); dd.append(torch.from_numpy(d))
        f=torch.stack(fe).to(DEV); b=torch.stack(bb).to(DEV); d=torch.stack(dd).to(DEV)
        h=mg(f); M,P,L=rollout(model,h,b)
        lb=L[b>0.5]; ln=L[b<0.5]
        print(f'  posterior log_tempo at BEAT frames: mean={float(lb.mean()):+.3f} std={float(lb.std()):.3f} (n={lb.numel()})')
        print(f'  posterior log_tempo at NON-beat   : mean={float(ln.mean()):+.3f} std={float(ln.std()):.3f} (n={ln.numel()})')
        print(f'  separation = {float(lb.mean()-ln.mean()):+.3f} nats-of-"tempo" (Cohen d={float((lb.mean()-ln.mean())/L.std()):.2f})')
        db_=L[d>0.5]
        print(f'  at DOWNBEAT frames: mean={float(db_.mean()):+.3f}')
        # AUC of log_tempo as a beat detector
        from sklearn.metrics import roc_auc_score
        y=b.reshape(-1).cpu().numpy(); x=L.reshape(-1).cpu().numpy()
        print(f'  AUC(log_tempo -> beat) = {roc_auc_score(y,x):.4f}   '
              f'AUC(log_tempo -> downbeat) = {roc_auc_score(d.reshape(-1).cpu().numpy(),x):.4f}')
        ph=P.reshape(-1).cpu().numpy()
        print(f'  AUC(cos phi   -> beat) = {roc_auc_score(y,np.cos(ph)):.4f}  '
              f'AUC(sin phi -> beat) = {roc_auc_score(y,np.sin(ph)):.4f}')
main()
