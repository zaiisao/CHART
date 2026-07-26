"""PROBE 1: is obs_contrast=1.000 real, or a measurement artifact?
Independent re-implementation + direct decoder phase sweeps + weight forensics."""
import sys, math, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms')
import variant_b as VB
from arm_i import LayerMerge, build_obs_cache
from audit_common import load_split, FPS
from vbpm.evaluate import _estimate_meter
DEV='cuda:0'; TWO_PI=2*math.pi
ARMS='/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms'

def load(tag):
    ck=torch.load(f'{ARMS}/arm_i_{tag}.pt',map_location=DEV)
    cfg=ck['config']; obs=ck['obs']
    obs_dim=2 if obs!='pca_gauss' else 32
    obs_type='bern' if obs=='head_bern' else 'gauss'
    m=VB.BarPointerVAE_B(h_dim=768,hidden=cfg['hidden'],num_meters=4,obs_dim=obs_dim,obs_type=obs_type).to(DEV)
    m.load_state_dict(ck['model']); m.eval()
    mg=LayerMerge().to(DEV); mg.load_state_dict(ck['merge']); mg.eval()
    return mg,m,ck

# ---------------- (b) DENSE PHASE SWEEP, fixed tempo+meter -------------------
@torch.no_grad()
def sweep(model,tag):
    print(f'\n### (b) DENSE PHASE SWEEP  [{tag}] ###')
    G=721
    ph=torch.linspace(0,TWO_PI,G,device=DEV)
    for lt,mi in [(-2.66,3),(-2.66,2),(-3.0,3),(-2.0,3)]:
        mt=F.one_hot(torch.full((G,),mi,device=DEV,dtype=torch.long),model.K).float()
        z=model.z_features(mt,ph,torch.full((G,),lt,device=DEV))
        po=model.h_dec(z)                 # obs emission raw output (logits or mean)
        pb=model.decoder(z)               # beat/downbeat logits
        def rng(x,name):
            r=(x.max(0).values-x.min(0).values).cpu().numpy()
            s=x.std(0).cpu().numpy(); mu=x.mean(0).cpu().numpy()
            print(f'   {name:8s} lt={lt:+.2f} m={mi+1}  mean={np.round(mu,4)} '
                  f'maxmin={np.round(r,6)} std={np.round(s,6)}')
        rng(po,'h_dec'); rng(pb,'decoder')
    # ---- weight forensics on the FIRST layer of both nets ----
    print(f'\n### (b2) FIRST-LAYER INPUT-WEIGHT NORMS  [{tag}] ###')
    names=['cos','sin','logT','m1','m2','m3','m4']
    for nm,net in (('h_dec',model.h_dec),('decoder',model.decoder)):
        W=net[0].weight.detach()           # [hidden, 7]
        cn=W.abs().mean(0).cpu().numpy()
        c2=W.norm(dim=0).cpu().numpy()
        print(f'   {nm:8s} mean|w| per input: '+'  '.join(f'{n}={v:.4f}' for n,v in zip(names,cn)))
        print(f'   {nm:8s} L2   per input: '+'  '.join(f'{n}={v:.4f}' for n,v in zip(names,c2)))
        # effective gain: how much does output move per unit input change (jacobian at typical z)
    # second layer scale
    for nm,net in (('h_dec',model.h_dec),('decoder',model.decoder)):
        print(f'   {nm:8s} L2out={net[2].weight.detach().norm().item():.4f} '
              f'bias_out={net[2].bias.detach().cpu().numpy().round(4)}')

# --------- (a) INDEPENDENT contrast, obs AND beat decoders -------------------
def my_barphase(downs,T):
    """my own true bar phase: linear 0..2pi between consecutive downbeats, extrapolated."""
    t=(np.arange(T)+0.5)/FPS
    d=np.asarray(downs,float)
    # piecewise-linear index in "bar units"
    barpos=np.interp(t,d,np.arange(len(d)),left=np.nan,right=np.nan)
    pre=t<d[0]; post=t>d[-1]
    barpos[pre]=(t[pre]-d[0])/max(d[1]-d[0],1e-6)
    barpos[post]=len(d)-1+(t[post]-d[-1])/max(d[-1]-d[-2],1e-6)
    return (barpos%1.0)*TWO_PI

@torch.no_grad()
def contrast(model,obs_t,btgt,dtgt,phi,m,lt,n_off=12):
    T=len(phi); dv=obs_t.device
    mt=F.one_hot(torch.tensor([m-1]*T,device=dv),model.K).float()
    ltv=torch.full((T,),lt,device=dv)
    ph=torch.from_numpy(phi).float().to(dv)
    def ll(p):
        z=model.z_features(mt,p,ltv)
        o=float(model.obs_logp(z,obs_t).mean())
        lg=model.decoder(z)
        b=float((-F.binary_cross_entropy_with_logits(lg[:,0],btgt,reduction='none')).mean())
        d=float((-F.binary_cross_entropy_with_logits(lg[:,1],dtgt,reduction='none')).mean())
        return o,b,d
    t_o,t_b,t_d=ll(ph)
    O,B,D=[],[],[]
    for k in range(1,n_off):
        a,b_,c=ll((ph+TWO_PI*k/n_off)%TWO_PI); O.append(a);B.append(b_);D.append(c)
    return dict(obs=t_o-float(np.mean(O)), beat=t_b-float(np.mean(B)), db=t_d-float(np.mean(D)),
                obs_best=t_o-max(O), beat_best=t_b-max(B), db_best=t_d-max(D),
                ll_true_obs=t_o, ll_true_beat=t_b, ll_true_db=t_d)

def main():
    ev=load_split('eval',with_feats=False)
    print(f'eval songs: {len(ev)}')
    obs_all=np.load(f'{ARMS}/act_eval.npz',allow_pickle=True)
    for tag in ['i_bern','ii_bern','i_gauss']:
        mg,model,ck=load(tag)
        sweep(model,tag)
        mode='head_bern' if ck['obs']=='head_bern' else 'head_gauss'
        rows=[]
        for s in ev:
            T=s['T']; downs=s['downs']; beats=s['beats']
            if len(downs)<3: continue
            a=np.clip(np.asarray(obs_all[s['stem']+'|act'],np.float32),1e-4,1-1e-4)
            o=a if mode=='head_bern' else np.log(a/(1-a))
            obs_t=torch.from_numpy(o[:T]).to(DEV)
            phi=my_barphase(downs,T)
            bt=np.zeros(T,np.float32); dt=np.zeros(T,np.float32)
            for x in beats:
                i=int(round(x*FPS))
                if 0<=i<T: bt[i]=1
            for x in downs:
                i=int(round(x*FPS))
                if 0<=i<T: dt[i]=1
            m=_estimate_meter(beats,downs)
            bar=float(np.median(np.diff(downs)))*FPS
            lt=math.log(TWO_PI/max(bar,1e-6))
            r=contrast(model,obs_t,torch.from_numpy(bt).to(DEV),torch.from_numpy(dt).to(DEV),phi,m,lt)
            r['stem']=s['stem']; rows.append(r)
        print(f'\n### (a) INDEPENDENT PER-FRAME LOG-LIK CONTRAST (nats/frame) [{tag}] n={len(rows)} ###')
        for k in ['obs','beat','db','obs_best','beat_best','db_best']:
            v=np.array([r[k] for r in rows])
            print(f'   d_{k:9s} mean={v.mean():+.6f}  median={np.median(v):+.6f}  '
                  f'ratio=exp={math.exp(v.mean()):.5f}  frac>0={np.mean(v>0):.2f}')
        for k in ['ll_true_obs','ll_true_beat','ll_true_db']:
            v=np.array([r[k] for r in rows]); print(f'   {k:13s} mean={v.mean():.5f}')
        json.dump(rows,open(f'/home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf/contrast_{tag}.json','w'),indent=1,default=float)
main()
