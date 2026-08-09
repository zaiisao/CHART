import sys,numpy as np,torch
from pathlib import Path
HERE=Path("/home/sogang/jaehoon/VBPM/rungs_lab/mert"); sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent))
from mert_r4_model import R4Conditioned, UNIFORM_FLOOR
DEV="cuda:3"; FPS=44100/1024
cache=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
ck=torch.load(HERE/"runs/mertr4_mertfull_bestsel.pt",weights_only=False)
m=R4Conditioned(fps=FPS,input_mode="featsmert",device=DEV,input_dim=ck["input_dim"]); m.load_state_dict(ck["model"]); m.eval()
mean,std=cache["feat_mean"].to(DEV),cache["feat_std"].to(DEV)
mm,ms=cache["mert_mean"].to(DEV),cache["mert_std"].to(DEV)
bpm=60.0*FPS/(m._min_interval+np.arange(m.num_tempi))
rng=np.random.default_rng(0)
ents=cache["val_entries"][:24]
def get(e):
    a=cache["val_acts"][e["stem"]]; f=cache["val_feats"][e["stem"]].astype(np.float32); mt=cache["val_mert"][e["stem"]].astype(np.float32)
    L=a.shape[0]
    if L>1400:
        s=(L-1400)//2; a,f,mt=a[s:s+1400],f[s:s+1400],mt[s:s+1400]
    a=torch.from_numpy(a).to(DEV); f=(torch.from_numpy(f).to(DEV)-mean)/std; mt=(torch.from_numpy(mt).to(DEV)-mm)/ms
    return a,f,mt
def mll(a,ti,prior_override=None):
    dens=m.chassis.log_class_densities(a)
    lp,lk,_=m.head_outputs(ti)
    if prior_override is not None: lp=prior_override
    li=m.conditioned_log_inits(lp)
    per=[dp.forward_log_likelihood(i,lk,dens,state_to_class=s) for dp,i,s in zip(m.chassis.dynamic_programs,li,m.chassis.state_to_classes)]
    return float(torch.logsumexp(torch.stack(per),0))/a.shape[0]
R={k:[] for k in("base","zero_trunk","mert_zero","mert_shuf","oracle_prior","uniform_prior")}
with torch.no_grad():
  for e in ents:
    a,f,mt=get(e); T=f.shape[0]; perm=torch.from_numpy(rng.permutation(T)).to(DEV)
    R["base"].append(-mll(a,torch.cat([f,mt],1)))
    R["mert_zero"].append(-mll(a,torch.cat([f,torch.zeros_like(mt)],1)))
    R["mert_shuf"].append(-mll(a,torch.cat([f,mt[perm]],1)))
    m._zero_trunk=True; R["zero_trunk"].append(-mll(a,torch.cat([f,mt],1))); m._zero_trunk=False
    ibi=np.diff(e["beat_times"]); ibi=ibi[ibi>1e-3]; tb=60.0/np.median(ibi)
    j=int(np.abs(bpm-tb).argmin())
    p=torch.full((m.num_tempi,),UNIFORM_FLOOR/m.num_tempi,device=DEV); p[j]+=1-UNIFORM_FLOOR
    R["oracle_prior"].append(-mll(a,torch.cat([f,mt],1),p.log()))
    u=torch.full((m.num_tempi,),1.0/m.num_tempi,device=DEV)
    R["uniform_prior"].append(-mll(a,torch.cat([f,mt],1),u.log()))
for k,v in R.items(): print(f"{k:14s} nll/frame {np.mean(v):.5f}  (delta vs base {np.mean(v)-np.mean(R['base']):+.5f})")
