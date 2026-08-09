import sys,numpy as np,torch
from pathlib import Path
M=Path("/home/sogang/jaehoon/VBPM/rungs_lab/mert"); sys.path.insert(0,str(M)); sys.path.insert(0,str(M.parent))
from mert_r4_model import R4Conditioned, UNIFORM_FLOOR
DEV="cuda:1"; FPS=44100/1024
c=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
mean,std=c["feat_mean"].to(DEV),c["feat_std"].to(DEV); mm,ms=c["mert_mean"].to(DEV),c["mert_std"].to(DEV)
ck=torch.load(M/"runs/mertr4_mertfull_bestsel.pt",weights_only=False)
m=R4Conditioned(fps=FPS,input_mode="featsmert",device=DEV,input_dim=ck["input_dim"]); m.load_state_dict(ck["model"]); m.eval()
bpm=60.0*FPS/(m._min_interval+np.arange(m.num_tempi))
def mll(a,ti,po=None):
    dens=m.chassis.log_class_densities(a); lp,lk,_=m.head_outputs(ti)
    if po is not None: lp=po
    li=m.conditioned_log_inits(lp)
    per=[dp.forward_log_likelihood(i,lk,dens,state_to_class=s) for dp,i,s in zip(m.chassis.dynamic_programs,li,m.chassis.state_to_classes)]
    return float(torch.logsumexp(torch.stack(per),0))/a.shape[0]
B=[];O=[];rank=[]
with torch.no_grad():
 for e in c["val_entries"]:
    s=e["stem"];a=c["val_acts"][s];f=c["val_feats"][s].astype(np.float32);mt=c["val_mert"][s].astype(np.float32)
    L=a.shape[0]
    if L>1400: st=(L-1400)//2;a,f,mt=a[st:st+1400],f[st:st+1400],mt[st:st+1400]
    a=torch.from_numpy(a).to(DEV);fz=(torch.from_numpy(f).to(DEV)-mean)/std;mz=(torch.from_numpy(mt).to(DEV)-mm)/ms
    ti=torch.cat([fz,mz],1)
    B.append(-mll(a,ti))
    ibi=np.diff(e["beat_times"]); ibi=ibi[ibi>1e-3]; tb=60.0/np.median(ibi)
    j=int(np.abs(bpm-tb).argmin())
    p=torch.full((m.num_tempi,),UNIFORM_FLOOR/m.num_tempi,device=DEV); p[j]+=1-UNIFORM_FLOOR
    O.append(-mll(a,ti,p.log()))
    # per-song: is the annotated tempo bin the argmin-NLL delta bin? scan all bins
B=np.array(B);O=np.array(O);d=O-B
from scipy.stats import wilcoxon
print("n",len(B),"learned %.5f oracle %.5f  oracle-worse in %d/%d  mean delta %+.5f sem %.5f  wilcoxon p %.3g"%(B.mean(),O.mean(),(d>0).sum(),len(d),d.mean(),d.std(ddof=1)/np.sqrt(len(d)),wilcoxon(O,B).pvalue))
