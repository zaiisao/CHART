import sys,numpy as np,torch
from pathlib import Path
M=Path("/home/sogang/jaehoon/VBPM/rungs_lab/mert"); R4=Path("/home/sogang/jaehoon/VBPM/rungs_lab/r4")
sys.path.insert(0,str(M)); sys.path.insert(0,str(M.parent))
from mert_r4_model import R4Conditioned as MM, UNIFORM_FLOOR
DEV="cuda:1"; FPS=44100/1024
c=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
mean,std=c["feat_mean"].to(DEV),c["feat_std"].to(DEV); mm,ms=c["mert_mean"].to(DEV),c["mert_std"].to(DEV)
ckm=torch.load(M/"runs/mertr4_mertfull_bestsel.pt",weights_only=False)
mert=MM(fps=FPS,input_mode="featsmert",device=DEV,input_dim=ckm["input_dim"]); mert.load_state_dict(ckm["model"]); mert.eval()
ckb=torch.load(R4/"runs/r4_run2b_c1400_bestsel.pt",weights_only=False)
print("bt ckpt", {k:v for k,v in ckb.items() if k!="model"})
bt=MM(fps=FPS,input_mode="feats",device=DEV,input_dim=256); bt.load_state_dict(ckb["model"]); bt.eval()
def nll(m,a,ti): return float(-m.marginal_ll(a,ti)/a.shape[0])
A=[];B=[];O=[];Z=[]
bpm=60.0*FPS/(mert._min_interval+np.arange(mert.num_tempi))
with torch.no_grad():
 for e in c["val_entries"][:24]:
    s=e["stem"]; a=c["val_acts"][s]; f=c["val_feats"][s].astype(np.float32); mt=c["val_mert"][s].astype(np.float32)
    L=a.shape[0]
    if L>1400: st=(L-1400)//2; a,f,mt=a[st:st+1400],f[st:st+1400],mt[st:st+1400]
    a=torch.from_numpy(a).to(DEV); fz=(torch.from_numpy(f).to(DEV)-mean)/std; mz=(torch.from_numpy(mt).to(DEV)-mm)/ms
    A.append(nll(mert,a,torch.cat([fz,mz],1))); B.append(nll(bt,a,fz))
A=np.array(A);B=np.array(B);d=A-B
print("MERT %.5f  BT %.5f  paired diff %.5f +- %.5f (sem)  wins BT %d/24"%(A.mean(),B.mean(),d.mean(),d.std(ddof=1)/np.sqrt(24),(d>0).sum()))
from scipy.stats import wilcoxon
print("wilcoxon p", wilcoxon(A,B).pvalue)
