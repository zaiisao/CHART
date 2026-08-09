import sys,numpy as np,torch,mir_eval.beat as meb
from pathlib import Path
M=Path("/home/sogang/jaehoon/VBPM/rungs_lab/mert");R4=Path("/home/sogang/jaehoon/VBPM/rungs_lab/r4")
sys.path.insert(0,str(M));sys.path.insert(0,str(M.parent))
from mert_r4_model import R4Conditioned as MM
DEV="cuda:3";FPS=44100/1024
c=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
mean,std=c["feat_mean"].to(DEV),c["feat_std"].to(DEV);mm,ms=c["mert_mean"].to(DEV),c["mert_std"].to(DEV)
ckb=torch.load(R4/"runs/r4_run2b_c1400_bestsel.pt",weights_only=False)
bt=MM(fps=FPS,input_mode="feats",device=DEV,input_dim=256);bt.load_state_dict(ckb["model"]);bt.eval()
fs=[]
with torch.no_grad():
 for e in c["val_entries"]:
    s=e["stem"];a=c["val_acts"][s]
    f=(torch.from_numpy(c["val_feats"][s].astype(np.float32)).to(DEV)-mean)/std
    ev=bt.decode(a,f.cpu().numpy(),deploy=True)
    r,q=meb.trim_beats(e["beat_times"]),meb.trim_beats(ev["beats"])
    fs.append(meb.f_measure(r,q) if len(q) and len(r) else 0.0)
fs=np.array(fs);np.save("f_btrun2b.npy",fs)
fm=np.load("f_mertfull.npy")
d=fm-fs
from scipy.stats import wilcoxon
print("BT-only F %.4f  MERT F %.4f"%(fs.mean(),fm.mean()))
print("paired diff %.4f +- %.4f (sem)  nonzero-diff songs %d"%(d.mean(),d.std(ddof=1)/np.sqrt(len(d)),(d!=0).sum()))
print("wilcoxon p", wilcoxon(fm,fs).pvalue if (d!=0).any() else "all identical")
