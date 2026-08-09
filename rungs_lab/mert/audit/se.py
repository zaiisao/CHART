import sys,numpy as np,torch,mir_eval.beat as meb
from pathlib import Path
HERE=Path("/home/sogang/jaehoon/VBPM/rungs_lab/mert"); sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent))
from mert_r4_model import R4Conditioned
DEV="cuda:1"; FPS=44100/1024
cache=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
ck=torch.load(HERE/"runs/mertr4_mertfull_bestsel.pt",weights_only=False)
m=R4Conditioned(fps=FPS,input_mode="featsmert",device=DEV,input_dim=ck["input_dim"]);m.load_state_dict(ck["model"]);m.eval()
mean,std=cache["feat_mean"].to(DEV),cache["feat_std"].to(DEV);mm,ms=cache["mert_mean"].to(DEV),cache["mert_std"].to(DEV)
fs=[]
with torch.no_grad():
 for e in cache["val_entries"]:
    s=e["stem"];a=cache["val_acts"][s]
    f=(torch.from_numpy(cache["val_feats"][s].astype(np.float32)).to(DEV)-mean)/std
    mt=(torch.from_numpy(cache["val_mert"][s].astype(np.float32)).to(DEV)-mm)/ms
    ev=m.decode(a,torch.cat([f,mt],1).cpu().numpy(),deploy=True)
    r,q=meb.trim_beats(e["beat_times"]),meb.trim_beats(ev["beats"])
    fs.append(meb.f_measure(r,q) if len(q) and len(r) else 0.0)
fs=np.array(fs);print("mean F %.4f  per-song std %.3f  SE %.4f  n=%d"%(fs.mean(),fs.std(ddof=1),fs.std(ddof=1)/np.sqrt(len(fs)),len(fs)))
print("min detectable diff (paired would be smaller); family spread 0.9033-0.9155 = %.4f"%(0.9155-0.9033))
np.save("f_mertfull.npy",fs)
