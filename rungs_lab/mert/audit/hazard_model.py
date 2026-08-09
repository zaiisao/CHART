import sys,numpy as np,torch
sys.path.insert(0,'/home/sogang/jaehoon/VBPM/rungs_lab/mert'); sys.path.insert(0,'/home/sogang/jaehoon/VBPM/rungs_lab')
from mert_r4_model import R4Conditioned
FPS=44100/1024; DEV="cuda:1"
cache=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
mean,std=cache["feat_mean"].to(DEV),cache["feat_std"].to(DEV); mmean,mstd=cache["mert_mean"].to(DEV),cache["mert_std"].to(DEV)
ck=torch.load("/home/sogang/jaehoon/VBPM/rungs_lab/mert/runs/mertr4_mertfull_bestsel.pt",weights_only=False)
m=R4Conditioned(fps=FPS,input_mode=ck["input"],device=DEV,input_dim=ck.get("input_dim")); m.load_state_dict(ck["model"]); m.eval()
H=[];W=[];hz_all=[]
with torch.no_grad():
    for e in cache["val_entries"][:60]:
        f=torch.from_numpy(cache["val_feats"][e["stem"]].astype(np.float32)).to(DEV); f=(f-mean)/std
        mm=torch.from_numpy(cache["val_mert"][e["stem"]].astype(np.float32)).to(DEV)
        t=torch.cat([f,(mm-mmean)/mstd],dim=1)
        _,_,d=m.head_outputs(t); w=d["component_weights"].cpu().numpy(); hz=1-w[:,0]
        wd=1.0/d["lambda_t"].cpu().numpy()
        H.append((hz.mean(),hz.std(),hz.std()/max(hz.mean(),1e-9),hz.min(),hz.max()))
        W.append((wd.mean(),wd.std()/max(wd.mean(),1e-9)))
        hz_all.append(hz)
H=np.array(H);W=np.array(W)
print("hazard: mean %.4f | within-song sd med %.5f | CV med %.4f | range med %.4f"%(H[:,0].mean(),np.median(H[:,1]),np.median(H[:,2]),np.median(H[:,4]-H[:,3])))
print("BETWEEN-song sd of mean hazard %.5f"%H[:,0].std())
print("width: CV med %.5f"%np.median(W[:,1]))
# temporal smoothness: lag-1 autocorr and effective bandwidth
x=hz_all[0]; x=x-x.mean()
print("hazard lag-1 autocorr %.4f  lag-43(1s) %.4f"%(np.corrcoef(x[:-1],x[1:])[0,1],np.corrcoef(x[:-43],x[43:])[0,1]))
