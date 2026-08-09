import sys,numpy as np,torch
from pathlib import Path
from scipy.stats import spearmanr
cache=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
mm=cache["mert_mean"].numpy(); ms=cache["mert_std"].numpy()
fm=cache["feat_mean"].numpy(); fs=cache["feat_std"].numpy()
X_m=[];X_b=[];y=[]
for e in cache["val_entries"]:
    s=e["stem"]
    m=cache["val_mert"][s].astype(np.float32); f=cache["val_feats"][s].astype(np.float32)
    X_m.append(((m-mm)/ms).mean(0)); X_b.append(((f-fm)/fs).mean(0))
    ibi=np.diff(e["beat_times"]); ibi=ibi[ibi>1e-3]; y.append(np.log(60.0/np.median(ibi)))
X_m=np.array(X_m,dtype=np.float64);X_b=np.array(X_b,dtype=np.float64);y=np.array(y)
print("n",len(y),"mert dim",X_m.shape[1])
def ridge(Xtr,ytr,a):
    Xc=Xtr-Xtr.mean(0); yc=ytr-ytr.mean()
    # dual form (n<<d)
    K=Xc@Xc.T
    al=np.linalg.solve(K+a*np.eye(len(K)),yc)
    return lambda X:(X-Xtr.mean(0))@(Xc.T@al)+ytr.mean()
def cv(X,name):
    n=len(y); idx=np.random.default_rng(0).permutation(n); folds=np.array_split(idx,5)
    best=None
    for a in np.logspace(0,7,15):
        pred=np.zeros(n)
        for f in folds:
            tr=np.setdiff1d(idx,f); pred[f]=ridge(X[tr],y[tr],a)(X[f])
        s=spearmanr(pred,y).statistic; r2=1-((pred-y)**2).sum()/((y-y.mean())**2).sum()
        if best is None or s>best[1]: best=(a,s,r2)
    print(f"{name:20s} best alpha {best[0]:.1e} spearman {best[1]:.4f} r2 {best[2]:.3f}")
cv(X_m,"MERT meanpool")
cv(X_b,"BT feats meanpool")
cv(np.hstack([X_b,X_m]),"BT+MERT")
