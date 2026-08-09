import numpy as np,torch
from scipy.stats import spearmanr
c=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
mm=c["mert_mean"].numpy();ms=c["mert_std"].numpy();fm=c["feat_mean"].numpy();fs=c["feat_std"].numpy()
Xm=[];Xb=[];y=[];ds=[]
for e in c["val_entries"]:
    s=e["stem"];m=c["val_mert"][s].astype(np.float32);f=c["val_feats"][s].astype(np.float32)
    Xm.append(((m-mm)/ms).mean(0));Xb.append(((f-fm)/fs).mean(0))
    ibi=np.diff(e["beat_times"]);ibi=ibi[ibi>1e-3];y.append(np.log(60.0/np.median(ibi)))
    ds.append(e.get("dataset", s.split("_")[0] if "_" in s else "?"))
Xm=np.array(Xm,float);Xb=np.array(Xb,float);y=np.array(y);ds=np.array(ds)
print("datasets:",{d:int((ds==d).sum()) for d in np.unique(ds)})
def ridge(Xtr,ytr,a):
    Xc=Xtr-Xtr.mean(0);yc=ytr-ytr.mean();K=Xc@Xc.T
    al=np.linalg.solve(K+a*np.eye(len(K)),yc);return lambda X:(X-Xtr.mean(0))@(Xc.T@al)+ytr.mean()
def run(X,name,folds):
    n=len(y)
    for a in [None]:
        pass
    # NESTED: inner alpha selection on train only
    pred=np.zeros(n)
    for te in folds:
        tr=np.setdiff1d(np.arange(n),te)
        # inner 4-fold on tr
        best=None;inner=np.array_split(np.random.default_rng(1).permutation(tr),4)
        for al in np.logspace(0,7,15):
            ip=np.zeros(len(tr));pos={v:i for i,v in enumerate(tr)}
            for f in inner:
                itr=np.setdiff1d(tr,f)
                p=ridge(X[itr],y[itr],al)(X[f])
                for v,q in zip(f,p): ip[pos[v]]=q
            s=spearmanr(ip,y[tr]).statistic
            if best is None or s>best[1]: best=(al,s)
        pred[te]=ridge(X[tr],y[tr],best[0])(X[te])
    print(f"{name:22s} spearman {spearmanr(pred,y).statistic:.4f}")
n=len(y);rnd=np.array_split(np.random.default_rng(0).permutation(n),5)
grp=[np.where(ds==d)[0] for d in np.unique(ds) if (ds==d).sum()>0]
print("--- random 5-fold, NESTED alpha")
run(Xb,"BT",rnd);run(Xm,"MERT",rnd);run(np.hstack([Xb,Xm]),"BT+MERT",rnd)
print("--- leave-one-dataset-out")
run(Xb,"BT",grp);run(Xm,"MERT",grp);run(np.hstack([Xb,Xm]),"BT+MERT",grp)
print("--- within-dataset spearman (BT), random 5fold nested")
for d in np.unique(ds):
    m=np.where(ds==d)[0]
    if len(m)<25: print("  skip",d,len(m));continue
    Xd=Xb[m];yd=y[m];
    pr=np.zeros(len(m));fo=np.array_split(np.random.default_rng(0).permutation(len(m)),5)
    for f in fo:
        tr=np.setdiff1d(np.arange(len(m)),f)
        Xc=Xd[tr]-Xd[tr].mean(0);K=Xc@Xc.T
        al=np.linalg.solve(K+3.2*np.eye(len(K)),yd[tr]-yd[tr].mean())
        pr[f]=(Xd[f]-Xd[tr].mean(0))@(Xc.T@al)+yd[tr].mean()
    print(f"  {d} n={len(m)} BT spearman {spearmanr(pr,yd).statistic:.3f}")
