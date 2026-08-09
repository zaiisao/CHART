import sys, numpy as np, torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM/rungs_lab")
C="/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt"
c=torch.load(C,weights_only=False)
print("keys",[k for k in c.keys()])
print("layers",c.get("mert_layers"))
mm,ms=c["mert_mean"],c["mert_std"]
fm,fs=c["feat_mean"],c["feat_std"]
print("mert_mean",tuple(mm.shape),"mert_std",tuple(ms.shape),"feat",tuple(fm.shape))
print("mert_std stats min %.4g p1 %.4g med %.4g max %.4g  frac_at_clamp %.4f"%(
  ms.min(),ms.quantile(0.01),ms.median(),ms.max(),(ms<=1e-3+1e-9).float().mean()))
print("feat_std stats min %.4g med %.4g max %.4g"%(fs.min(),fs.median(),fs.max()))
# post-standardization variance per stream on a few crops
bt_v=[];mt_v=[]
for cc in c["crops"][:8]:
    f=torch.from_numpy(cc["feats"]); m=torch.from_numpy(cc["mert"].astype(np.float32))
    bt_v.append((((f-fm)/fs)**2).mean().item()); mt_v.append((((m-mm)/ms)**2).mean().item())
    print(cc["stem"],"acts",cc["acts"].shape,"feats",f.shape,"mert",m.shape,"start",cc["start"])
print("post-std mean-square: BT %.4f  MERT %.4f"%(np.mean(bt_v),np.mean(mt_v)))
# per-dim post-std variance distribution on one crop
m=torch.from_numpy(c["crops"][0]["mert"].astype(np.float32)); z=(m-mm)/ms
v=z.var(0); print("MERT per-dim post-std var: med %.3f p95 %.3f max %.3f frac<0.01 %.3f"%(v.median(),v.quantile(.95),v.max(),(v<0.01).float().mean()))
f=torch.from_numpy(c["crops"][0]["feats"]); zf=(f-fm)/fs; vf=zf.var(0)
print("BT per-dim post-std var: med %.3f max %.3f"%(vf.median(),vf.max()))
# how many val songs, length agreement
bad=0
for e in c["val_entries"][:50]:
    a=c["val_acts"][e["stem"]]; m=c["val_mert"][e["stem"]]; f=c["val_feats"][e["stem"]]
    if not (a.shape[0]==m.shape[0]==f.shape[0]): bad+=1; print("LEN MISMATCH",e["stem"],a.shape,f.shape,m.shape)
print("val len mismatches in 50:",bad, "n_val",len(c["val_entries"]),"n_crops",len(c["crops"]))
torch.save({"stems":[cc["stem"] for cc in c["crops"][:6]]}, "/home/sogang/jaehoon/VBPM/rungs_lab/mert/audit/stems.pt")
# lag check: MERT energy-derivative vs BT beat activation
from scipy.signal import correlate
for cc in c["crops"][:6]:
    a=cc["acts"][:,0]; m=cc["mert"].astype(np.float32)
    z=(m-mm.numpy())/ms.numpy()
    d=np.abs(np.diff(z,axis=0)).mean(1)   # spectral-flux-ish from MERT
    d=(d-d.mean())/(d.std()+1e-9); aa=(a-a.mean())/(a.std()+1e-9); aa=aa[1:]
    n=len(d); cc_=correlate(aa,d,"full")/n
    lags=np.arange(-n+1,n); sel=(np.abs(lags)<=20)
    L=lags[sel]; V=cc_[sel]
    print("%-28s bestlag %+d  r %.3f  r@0 %.3f"%(cc["stem"],L[np.argmax(V)],V.max(),V[L==0][0]))
