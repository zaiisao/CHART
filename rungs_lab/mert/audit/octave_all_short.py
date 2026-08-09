"""Full-power octave test: for EVERY fold-0 val song, prior log-odds of mass at
annotated tempo vs at its octave partners (2x and 0.5x). n=146, no decode gate."""
import sys,json,numpy as np,torch
sys.path.insert(0,'/home/sogang/jaehoon/VBPM/rungs_lab/mert'); sys.path.insert(0,'/home/sogang/jaehoon/VBPM/rungs_lab/r4'); sys.path.insert(0,'/home/sogang/jaehoon/VBPM/rungs_lab')
FPS=44100/1024; DEV="cuda:1"
cache=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
mean,std=cache["feat_mean"].to(DEV),cache["feat_std"].to(DEV); mmean,mstd=cache["mert_mean"].to(DEV),cache["mert_std"].to(DEV)
def load(kind,path):
    M=__import__("mert_r4_model" if kind=="mert" else "r4_model").R4Conditioned
    ck=torch.load(path,weights_only=False)
    kw={} if kind=="r4" else {"input_dim":ck.get("input_dim")}
    m=M(fps=FPS,input_mode=ck["input"],device=DEV,**kw); m.load_state_dict(ck["model"]); m.eval(); return m,ck["input"]
def tin(inp,e):
    if inp=="acts": return torch.from_numpy(cache["val_acts"][e["stem"]]).to(DEV)
    f=torch.from_numpy(cache["val_feats"][e["stem"]].astype(np.float32)).to(DEV); f=(f-mean)/std
    if inp=="feats": return f
    mm=torch.from_numpy(cache["val_mert"][e["stem"]].astype(np.float32)).to(DEV)
    return torch.cat([f,(mm-mmean)/mstd],dim=1)
ann={e["stem"]:60.0/np.median(np.diff(e["beat_times"])[np.diff(e["beat_times"])>1e-3]) for e in cache["val_entries"]}
CK=[("mert","MERT_long","/home/sogang/jaehoon/VBPM/rungs_lab/mert/runs/mertr4_mertfull_bestsel.pt"),
    ("r4","BTunsup_long","/home/sogang/jaehoon/VBPM/rungs_lab/r4/runs/r4_run2b_c1400_bestsel.pt"),
    ("r4","BTsup_long","/home/sogang/jaehoon/VBPM/rungs_lab/r4/runs/r4_supj_full_bestsel.pt"),
    ("mert","BT_SHORT150","/home/sogang/jaehoon/VBPM/rungs_lab/mert/runs/short_bt150_bestsel.pt"),
    ("mert","MERT_SHORT150","/home/sogang/jaehoon/VBPM/rungs_lab/mert/runs/short_mert150_bestsel.pt")]
R={}
for kind,name,path in CK:
    m,inp=load(kind,path); g=60.0*FPS/(m._min_interval+np.arange(m.num_tempi))
    def dens(p,b):
        s=np.abs(g/b-1.0)<0.06
        return (p[s].sum()/max(s.sum(),1)) if s.sum() else np.nan
    rows=[]
    with torch.no_grad():
        for e in cache["val_entries"]:
            _,_,d=m.head_outputs(tin(inp,e)); p=d["prior"].cpu().numpy(); a=ann[e["stem"]]
            dc=dens(p,a); dh=dens(p,a*2); dl=dens(p,a/2)
            partners=[x for x in (dh,dl) if x==x]
            if not partners or dc!=dc: continue
            rows.append(dict(stem=e["stem"],ds=e["dataset"],lo=float(np.log((dc+1e-12)/(max(partners)+1e-12)))))
    R[name]={r["stem"]:r["lo"] for r in rows}; R.setdefault("_ds",{}).update({r["stem"]:r["ds"] for r in rows})
    print(name,"n",len(rows),"mean log-odds(correct vs best octave partner) %.3f"%np.mean([r["lo"] for r in rows]),
          "frac>0 %.3f"%np.mean([r["lo"]>0 for r in rows]),flush=True)
from scipy.stats import wilcoxon
common=sorted(set(R["MERT"])&set(R["BTunsup"])&set(R["BTsup"]))
a=np.array([R["MERT"][s] for s in common]); b=np.array([R["BTunsup"][s] for s in common]); s_=np.array([R["BTsup"][s] for s in common])
print("n_common",len(common))
print("MERT-BTunsup  mean %+.3f  wins %d/%d  wilcoxon p=%.3g"%((a-b).mean(),(a>b).sum(),len(a),wilcoxon(a,b).pvalue))
print("BTsup-BTunsup mean %+.3f  wins %d/%d  wilcoxon p=%.3g"%((s_-b).mean(),(s_>b).sum(),len(a),wilcoxon(s_,b).pvalue))
print("MERT-BTsup    mean %+.3f  wins %d/%d  wilcoxon p=%.3g"%((a-s_).mean(),(a>s_).sum(),len(a),wilcoxon(a,s_).pvalue))
json.dump({k:v for k,v in R.items()},open("octave_all.json","w"),indent=1)
