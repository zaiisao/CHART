import sys, json, numpy as np, torch
sys.path.insert(0,'/home/sogang/jaehoon/VBPM/rungs_lab/mert')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM/rungs_lab/r4')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM/rungs_lab')
FPS=44100/1024; DEV="cuda:1"
cache=torch.load("/disk4/jaehoon/VBPM_cache/mert/cache_mert_fold0_c1400.pt",weights_only=False)
mean,std=cache["feat_mean"].to(DEV),cache["feat_std"].to(DEV)
mmean,mstd=cache["mert_mean"].to(DEV),cache["mert_std"].to(DEV)

def load(kind,path):
    if kind=="mert":
        from mert_r4_model import R4Conditioned as M
    else:
        import importlib; sys.modules.pop('r4_model',None)
        from r4_model import R4Conditioned as M
    ck=torch.load(path,weights_only=False)
    kw={} if kind=="r4" else {"input_dim":ck.get("input_dim")}
    m=M(fps=FPS,input_mode=ck["input"],device=DEV,**kw)
    m.load_state_dict(ck["model"]); m.eval(); return m,ck["input"]

def tin(m,inp,e):
    if inp=="acts": return torch.from_numpy(cache["val_acts"][e["stem"]]).to(DEV)
    f=torch.from_numpy(cache["val_feats"][e["stem"]].astype(np.float32)).to(DEV); f=(f-mean)/std
    if inp=="feats": return f
    mm=torch.from_numpy(cache["val_mert"][e["stem"]].astype(np.float32)).to(DEV)
    return torch.cat([f,(mm-mmean)/mstd],dim=1)

ann={}
for e in cache["val_entries"]:
    ibi=np.diff(e["beat_times"]); ibi=ibi[ibi>1e-3]; ann[e["stem"]]=60.0/np.median(ibi)

CK=[("mert","MERT","/home/sogang/jaehoon/VBPM/rungs_lab/mert/runs/mertr4_mertfull_bestsel.pt"),
    ("r4","BTunsup","/home/sogang/jaehoon/VBPM/rungs_lab/r4/runs/r4_run2b_c1400_bestsel.pt"),
    ("r4","BTsup","/home/sogang/jaehoon/VBPM/rungs_lab/r4/runs/r4_supj_full_bestsel.pt")]

# ---- case set from BTunsup zero-trunk decode, widened window ----
mref,iref=load(*CK[1][:1],CK[1][2]) if False else load("r4","/home/sogang/jaehoon/VBPM/rungs_lab/r4/runs/r4_run2b_c1400_bestsel.pt")
g=60.0*FPS/(mref._min_interval+np.arange(mref.num_tempi))
mref._zero_trunk=True
cases_narrow,cases_wide=[],[]
with torch.no_grad():
    for e in cache["val_entries"]:
        a=cache["val_acts"][e["stem"]]
        ev=mref.decode(a, a if iref=="acts" else tin(mref,iref,e).cpu().numpy(), deploy=True)
        if len(ev["beats"])<4: continue
        est=60.0/np.median(np.diff(ev["beats"])); r=est/ann[e["stem"]]
        c={"stem":e["stem"],"est":float(est),"ann":float(ann[e["stem"]]),"r":float(r)}
        if 1.8<r<2.2 or 0.45<r<0.55: cases_narrow.append(c)
        if 1.6<r<2.6 or 0.38<r<0.62: cases_wide.append(c)
mref._zero_trunk=False
print("narrow n=",len(cases_narrow),"wide n=",len(cases_wide),flush=True)

def masses(prior,bpm):
    sel=np.abs(g/bpm-1.0)<0.10
    return float(prior[sel].sum()), int(sel.sum())

out={}
for kind,name,path in CK:
    m,inp=load(kind,path)
    rows=[]
    with torch.no_grad():
        for c in cases_wide:
            e=next(x for x in cache["val_entries"] if x["stem"]==c["stem"])
            _,_,d=m.head_outputs(tin(m,inp,e)); p=d["prior"].cpu().numpy()
            mc,nc=masses(p,c["ann"]); mw,nw=masses(p,c["est"])
            rows.append(dict(stem=c["stem"],mc=mc,nc=nc,mw=mw,nw=nw,
                             dens_c=mc/nc,dens_w=mw/nw,
                             flip=int(mc>mw),flip_dens=int(mc/nc>mw/nw),
                             logodds=float(np.log((mc+1e-9)/(mw+1e-9)))))
    out[name]=rows
    nar=[r for r in rows if r["stem"] in {x["stem"] for x in cases_narrow}]
    print(f"{name}: wide n={len(rows)} flip(mass)={sum(r['flip'] for r in rows)} flip(density)={sum(r['flip_dens'] for r in rows)} "
          f"mean_logodds={np.mean([r['logodds'] for r in rows]):.3f} mean_mw={np.mean([r['mw'] for r in rows]):.3f} mean_mc={np.mean([r['mc'] for r in rows]):.3f}",flush=True)
    print(f"   narrow8: flip={sum(r['flip'] for r in nar)} flip_dens={sum(r['flip_dens'] for r in nar)} mean_logodds={np.mean([r['logodds'] for r in nar]):.3f}",flush=True)
json.dump({"cases_wide":cases_wide,"cases_narrow":cases_narrow,"out":out},open("octave_graded.json","w"),indent=1)
# paired comparison MERT vs BTunsup on wide set
a=np.array([r["logodds"] for r in out["MERT"]]); b=np.array([r["logodds"] for r in out["BTunsup"]]); s=np.array([r["logodds"] for r in out["BTsup"]])
from scipy.stats import wilcoxon
print("MERT-BTunsup logodds delta mean",float((a-b).mean()),"wins",int((a>b).sum()),"/",len(a),"wilcoxon p",wilcoxon(a,b).pvalue)
print("BTsup-BTunsup delta mean",float((s-b).mean()),"wins",int((s>b).sum()))
