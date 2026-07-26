"""Is n_ratio=2.39 caused by (a) the inferred TEMPO being too fast, or (b) phase JITTER
creating spurious wrap detections? Opposite fixes -- diagnose before grounding."""
import sys, glob, math, json
import numpy as np, torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
import variant_b as VB
from vbpm.evaluate import beats_from_barphase, f_measure, _estimate_meter
CACHE="/disk1/jaehoon/vbpm_mert_cache"; fps=50.0; TWO_PI=2*math.pi; dev="cuda:0"
ck=torch.load("/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/varB_mert.pt",map_location=dev,weights_only=False)
print("checkpoint keys:",list(ck.keys())[:8])
model=VB.BarPointerVAE_B(h_dim=768,hidden=128,num_meters=4,obs_dim=32,obs_type="gauss").to(dev)
sd=ck.get("model",ck); model.load_state_dict(sd,strict=False); model.eval()
merge=torch.nn.Parameter(torch.zeros(13,device=dev))
if "layer_logits" in ck: merge.data=torch.as_tensor(ck["layer_logits"],device=dev)
elif "merge" in ck and "layer_logits" in ck["merge"]: merge.data=ck["merge"]["layer_logits"].to(dev)
proj=VB.MertObsProjector(768,32,seed=0).to(dev)

rows=[]
with torch.no_grad():
    for f in sorted(glob.glob(f"{CACHE}/eval__*.npz"))[:15]:
        d=np.load(f,allow_pickle=True)
        T=min(int(d["feats"].shape[1]),1200)
        feats=torch.from_numpy(d["feats"][:,:T].astype(np.float32)).unsqueeze(0).to(dev)
        w=torch.softmax(merge,0); h=torch.einsum("l,bltf->btf",w,feats)
        obs=proj(h)
        out=VB.particle_filter(model,h,obs,K=300,alpha=1.0)
        ph = out["phase"] if isinstance(out,dict) and "phase" in out else (out[0] if isinstance(out,tuple) else out)
        ph = np.asarray(ph.cpu() if torch.is_tensor(ph) else ph).squeeze()
        beats=np.asarray(d["beats"],float); downs=np.asarray(d["downs"],float)
        ref=beats[beats<T/fps]; dref=downs[downs<T/fps]
        if len(ref)<3: continue
        m=_estimate_meter(ref,dref)
        est=beats_from_barphase(ph,m,fps)
        # INFERRED tempo from the phase trajectory: median positive per-frame advance
        dphi=np.diff(ph); dphi=np.angle(np.exp(1j*dphi))          # wrap-safe increments
        adv=np.median(dphi[dphi>0]) if (dphi>0).any() else np.nan
        inf_bpm=60.0*fps*m*adv/TWO_PI
        true_bpm=60.0/np.median(np.diff(ref))
        # jitter: how much do increments vary (a clean pointer = tiny std)
        jit=np.std(dphi)
        rows.append(dict(inf_bpm=inf_bpm,true_bpm=true_bpm,ratio_bpm=inf_bpm/true_bpm,
                         n_est=len(est),n_true=len(ref),n_ratio=len(est)/len(ref),jitter=jit,
                         F=f_measure(ref,est)))
import statistics as st
print(f"\n{'song':>4} {'infBPM':>7} {'trueBPM':>8} {'BPMratio':>8} {'n_ratio':>8} {'jitter':>7} {'F':>6}")
for i,r in enumerate(rows):
    print(f"{i:>4} {r['inf_bpm']:7.1f} {r['true_bpm']:8.1f} {r['ratio_bpm']:8.2f} {r['n_ratio']:8.2f} {r['jitter']:7.3f} {r['F']:6.3f}")
print(f"\nMEDIAN BPM ratio (inferred/true) = {st.median([r['ratio_bpm'] for r in rows]):.2f}")
print(f"MEDIAN n_ratio (est beats/true)  = {st.median([r['n_ratio'] for r in rows]):.2f}")
print(f"MEDIAN phase-increment jitter     = {st.median([r['jitter'] for r in rows]):.3f} rad")
print("\nDIAGNOSIS:")
br=st.median([r['ratio_bpm'] for r in rows]); nr=st.median([r['n_ratio'] for r in rows])
if br>1.6: print("  -> TEMPO is genuinely too fast: grounding the tempo is the right fix.")
elif nr>1.6 and br<1.4: print("  -> TEMPO IS ~CORRECT but n_ratio high => PHASE JITTER creates spurious wraps.")
else: print("  -> mixed/unclear; inspect per-song rows.")
