import sys, math, numpy as np, torch
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf")
import variant_b as VB
from audit_common import load_split, banner
from p3align_agentP3 import LayerMerge, build_obs_cache, DEV, ARMS
for arm,hd,tg in (("i",768,"i_bern"),("ii",2,"ii_bern")):
    sd=torch.load(f"{ARMS}/arm_i_{tg}.pt",map_location="cpu")
    mg=LayerMerge().to(DEV); mg.load_state_dict(sd["merge"]); mg.eval()
    m=VB.BarPointerVAE_B(h_dim=hd,hidden=128,num_meters=4,obs_dim=2,obs_type="bern").to(DEV)
    m.load_state_dict(sd["model"]); m.eval()
    ev=load_split("eval",with_feats=True,cap=12); oc=build_obs_cache(ev,f"{ARMS}/act_eval.npz")
    rr=[]
    with torch.no_grad():
        for s in ev:
            T=min(s["feats"].shape[1],3000)
            f=torch.from_numpy(np.asarray(s["feats"][:,:T,:],np.float32)).unsqueeze(0).to(DEV)
            o=torch.from_numpy(oc[s["stem"]][:T]).unsqueeze(0).to(DEV)
            h=mg(f) if arm=="i" else o
            ctx=m.encode_prior(h)[0]
            rr.append(m.prior_phase_conc(ctx).cpu().numpy())
            del f,o,h
    rr=np.concatenate(rr)
    # wrapped Cauchy: mean resultant length = rho; sd of one-step phase innovation
    print(f"arm {arm}: DEPLOY prior phase concentration rho  mean={rr.mean():.4f} "
          f"median={np.median(rr):.4f} p95={np.percentile(rr,95):.4f} max={rr.max():.4f}"
          f"   (rho=1 -> deterministic advance, rho=0 -> UNIFORM redraw every frame)")
