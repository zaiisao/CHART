"""DESIGN-TIME DIAGNOSIS: what are the LEARNED prior transition's parameters, numerically?
Reads the same checkpoint the decisive 2x2 used (vbpm_arms/arm_i_ii_bern.pt) and reports
the per-frame prior kernel parameters over eval songs. No training, no anchoring."""
import sys, math, json
import numpy as np, torch
for p in ("/home/sogang/jaehoon/VBPM_reintegration","/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix",
          "/home/sogang/jaehoon/VBPM_reintegration/vbpm_final","/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"):
    sys.path.insert(0,p)
import variant_b as VB
from emission import load_act, load_split
dev="cuda:0"
ck=torch.load("/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms/arm_i_ii_bern.pt",map_location="cpu")
model=VB.BarPointerVAE_B(h_dim=2,hidden=ck["config"]["hidden"],num_meters=4,obs_dim=2,obs_type="bern").to(dev)
model.load_state_dict(ck["model"]); model.eval()
print("a_lv (level OU) =%.5f   tempo dof=%.3f  level_offset=%.3f"%(float(model.level_ar()),float(model.tempo_dof()),float(model.level_offset)))
ev=load_split("eval"); ae=load_act("eval")
RHO=[];SLV=[];SDV=[];ADV=[]
with torch.no_grad():
    for s in ev[:30]:
        a=ae.get(s["stem"]); 
        if a is None: continue
        h=torch.from_numpy(a[:min(len(a),s["T"])]).unsqueeze(0).to(dev)
        ctx=model.encode_prior(h)[0]
        RHO.append(model.prior_phase_conc(ctx).cpu().numpy())
        SLV.append(model.prior_level_scale(ctx).cpu().numpy())
        SDV.append(model.prior_dev_scale(ctx).cpu().numpy())
        ADV.append(model.prior_dev_coef(ctx).cpu().numpy())
def st(n,x):
    x=np.concatenate(x); 
    print("  %-18s mean=%.5f  sd=%.5f  p5=%.5f p50=%.5f p95=%.5f"%(n,x.mean(),x.std(),*np.percentile(x,[5,50,95])))
    return x
print("LEARNED per-frame prior params (30 eval songs):")
rho=st("phase rho",RHO); slv=st("sigma_level",SLV); sdv=st("sigma_dev",SDV); adv=st("a_dev",ADV)
g=-np.log(np.clip(rho,1e-9,1-1e-9))
print("  -> wrapped-Cauchy gamma: mean=%.4f median=%.4f (PHYSICAL gamma=0.000555)"%(g.mean(),np.median(g)))
phidot=0.0626
print("  -> implied frac_neg floor (1/pi)atan(gamma/phidot): mean=%.3f"%(np.arctan(g/phidot)/math.pi).mean())
print("  -> per-song SD of mean rho across songs = %.5f (audio-adaptivity of the learned kernel)"%np.std([r.mean() for r in RHO]))
print("  -> per-song SD of mean sigma_level      = %.5f"%np.std([r.mean() for r in SLV]))
# tempo random-walk: how far does the level wander in 1000 frames?
print("  -> level RW sd over 1000 frames (no OU) = %.3f nats ; with OU a=%.3f stationary sd=%.3f"%(
    slv.mean()*math.sqrt(1000), float(model.level_ar()), slv.mean()/math.sqrt(max(1-float(model.level_ar())**2,1e-6))))
print("  -> dev stationary sd = %.3f nats"%(sdv.mean()/math.sqrt(max(1-(adv.mean()**2),1e-6))))
