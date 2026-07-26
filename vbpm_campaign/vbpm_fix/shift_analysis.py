"""Deeper SHIFT-TEST analysis than a single max().

Why: max_t |circ diff| over a 1600-frame trajectory conflates two very different things
  (a) the deploy path RESPONDING to the audio (what we want to measure), and
  (b) an open-loop metronome whose ONE sampled initial tempo changed by 1e-4 because
      prior_init_head(mean(prior_ctx)) saw a slightly different h -- integrated over 1600
      frames that alone reaches pi.
So we report, per deploy path:
  max_absdiff        : the literal statistic requested
  mean_absdiff       : mean over the trajectory  (pi/2 = 1.571 for UNRELATED phases)
  mean_absdiff_first200 : drift-free early window
  TRACKING TEST      : mean |circdiff(phase_shift[t], phase_orig[t-25])|   (follows input -> small)
                  vs   mean |circdiff(phase_shift[t], phase_orig[t])|      (ignores input -> small)
                  A path that TRACKS the audio has  track << notrack.
                  A BLIND path has  track >> notrack  (it kept the old timing).
"""
import sys, json, glob, math
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
import numpy as np, torch
from vbpm.model import BarPointerVAE
from vbpm.elbo import free_run
from model_ab import BarPointerVAE_AB
from elbo_ab import free_run_ab, particle_filter
import common as C

DEV="cuda:0"; SHIFT=25; T=1600; NS=8; SEED=1234

def paths_for(variant):
    return ["baseline_freerun"] if variant=="baseline" else ["A_freerun","noA_freerun","B_filter","AB_filter"]

@torch.no_grad()
def run(kind, model, h):
    torch.manual_seed(SEED)
    if kind=="baseline_freerun": return free_run(model,h)["phase_mu"][0].float().cpu().numpy()
    if kind=="A_freerun":  return free_run_ab(model,h,use_corr=True)["phase_mu"][0].float().cpu().numpy()
    if kind=="noA_freerun":return free_run_ab(model,h,use_corr=False)["phase_mu"][0].float().cpu().numpy()
    return particle_filter(model,h,K=500,use_corr=(kind=="AB_filter"),
                           temper=1.0)["phase_path"][0].float().cpu().numpy()

def main(tags):
    songs=C.load_split("eval",cap=NS,with_feats=False)
    out={}
    for tag in tags:
        d=f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/runs/{tag}"
        cfg=json.load(open(f"{d}/result.json"))["args"]; variant=cfg["variant"]
        model=(BarPointerVAE(h_dim=8,hidden=cfg["hidden"],num_meters=4) if variant=="baseline"
               else BarPointerVAE_AB(h_dim=8,hidden=cfg["hidden"],num_meters=4,
                                     max_phase_corr=cfg.get("max_phase_corr",0.30))).to(DEV)
        model.load_state_dict(torch.load(f"{d}/final.pt",map_location=DEV)["model"]); model.eval()
        res={k:{"max":[],"mean":[],"mean200":[],"track":[],"notrack":[]} for k in paths_for(variant)}
        for s in songs:
            n=min(s["T"],T)
            h0=torch.from_numpy(C.dirac_h(s["beats"],s["downs"],0,n,0,np.random.default_rng(0))).unsqueeze(0).to(DEV)
            h1=torch.from_numpy(C.dirac_h(s["beats"],s["downs"],0,n,SHIFT,np.random.default_rng(0))).unsqueeze(0).to(DEV)
            for k in res:
                p0=run(k,model,h0); p1=run(k,model,h1)
                dd=C.circ_absdiff(p0,p1)
                res[k]["max"].append(float(dd.max())); res[k]["mean"].append(float(dd.mean()))
                res[k]["mean200"].append(float(dd[:200].mean()))
                res[k]["track"].append(float(C.circ_absdiff(p1[SHIFT:],p0[:-SHIFT]).mean()))
                res[k]["notrack"].append(float(dd.mean()))
        print(f"\n##### {tag}   (unrelated-phase reference: mean|circdiff| = pi/2 = 1.571)")
        print(f"{'deploy path':22s} {'max':>7s} {'mean':>7s} {'mean200':>8s} | {'TRACK(shifted)':>14s} {'NOTRACK(orig)':>14s}  verdict")
        for k,v in res.items():
            tr=np.mean(v["track"]); nt=np.mean(v["notrack"])
            verdict = "TRACKS AUDIO" if tr<0.6*nt else ("BLIND" if tr>1.3*nt else "ambiguous")
            print(f"{k:22s} {np.mean(v['max']):7.3f} {np.mean(v['mean']):7.3f} {np.mean(v['mean200']):8.3f} | "
                  f"{tr:14.3f} {nt:14.3f}  {verdict}")
        out[tag]={k:{kk:float(np.mean(vv)) for kk,vv in v.items()} for k,v in res.items()}
    json.dump(out,open("/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/runs/shift_analysis.json","w"),indent=1)

if __name__=="__main__": main(sys.argv[1:])
