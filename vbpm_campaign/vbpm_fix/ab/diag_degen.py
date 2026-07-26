"""Degeneracy audit: does the deploy pointer ACTUALLY ADVANCE, or is it parked?"""
import sys,math
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration"); sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/ab")
import numpy as np, torch, json
from model_ab import BarPointerVAE_AB
from elbo_ab import free_run_ab, particle_filter
import common as C
DEV="cuda:0"; T=1600
out={}
for tag in sys.argv[1:]:
    d=f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/ab/runs/{tag}"
    cfg=json.load(open(f"{d}/result.json"))["args"]
    m=BarPointerVAE_AB(h_dim=8,hidden=cfg["hidden"],num_meters=4,
                       max_phase_corr=cfg.get("max_phase_corr",0.30)).to(DEV)
    m.load_state_dict(torch.load(f"{d}/final.pt",map_location=DEV)["model"]); m.eval()
    songs=C.load_split("eval",cap=8,with_feats=False)
    print(f"\n### {tag}  obs_w={cfg['obs_weight']} max_corr={cfg.get('max_phase_corr',0.30)}  [8 songs, DIRAC oracle]")
    print(f"{'path':12s} {'beat_F':>7s} {'db_F':>6s} {'bars_adv':>9s} {'bars_TRUE':>10s} {'n_wrap':>7s} {'phi_rng':>8s}")
    for kind in ["A_freerun","noA_freerun","B_filter","AB_filter"]:
        rows=[]
        for s in songs:
            n=min(s["T"],T)
            h=torch.from_numpy(C.dirac_h(s["beats"],s["downs"],0,n,0,np.random.default_rng(0))).unsqueeze(0).to(DEV)
            torch.manual_seed(1234)
            if kind=="A_freerun":   p=free_run_ab(m,h,use_corr=True)["phase_mu"][0]
            elif kind=="noA_freerun":p=free_run_ab(m,h,use_corr=False)["phase_mu"][0]
            else: p=particle_filter(m,h,K=500,use_corr=(kind=="AB_filter"),temper=1.0)["phase_path"][0]
            rows.append(C.score_phase(p.float().cpu().numpy(),s,n))
        a=C.aggregate(rows); out[f"{tag}/{kind}"]=a
        print(f"{kind:12s} {a['beat_F']:7.3f} {a['db_F']:6.3f} {a['bars_advanced']:9.2f} {a['bars_true']:10.2f} {a['n_wrap']:7.1f} {a['phi_range']:8.3f}")
json.dump(out,open("/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/ab/runs/degen_audit.json","w"),indent=1)
