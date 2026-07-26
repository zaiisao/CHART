"""Honest test: VBPM (latent-only likelihood) on FROZEN MERT features, learnable per-layer
weighting ('little bit of every layer'), KL-warmup, scored on FREE-RUN beat-F (prior-only,
no beats fed) fold-honest (train folds!=0, eval fold 0). Nothing marked, nothing autoencoded.
"""
import sys, glob, json, time, math, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run
from vbpm.evaluate import (beats_from_barphase, downbeats_from_barphase, beats_from_activation,
                           metronome, f_measure, _estimate_meter)

CACHE = "/disk1/jaehoon/vbpm_mert_cache"

class LayerMerge(nn.Module):
    """Learnable softmax over the 13 MERT layers -> [.,T,768]. The 'little bit of every layer'."""
    def __init__(self, n_layers=13):
        super().__init__(); self.layer_logits = nn.Parameter(torch.zeros(n_layers))
    def forward(self, feats):                     # feats [B,13,T,768]
        w = torch.softmax(self.layer_logits, 0)
        return torch.einsum("l,bltf->btf", w, feats)
    def weights(self):
        return torch.softmax(self.layer_logits.detach(), 0).cpu().numpy()

def load_split(split):
    out=[]
    for f in sorted(glob.glob(f"{CACHE}/{split}__*.npz")):
        d=np.load(f, allow_pickle=True)
        out.append(dict(stem=Path(f).stem, feats=d["feats"], beats=np.asarray(d["beats"],float),
                        downs=np.asarray(d["downs"],float), fps=float(d["fps"]), dataset=str(d["dataset"])))
    return out

def targets(beats, downs, start, n, fps):
    b=np.zeros(n,np.float32); db=np.zeros(n,np.float32)
    for t in beats:
        i=int(round(t*fps))-start
        if 0<=i<n: b[i]=1.0
    for t in downs:
        i=int(round(t*fps))-start
        if 0<=i<n: db[i]=1.0
    return b, db

def sample_batch(songs, bs, frames, fps, device, rng):
    fe,bb,dd=[],[],[]
    for _ in range(bs):
        s=songs[rng.integers(len(songs))]; T=s["feats"].shape[1]
        if T<=frames: continue
        st=int(rng.integers(0,T-frames))
        fe.append(torch.from_numpy(s["feats"][:,st:st+frames,:].astype(np.float32)))
        b,db=targets(s["beats"],s["downs"],st,frames,fps); bb.append(torch.from_numpy(b)); dd.append(torch.from_numpy(db))
    return (torch.stack(fe).to(device), torch.stack(bb).to(device), torch.stack(dd).to(device))

@torch.no_grad()
def eval_freerun(merge, model, songs, fps, device, max_frames=1600):
    merge.eval(); model.eval()
    acc={"beat_phase":[], "downbeat_phase":[], "decoder":[], "metronome":[]}
    for s in songs:
        T=min(s["feats"].shape[1], max_frames)
        feats=torch.from_numpy(s["feats"][:,:T,:].astype(np.float32)).unsqueeze(0).to(device)
        h=merge(feats)
        out=free_run(model, h)
        pm=out["phase_mu"][0,:T].cpu().numpy(); dec=out["decoder_prob"][0,:T].cpu().numpy()
        ref=s["beats"][s["beats"]<T/fps]; dref=s["downs"][s["downs"]<T/fps]
        if len(ref)<2: continue
        m=_estimate_meter(ref,dref)
        acc["beat_phase"].append(f_measure(ref, beats_from_barphase(pm,m,fps)))
        if len(dref)>=2: acc["downbeat_phase"].append(f_measure(dref, downbeats_from_barphase(pm,fps)))
        acc["decoder"].append(f_measure(ref, beats_from_activation(dec,fps)))
        acc["metronome"].append(f_measure(ref, metronome(T,fps)))
    merge.train(); model.train()
    return {k:(float(np.mean(v)) if v else float("nan")) for k,v in acc.items()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=2500); ap.add_argument("--warmup",type=int,default=1200)
    ap.add_argument("--bs",type=int,default=16); ap.add_argument("--frames",type=int,default=512)
    ap.add_argument("--hidden",type=int,default=128); ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--eval_every",type=int,default=300); ap.add_argument("--out",default="runs/mert_vbpm")
    ap.add_argument("--device",default="cuda:0")
    a=ap.parse_args()
    dev=torch.device(a.device); rng=np.random.default_rng(0); torch.manual_seed(0)
    Path(a.out).mkdir(parents=True,exist_ok=True)
    train=load_split("train"); ev=load_split("eval")[:30]; fps=train[0]["fps"] if train else 50.0
    print(f"train {len(train)} eval {len(ev)} songs | fps {fps} | frames {a.frames} bs {a.bs} steps {a.steps} warmup {a.warmup}",flush=True)

    merge=LayerMerge().to(dev); model=BarPointerVAE(h_dim=768,hidden=a.hidden,num_meters=4).to(dev)  # latent_only=True
    opt=torch.optim.AdamW(list(merge.parameters())+list(model.parameters()),lr=a.lr)
    mf=open(Path(a.out)/"metrics.jsonl","w"); best=-1; t0=time.time()
    for step in range(1,a.steps+1):
        beta=min(1.0, step/max(a.warmup,1))
        temp=1.0+(0.3-1.0)*min(step/a.steps,1.0)
        feats,b,db=sample_batch(train,a.bs,a.frames,fps,dev,rng)
        h=merge(feats)
        opt.zero_grad(); loss,info=strict_elbo(model,h,b,db,temperature=temp,beta=beta)
        if not torch.isfinite(loss): print("NaN@",step,flush=True); break
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(merge.parameters())+list(model.parameters()),5.0); opt.step()
        if step%50==0 or step==1:
            print(f"s{step:5d} b={beta:.2f} rec_b={info['recon_beat']:6.2f} rec_db={info['recon_db']:6.2f} "
                  f"kl(phi={info['kl_phase']:.2f} lv={info['kl_level']:.2f} dv={info['kl_dev']:.2f} m={info['kl_meter']:.3f}) "
                  f"nu={info['tempo_dof']:.2f} | {step/(time.time()-t0):.1f} it/s",flush=True)
        if step%a.eval_every==0 or step==a.steps:
            r=eval_freerun(merge,model,ev,fps,dev)
            lw=merge.weights()
            print(f"  [EVAL s{step}] FREE-RUN beat_F={r['beat_phase']:.3f} db_F={r['downbeat_phase']:.3f} "
                  f"decoder_F={r['decoder']:.3f} metronome_F={r['metronome']:.3f} | top layers={np.argsort(lw)[::-1][:4]}",flush=True)
            mf.write(json.dumps({"step":step,"beta":beta,**r,"layer_w":lw.tolist(),
                     "rec_b":info["recon_beat"],"kl_phase":info["kl_phase"]})+"\n"); mf.flush()
            if r["beat_phase"]==r["beat_phase"] and r["beat_phase"]>best:
                best=r["beat_phase"]; torch.save({"merge":merge.state_dict(),"model":model.state_dict()},Path(a.out)/"best.pt")
    print(f"DONE best free-run beat_F={best:.3f}",flush=True)

if __name__=="__main__": main()
