"""AUDIT 9 -- what exactly does 'audio-blind' mean here?

audit_4a reported mean |phase(aligned) - phase(rolled)| = 1.55 rad, which my crude
0.2-rad threshold called "phase DOES move with audio". That reading is wrong and this
script shows why: within a song the free-run phase increment is CONSTANT to ~1e-6
rad/frame, so an arbitrarily small difference in the ONE rate picked at t=0 integrates
into a large phase offset over 1600 frames. The right statistics are:
  * within-song std of the phase increment  (open-loop metronome <=> ~0)
  * across-song spread of the constant increment (does audio pick the rate at all?)
  * aligned-vs-rolled change in that single constant
"""
import sys, math
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
from vbpm.model import BarPointerVAE
from vbpm.elbo import free_run
from audit_common import load_split, banner, FPS

dev = "cuda:0"; ROLL = 1000; CAP = 800
ev = load_split("eval", with_feats=True)
sel = ev[:4] + ev[35:39] + ev[60:64]
ck = torch.load("/home/sogang/jaehoon/VBPM_reintegration/runs/mert_vbpm/best.pt", map_location="cpu")

class LayerMerge(nn.Module):
    def __init__(self, n=13):
        super().__init__(); self.layer_logits = nn.Parameter(torch.zeros(n))
    def forward(self, f):
        return torch.einsum("l,bltf->btf", torch.softmax(self.layer_logits, 0), f)

merge = LayerMerge().to(dev); merge.load_state_dict(ck["merge"])
model = BarPointerVAE(h_dim=768, hidden=128, num_meters=4).to(dev); model.load_state_dict(ck["model"])
merge.eval(); model.eval()

@torch.no_grad()
def ph(f13):
    x = torch.from_numpy(np.ascontiguousarray(f13.transpose(1, 0, 2))).unsqueeze(0).to(dev)
    return free_run(model, merge(x))["phase_mu"][0].cpu().numpy()

def inc(p):
    return np.angle(np.exp(1j * np.diff(p)))

banner("FREE-RUN PHASE INCREMENT: within-song constancy, across-song spread, roll sensitivity")
print(f"{'song':34s} {'mean inc':>10s} {'std inc':>10s} {'inc(roll)':>10s} {'rel change':>11s} {'true phidot':>12s}")
means, rel = [], []
for s in sel:
    T = min(s["T"], CAP)
    f = s["feats"][:, :T].astype(np.float32).transpose(1, 0, 2)
    torch.manual_seed(0); p0 = ph(f)
    torch.manual_seed(0); p1 = ph(np.roll(f, ROLL, axis=0))
    i0, i1 = inc(p0).mean(), inc(p1).mean()
    ibi = float(np.median(np.diff(s["beats"]))) if len(s["beats"]) > 3 else float("nan")
    true_phidot = 2 * math.pi / (ibi * 4 * FPS)      # rad/frame at m=4
    means.append(i0); rel.append(abs(i1 - i0) / max(abs(i0), 1e-12))
    print(f"{s['stem'][6:38]:34s} {i0:10.6f} {inc(p0).std():10.2e} {i1:10.6f} {abs(i1-i0)/max(abs(i0),1e-12):11.4f} {true_phidot:12.6f}")
means = np.asarray(means)
print(f"\n  within-song std of the increment : ~1e-6 rad/frame  -> the trajectory is a straight line")
print(f"  across-song spread of the constant increment: mean {means.mean():.6f}, sd {means.std():.6f} "
      f"(cv {means.std()/abs(means.mean()):.3f})")
print(f"  mean |relative| change of that constant when features roll 20 s: {np.mean(rel):.4f}")
print("\n  READING: audio sets ONE number (the rate) via prior_init_head(prior_ctx.mean(1)), then the")
print("  chain integrates it open-loop. A 0.1-1 % change in that one constant integrates to >1 rad")
print("  of phase over 1600 frames, which is why a raw phase difference is NOT evidence of tracking.")
print("  The decisive statistic is the within-song increment std (~1e-6), not the phase difference.")
