"""AUDIT 4a (fast subset) -- same test as audit_4a_timeroll_mert.py on 20 songs,
so a number exists even under GPU contention."""
import sys, math, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
from vbpm.model import BarPointerVAE
from vbpm.elbo import free_run
from audit_common import load_split, banner, FPS
from roll_control import roll_control

dev = "cuda:0"; ROLL = 1000; CAP = 1600; N = 20
ev = load_split("eval", with_feats=True, cap=None)
sel = ev[:10] + ev[40:50]          # 10 ballroom + 10 beatles/hainsworth
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
def vbpm_phase(f13):
    x = torch.from_numpy(np.ascontiguousarray(f13.transpose(1, 0, 2))).unsqueeze(0).to(dev)
    return free_run(model, merge(x))["phase_mu"][0].cpu().numpy()

feat_fn = lambda s, T: s["feats"][:, :T].astype(np.float32).transpose(1, 0, 2)
t0 = time.time()
banner(f"TRAINED MERT VBPM (runs/mert_vbpm/best.pt) -- ROLL CONTROL on {len(sel)} songs, cap {CAP}")
roll_control(vbpm_phase, feat_fn, sel, roll=ROLL, cap=CAP, label="mert_vbpm best.pt")
print(f"  ({time.time()-t0:.0f}s)")

banner("AUDIO-BLINDNESS OF free_run (same seed both arms)")
dp, di = [], []
with torch.no_grad():
    for s in sel:
        T = min(s["T"], CAP)
        f = feat_fn(s, T)
        torch.manual_seed(0); p0 = vbpm_phase(f)
        torch.manual_seed(0); p1 = vbpm_phase(np.roll(f, ROLL, axis=0))
        dp.append(np.abs(np.angle(np.exp(1j * (p0 - p1)))).mean())
        di.append(np.angle(np.exp(1j * np.diff(p0))).std())
print(f"  mean |phase_mu(aligned) - phase_mu(rolled)| = {np.mean(dp):.4f} rad ({100*np.mean(dp)/(2*math.pi):.2f} % of a bar)")
print(f"  std of the free-run phase increment within a song = {np.mean(di):.3e} rad/frame")
print(f"  -> {'OPEN-LOOP METRONOME (audio-blind) CONFIRMED' if np.mean(dp) < 0.2 else 'phase DOES move with audio'}")
