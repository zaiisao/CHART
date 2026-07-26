"""AUDIT 4a -- TIME-ROLL LEAK CONTROL on the real-MERT setting.

Object under test: the only existing trained MERT VBPM checkpoint,
runs/mert_vbpm/best.pt (train_mert.py, "DONE best free-run beat_F=0.340").

Also (i) proves the control HAS POWER by rolling the ORACLE ideal-phase read-out,
and (ii) re-measures the audio-blindness of free_run directly: how far does the
free-run phase trajectory move when the features are rolled by 20 s?
"""
import sys, math
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
from vbpm.model import BarPointerVAE
from vbpm.elbo import free_run
from audit_common import (load_split, ideal_barphase, truncate, score_phase, agg, ratio,
                          banner, metronome, f_measure, FPS)
from roll_control import roll_control

dev = "cuda:0"
ROLL = 1000            # 20 s
CAP = 1600             # same frame cap train_mert.py evaluated under

ev = load_split("eval", with_feats=True)
print(f"eval {len(ev)} songs", flush=True)

# ---------------------------------------------------------------- power check
banner("POWER CHECK: does the roll control fire on a KNOWN-GOOD signal?")
rows_a, rows_r = [], []
for s in ev:
    T, ref, dref = truncate(s, None)
    if len(ref) < 2 or len(dref) < 2: continue
    ph = ideal_barphase(dref, T, mode="extrap")
    rows_a.append(dict(**score_phase(ph, ref, dref, T), metronome_F=f_measure(ref, metronome(T, FPS))))
    rows_r.append(dict(**score_phase(np.roll(ph, ROLL), ref, dref, T), metronome_F=f_measure(ref, metronome(T, FPS))))
aa, ar = agg(rows_a, ["beat_F", "metronome_F"]), agg(rows_r, ["beat_F", "metronome_F"])
print(f"  ORACLE ideal phase  aligned beat_F={aa['beat_F']:.3f}   rolled(+{ROLL}) beat_F={ar['beat_F']:.3f}   "
      f"metronome={aa['metronome_F']:.3f}")
print(f"  -> control has power: {'YES' if aa['beat_F'] - ar['beat_F'] > 0.4 else 'NO -- CONTROL IS BLIND'}")

# ---------------------------------------------------------------- the trained VBPM
banner("TRAINED MERT VBPM (runs/mert_vbpm/best.pt) UNDER THE ROLL CONTROL")
ck = torch.load("/home/sogang/jaehoon/VBPM_reintegration/runs/mert_vbpm/best.pt", map_location="cpu")

class LayerMerge(nn.Module):
    def __init__(self, n=13):
        super().__init__(); self.layer_logits = nn.Parameter(torch.zeros(n))
    def forward(self, feats):
        return torch.einsum("l,bltf->btf", torch.softmax(self.layer_logits, 0), feats)

merge = LayerMerge().to(dev); merge.load_state_dict(ck["merge"])
model = BarPointerVAE(h_dim=768, hidden=128, num_meters=4).to(dev); model.load_state_dict(ck["model"])
merge.eval(); model.eval()

@torch.no_grad()
def phase_fn(f):                                   # f [T,13*?]: we pass raw [13,T,768] flattened below
    raise RuntimeError

@torch.no_grad()
def vbpm_phase(feat_13):                           # feat_13 [T,13,768] (time-major so roll axis=0 works)
    x = torch.from_numpy(np.ascontiguousarray(feat_13.transpose(1, 0, 2))).unsqueeze(0).to(dev)  # [1,13,T,768]
    h = merge(x)
    out = free_run(model, h)
    return out["phase_mu"][0].cpu().numpy()

feat_fn = lambda s, T: s["feats"][:, :T].astype(np.float32).transpose(1, 0, 2)   # [T,13,768]
res = roll_control(vbpm_phase, feat_fn, ev, roll=ROLL, cap=CAP, label="mert_vbpm best.pt")

# ---------------------------------------------------------------- audio-blindness re-measure
banner("AUDIO-BLINDNESS OF free_run (direct measurement, same seed both arms)")
d_phase, d_lt, d_inc = [], [], []
with torch.no_grad():
    for s in ev[:20]:
        T = min(s["T"], CAP)
        f = s["feats"][:, :T].astype(np.float32).transpose(1, 0, 2)
        torch.manual_seed(0); p0 = vbpm_phase(f)
        torch.manual_seed(0); p1 = vbpm_phase(np.roll(f, ROLL, axis=0))
        d = np.abs(np.angle(np.exp(1j * (p0 - p1))))
        d_phase.append(d.mean())
        inc = np.angle(np.exp(1j * np.diff(p0)))
        d_inc.append(inc.std())
print(f"  mean |phase_mu(aligned) - phase_mu(rolled)| = {np.mean(d_phase):.4f} rad "
      f"({100*np.mean(d_phase)/(2*math.pi):.2f} % of a bar)")
print(f"  std of the free-run phase increment within a song = {np.mean(d_inc):.3e} rad/frame")
print(f"  -> {'OPEN-LOOP METRONOME (audio-blind) CONFIRMED' if np.mean(d_phase) < 0.2 else 'phase DOES move with audio'}")
