"""Mechanism: emission response surface over (phase, log_tempo), and where the PF lives."""
import sys, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
import variant_b as VB
from vbpm.distributions import TWO_PI
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
DEV = "cuda:0"
for arm, hd in (("i_bern", 768), ("ii_bern", 2)):
    sd = torch.load(f"{ARMS}/arm_i_{arm}.pt", map_location="cpu")
    m = VB.BarPointerVAE_B(h_dim=hd, hidden=128, num_meters=4, obs_dim=2, obs_type="bern").to(DEV)
    m.load_state_dict(sd["model"]); m.eval()
    print("=" * 78); print("arm", arm)
    ph = torch.linspace(0, TWO_PI, 64, device=DEV)
    met = F.one_hot(torch.full((64,), 3, device=DEV), 4).float()
    print("  log_tempo |  p(beat) range over phase  |  p(db) range  | max-min log p(o=1) over phase")
    for lt in [-6.0, -4.9, -3.55, -2.86, -2.18, -1.0, -0.5, 0.0, 1.0]:
        z = m.z_features(met, ph, torch.full((64,), lt, device=DEV))
        with torch.no_grad():
            pr = torch.sigmoid(m.h_dec(z))
            lp1 = -F.binary_cross_entropy_with_logits(
                m.h_dec(z), torch.ones(64, 2, device=DEV), reduction="none").sum(-1)
        print(f"   {lt:+6.2f}  |  {pr[:,0].min():.4f}..{pr[:,0].max():.4f}  |  "
              f"{pr[:,1].min():.4f}..{pr[:,1].max():.4f}  |  "
              f"{float(lp1.max()-lp1.min()):.5f} nats")
    # dynamic-range argument: contribution to the h_dec first-layer pre-activation
    W = m.h_dec[0].weight.detach()               # [128,7]
    # training-time log_tempo sd ~2.0, phase features have sd 1/sqrt(2)=0.707
    contrib = W.abs().mean(0) * torch.tensor([0.707, 0.707, 2.0, .5, .5, .5, .5], device=DEV)
    print("  first-layer pre-activation sd contribution per z_feat dim "
          "[cos,sin,logtempo,m1..m4]:")
    print("   ", " ".join(f"{v:.4f}" for v in contrib.tolist()))
