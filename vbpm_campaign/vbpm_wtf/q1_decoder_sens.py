"""Q1: sensitivity of BOTH decoders to each z_feat dim (CPU, checkpoint-only)."""
import math, sys
import torch, torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
import variant_b as VB

ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
TWO_PI = 2 * math.pi
NAMES = ["cos_phi", "sin_phi", "log_tempo", "m1", "m2", "m3", "m4"]

for tag, hdim in (("ii_bern", 2), ("i_bern", 768)):
    ck = torch.load(f"{ARMS}/arm_i_{tag}.pt", map_location="cpu")
    cfg = ck["config"]
    model = VB.BarPointerVAE_B(h_dim=hdim, hidden=cfg["hidden"], num_meters=4,
                               obs_dim=2, obs_type="bern")
    model.load_state_dict(ck["model"]); model.eval()
    print("=" * 78); print(f"ARM {tag}   h_dim={hdim}")
    with torch.no_grad():
        n = 360
        ph = torch.linspace(0, TWO_PI * (n - 1) / n, n)
        for lt_val in (-3.55, -2.66, -2.18):
            mt = F.one_hot(torch.full((n,), 3), 4).float()
            lt = torch.full((n,), lt_val)
            zf = model.z_features(mt, ph, lt)
            pb = torch.sigmoid(model.decoder(zf))
            po = torch.sigmoid(model.h_dec(zf))
            print(f"  lt={lt_val:+.2f}  phase sweep -> p(beat) [{pb[:,0].min():.5f},{pb[:,0].max():.5f}] "
                  f"rng={float(pb[:,0].max()-pb[:,0].min()):.6f} | p(down) rng="
                  f"{float(pb[:,1].max()-pb[:,1].min()):.6f} | obs0 [{po[:,0].min():.5f},{po[:,0].max():.5f}] "
                  f"rng={float(po[:,0].max()-po[:,0].min()):.6f} | obs1 rng={float(po[:,1].max()-po[:,1].min()):.6f}")
        m = 200
        for band in ((-3.55, -2.18), (-8.0, 2.0)):
            lt = torch.linspace(band[0], band[1], m)
            mt = F.one_hot(torch.full((m,), 3), 4).float()
            for phv in (0.0, math.pi):
                zf = model.z_features(mt, torch.full((m,), phv), lt)
                pb = torch.sigmoid(model.decoder(zf)); po = torch.sigmoid(model.h_dec(zf))
                print(f"  lt sweep {band} phi={phv:.2f} -> p(beat) [{pb[:,0].min():.5f},{pb[:,0].max():.5f}] "
                      f"rng={float(pb[:,0].max()-pb[:,0].min()):.6f} | obs0 rng={float(po[:,0].max()-po[:,0].min()):.6f}")
        mm = torch.eye(4)
        zf = model.z_features(mm, torch.zeros(4), torch.full((4,), -2.66))
        pb = torch.sigmoid(model.decoder(zf)); po = torch.sigmoid(model.h_dec(zf))
        print("  meter sweep p(beat):", [round(float(x), 5) for x in pb[:, 0]],
              " obs0:", [round(float(x), 5) for x in po[:, 0]])
        for nm, W in (("decoder", model.decoder[0].weight), ("h_dec", model.h_dec[0].weight)):
            v = W.abs().mean(0)
            print(f"  {nm} mean|W| per dim: " + " ".join(f"{a}={float(b):.4f}" for a, b in zip(NAMES, v)))
