"""QUICK non-adversarial check: sweep bar phase through BOTH decoders on the trained models.
If the emission's output is flat in phase, obs_contrast=1.000 is real and the emission is
phase-blind. If the beat decoder is ALSO flat, then rec_beat<base was never phase information."""
import sys, math, torch, numpy as np
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0,"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
import variant_b as VB
dev="cuda:0"; TWO_PI=2*math.pi
for tag,hdim in [("ii_bern",2),("i_bern",768)]:
    p=f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms/arm_i_{tag}.pt"
    ck=torch.load(p,map_location=dev,weights_only=False)
    m=VB.BarPointerVAE_B(h_dim=hdim,hidden=128,num_meters=4,obs_dim=2,obs_type="bern").to(dev)
    m.load_state_dict(ck["model"],strict=False); m.eval()
    N=180
    phi=torch.linspace(0,TWO_PI,N,device=dev)
    lt=torch.full((N,),-2.7,device=dev)                 # plausible tempo
    mt=torch.zeros(N,4,device=dev); mt[:,3]=1.0          # meter=4 one-hot
    zf=m.z_features(mt,phi,lt)
    with torch.no_grad():
        obs=m.h_dec(zf)          # observation emission
        beat=m.decoder(zf)       # beat/downbeat decoder
    print(f"===== {tag} =====")
    for nm,out in (("OBS  emission h_dec",obs),("BEAT decoder      ",beat)):
        r=[(float(out[:,c].max()-out[:,c].min()), float(out[:,c].std())) for c in range(out.shape[1])]
        print(f"  {nm}: per-channel  range={[f'{x[0]:.4f}' for x in r]}  std={[f'{x[1]:.4f}' for x in r]}")
    # how much do the cos/sin (phase) input weights matter vs the rest?
    W_obs=m.h_dec[0].weight.detach(); W_dec=m.decoder[0].weight.detach()
    for nm,W in (("h_dec",W_obs),("decoder",W_dec)):
        ph=W[:,:2].abs().mean().item(); tempo=W[:,2].abs().mean().item(); met=W[:,3:].abs().mean().item()
        print(f"  {nm} input-weight magnitude: phase(cos,sin)={ph:.4f}  log_tempo={tempo:.4f}  meter={met:.4f}")
