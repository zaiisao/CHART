"""ENCODER PROBE B: train the REAL model on Dirac h (same recipe as probe_dirac.py, cuda:3),
then interrogate the amortized posterior q_phi(z|h,b):
  (c) per-latent KL over training -- which latents are alive/dead
  (b) MI/shift test: shift the Dirac beat grid by delta seconds; does q's phase mean shift too?
  + q phase vs ORACLE bar phase, q log_tempo vs TRUE phidot, q meter vs TRUE meter
"""
import sys, math, time, json
import numpy as np, torch
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_debug")
from enc_common import *  # noqa
from vbpm.model import BarPointerVAE
from vbpm.elbo import strict_elbo, free_run
from vbpm.evaluate import (beats_from_barphase, downbeats_from_barphase, beats_from_activation,
                           metronome, f_measure, _estimate_meter)

dev = "cuda:3"
STEPS, WARM, BS, FR = 1000, 600, 16, 256
train = load("train"); ev = load("eval", 30)
print(f"train {len(train)} eval {len(ev)}", flush=True)

torch.manual_seed(0); rng = np.random.default_rng(0)
model = BarPointerVAE(h_dim=H_DIM, hidden=128, num_meters=4).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)


def batch(rng, bs=BS, fr=FR):
    hs, bs_, ds = [], [], []
    for _ in range(bs):
        s = train[rng.integers(len(train))]
        if s["T"] <= fr:
            continue
        st = int(rng.integers(0, s["T"] - fr))
        hs.append(dirac_h(s["beats"], s["downs"], st, fr, rng))
        b, d = targets(s["beats"], s["downs"], st, fr)
        bs_.append(b); ds.append(d)
    return (torch.from_numpy(np.stack(hs)).to(dev), torch.from_numpy(np.stack(bs_)).to(dev),
            torch.from_numpy(np.stack(ds)).to(dev))


@torch.no_grad()
def free_eval(songs, max_frames=1600):
    model.eval(); acc = {"beat_phase": [], "downbeat_phase": [], "decoder": [], "metronome": []}
    for s in songs:
        T = min(s["T"], max_frames)
        h = torch.from_numpy(dirac_h(s["beats"], s["downs"], 0, T)).unsqueeze(0).to(dev)
        out = free_run(model, h)
        pm = out["phase_mu"][0, :T].cpu().numpy(); dec = out["decoder_prob"][0, :T].cpu().numpy()
        ref = s["beats"][s["beats"] < T / FPS]; dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 2:
            continue
        m = _estimate_meter(ref, dref)
        acc["beat_phase"].append(f_measure(ref, beats_from_barphase(pm, m, FPS)))
        if len(dref) >= 2:
            acc["downbeat_phase"].append(f_measure(dref, downbeats_from_barphase(pm, FPS)))
        acc["decoder"].append(f_measure(ref, beats_from_activation(dec, FPS)))
        acc["metronome"].append(f_measure(ref, metronome(T, FPS)))
    model.train()
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


hist = []
t0 = time.time()
for step in range(1, STEPS + 1):
    beta = min(1.0, step / WARM); temp = 1.0 + (0.3 - 1.0) * min(step / STEPS, 1.0)
    h, b, d = batch(rng)
    opt.zero_grad()
    loss, info = strict_elbo(model, h, b, d, temperature=temp, beta=beta)
    if not torch.isfinite(loss):
        print("NaN@", step, flush=True); break
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    # gradient reaching the ENCODER specifically
    genc = sum(float(p.grad.norm()) ** 2 for n, p in model.named_parameters()
               if p.grad is not None and n.startswith(("post_gru", "post_ctx", "post_head"))) ** 0.5
    opt.step()
    if step % 50 == 0:
        hist.append(dict(step=step, **{k: info[k] for k in
                    ["recon_beat", "recon_db", "kl_phase", "kl_level", "kl_dev", "kl_meter", "n_cross"]},
                    gnorm=float(gn), g_enc=genc))
        print(f"s{step:4d} b={beta:.2f} rec_b={info['recon_beat']:7.2f} rec_db={info['recon_db']:7.2f} "
              f"KL[ph={info['kl_phase']:8.2f} lv={info['kl_level']:8.2f} dv={info['kl_dev']:6.2f} "
              f"m={info['kl_meter']:5.2f}] ncross={info['n_cross']:6.1f} |g|={float(gn):7.2f} "
              f"|g_enc|={genc:7.3f} {step/(time.time()-t0):.1f}it/s", flush=True)
    if step in (300, 600, STEPS):
        r = free_eval(ev)
        print(f"  [FREE-RUN s{step}] beat_F={r['beat_phase']:.3f} db_F={r['downbeat_phase']:.3f} "
              f"dec_F={r['decoder']:.3f} metro={r['metronome']:.3f}", flush=True)

torch.save(model.state_dict(), "/home/sogang/jaehoon/VBPM_reintegration/vbpm_debug/enc_trained.pt")
json.dump(hist, open("/home/sogang/jaehoon/VBPM_reintegration/vbpm_debug/enc_hist.json", "w"))
print("saved.", flush=True)
