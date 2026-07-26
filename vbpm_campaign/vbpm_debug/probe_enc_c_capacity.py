"""ENCODER PROBE C: capacity / resolution of the amortized posterior, isolated from the ELBO.

C1  frozen (random-init) encoder: is beat/phase info even PRESENT in post_ctx?
    linear read-out post_ctx -> (a) beat indicator, (b) cos/sin of oracle bar phase.
C2  SUPERVISED fit of the REAL encoder path (post_gru + post_ctx + post_head -> atan2(u,v))
    to the ORACLE bar phase, on Dirac input. If this fits, the encoder HAS the capacity and
    the 0.063-rad/frame resolution -> the failure is the objective, not the amortization.
C3  fine-resolution check: per-frame increment of the fitted q phase vs true phidot.
C4  what the wrapped-Cauchy SAMPLE does to a perfect q mean (rho at init = 0.5).
"""
import sys, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_debug")
from enc_common import *  # noqa
from vbpm.model import BarPointerVAE
from vbpm.distributions import sample_wrapped_cauchy

dev = "cuda:2"
FR = 256
torch.manual_seed(0); rng = np.random.default_rng(0)
train = load("train", 60); ev = load("eval", 20)


def make(songs, n_per=4, fr=FR, rng=rng):
    H, B, D, PH, OK, PD = [], [], [], [], [], []
    for s in songs:
        if s["T"] <= fr or len(s["downs"]) < 3:
            continue
        for _ in range(n_per):
            st = int(rng.integers(0, s["T"] - fr))
            r = oracle_barphase(s["beats"], s["downs"], st, fr)
            ph, ok = r
            H.append(dirac_h(s["beats"], s["downs"], st, fr, rng))
            b, d = targets(s["beats"], s["downs"], st, fr)
            B.append(b); D.append(d); PH.append(ph); OK.append(ok); PD.append(true_phidot(s))
    t = lambda x, dt=torch.float32: torch.tensor(np.stack(x), dtype=dt, device=dev)
    return t(H), t(B), t(D), t(PH), torch.tensor(np.stack(OK), device=dev), t(PD)


Htr, Btr, Dtr, PHtr, OKtr, PDtr = make(train)
Hev, Bev, Dev, PHev, OKev, PDev = make(ev, n_per=2)
print(f"C: train {Htr.shape} eval {Hev.shape}", flush=True)

model = BarPointerVAE(h_dim=H_DIM, hidden=128, num_meters=4).to(dev)

print("=" * 78); print("C1  FROZEN random encoder: is the info in post_ctx?"); print("=" * 78)
with torch.no_grad():
    ctx_tr = model.encode_posterior(Htr, Btr)
    ctx_ev = model.encode_posterior(Hev, Bev)
print(f"  post_ctx train {tuple(ctx_tr.shape)} std {ctx_tr.std():.4f}")
for name, tgt_tr, tgt_ev, kind in [
        ("beat indicator", Btr, Bev, "bce"),
        ("oracle (cos,sin) bar phase",
         torch.stack([torch.cos(PHtr), torch.sin(PHtr)], -1),
         torch.stack([torch.cos(PHev), torch.sin(PHev)], -1), "mse")]:
    lin = nn.Linear(128, 1 if kind == "bce" else 2).to(dev)
    op = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for i in range(1500):
        op.zero_grad()
        o = lin(ctx_tr)
        if kind == "bce":
            l = F.binary_cross_entropy_with_logits(o.squeeze(-1), tgt_tr)
        else:
            l = ((o - tgt_tr) ** 2)[OKtr].mean()
        l.backward(); op.step()
    with torch.no_grad():
        o = lin(ctx_ev)
        if kind == "bce":
            base = F.binary_cross_entropy(torch.full_like(Bev, float(Btr.mean())), Bev)
            print(f"  {name:28s}: heldout BCE {l.item():.4f}/{F.binary_cross_entropy_with_logits(o.squeeze(-1),Bev).item():.4f}"
                  f"  base-rate {base.item():.4f}")
        else:
            pred = torch.atan2(o[..., 1], o[..., 0])
            err = torch.abs((pred - PHev + math.pi) % TWO_PI - math.pi)[OKev]
            print(f"  {name:28s}: heldout mean|circ err| {err.mean():.3f} rad (chance {math.pi/2:.3f})")

print(); print("=" * 78)
print("C2  SUPERVISED fit of the REAL encoder path -> ORACLE bar phase")
print("=" * 78)
torch.manual_seed(0)
m2 = BarPointerVAE(h_dim=H_DIM, hidden=128, num_meters=4).to(dev)
params = list(m2.post_gru.parameters()) + list(m2.post_ctx.parameters()) + list(m2.post_head.parameters())
op = torch.optim.AdamW(params, lr=3e-4)
z0 = m2.z0.unsqueeze(0)


def q_phase_teacher(m, H, B, PH):
    """Run the head with TEACHER-FORCED z_prev built from the ORACLE phase/tempo (the
    best case for the recursion), and return q phase mu [B,T] + level/dev mus."""
    ctx = m.encode_posterior(H, B)
    Bn, T, _ = H.shape
    outs, lv, dv = [], [], []
    for t in range(T):
        if t == 0:
            zp = m.z0.unsqueeze(0).expand(Bn, -1)
        else:
            zp = torch.cat([torch.cos(PH[:, t - 1:t]), torch.sin(PH[:, t - 1:t]),
                            torch.log(PDtr_b), F.one_hot(torch.full((Bn,), 2, device=dev), 4).float()], -1)
        vec = m.post_head(torch.cat([ctx[:, t], zp], -1))
        mm, phmu, phrho, lvmu, lvs, dvmu, dvs = m.unpack(vec)
        outs.append(phmu); lv.append(lvmu); dv.append(dvmu)
    return torch.stack(outs, 1), torch.stack(lv, 1), torch.stack(dv, 1)


PDtr_b = torch.log(PDtr).unsqueeze(-1)
for it in range(400):
    op.zero_grad()
    qmu, qlv, qdv = q_phase_teacher(m2, Htr, Btr, PHtr)
    loss = (1.0 - torch.cos(qmu - PHtr))[OKtr].mean() \
        + ((qlv + qdv) - torch.log(PDtr).unsqueeze(1)).pow(2).mean()
    loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); op.step()
    if it % 100 == 0 or it == 399:
        with torch.no_grad():
            PDtr_b_save = PDtr_b
            globals()['PDtr_b'] = torch.log(PDev).unsqueeze(-1)
            qe, qle, qde = q_phase_teacher(m2, Hev, Bev, PHev)
            globals()['PDtr_b'] = PDtr_b_save
            err = torch.abs((qe - PHev + math.pi) % TWO_PI - math.pi)[OKev]
            R = torch.abs(torch.exp(1j * ((qe - PHev)[OKev]).to(torch.complex64)).mean())
            lt = (qle + qde).mean(1)
        print(f"  it{it:4d} loss {loss.item():.4f} | HELDOUT mean|err| {err.mean():.3f} rad"
              f"  R {float(R):.3f}  q log_tempo {float(lt.mean()):+.3f} (true {float(torch.log(PDev).mean()):+.3f})",
              flush=True)

print(); print("=" * 78)
print("C3  fine RESOLUTION: per-frame increment of the FITTED q phase")
print("=" * 78)
with torch.no_grad():
    globals()['PDtr_b'] = torch.log(PDev).unsqueeze(-1)
    qe, _, _ = q_phase_teacher(m2, Hev, Bev, PHev)
d = ((qe[:, 1:] - qe[:, :-1] + math.pi) % TWO_PI - math.pi).cpu().numpy()
tru = ((PHev[:, 1:] - PHev[:, :-1] + math.pi) % TWO_PI - math.pi).cpu().numpy()
print(f"  fitted q d(phase)/frame: mean {d.mean():+.4f} std {d.std():.4f} median|.| {np.median(np.abs(d)):.4f}")
print(f"  true      d(phase)/frame: mean {tru.mean():+.4f} std {tru.std():.4f} median|.| {np.median(np.abs(tru)):.4f}")
print(f"  corr(fitted inc, true inc) = {np.corrcoef(d.ravel(), tru.ravel())[0,1]:.3f}")

print(); print("=" * 78)
print("C4  what the wrapped-Cauchy SAMPLER does to a PERFECT q mean")
print("=" * 78)
mu = PHev.reshape(-1)
for rho in [0.5, 0.8, 0.9, 0.99, 0.999]:
    r = torch.full_like(mu, rho)
    smp = sample_wrapped_cauchy(mu, r)
    e = torch.abs((smp - mu + math.pi) % TWO_PI - math.pi)
    print(f"  rho={rho:<6} gamma={-math.log(rho):.4f}  mean|err| {e.mean():.3f} rad"
          f"  frac|err|>0.5*phidot(0.03) {float((e>0.03).float().mean()):.3f}"
          f"  frac|err|>pi/4 {float((e>math.pi/4).float().mean()):.3f}")
