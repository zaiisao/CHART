"""P2: MINIMAL-MODEL ABLATION LADDER on Dirac input.

Build the simplest bar pointer that shares VBPM's structure, then re-add VBPM pieces one
at a time and report FREE-RUN beat_F after each. The first rung that collapses is the culprit.

Rungs
  A  oracle phi0 + oracle constant phidot, deterministic advance, no meter latent,
     decoder = (3+K)->128->2 MLP trained on oracle z.           [architecture ceiling]
  B1 + learned INITIAL PHASE from h (init head), oracle tempo
  B2 + learned TEMPO from h (init head), oracle phi0
  B3 + both learned                                             [(i) learned tempo]
  C  + stochastic wrapped-Cauchy phase in the recon rollout      [(ii) stochastic phase]
  D  + meter latent (Gumbel) + bar gate                          [(iii) meter]
  E  + full KL/ELBO terms (posterior means still pinned to oracle)[(iv) objective]
  F  + amortized posterior instead of oracle latents             [(v) = the real VBPM]

Two supervision regimes for the learned init scalars:
  'sup'   : direct regression to the oracle (what a learned head COULD do)
  'recon' : gradients only through the reconstruction/ELBO (what VBPM actually does)
"""
import sys, math, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_debug")
from common import (load, est_meter, oracle_const_phase, oracle_pw_phase, dirac_h,
                    targets, FPS, TWO_PI, H_DIM)
from vbpm.evaluate import beats_from_barphase, downbeats_from_barphase, beats_from_activation, \
                          metronome, f_measure
from vbpm.distributions import (sample_wrapped_cauchy, kl_wrapped_cauchy, kl_log_normal,
                                gumbel_softmax, kl_categorical)

DEV = "cuda:1"
K = 4
CROP = 256
BS = 16
STEPS = 1200
MAXF = 1600
N_EVAL = 30

train = load("train"); ev = load("eval", N_EVAL)


# ---------------------------------------------------------------- oracle bundle
def bundle(song, start, n):
    ph, phidot, _ = oracle_const_phase(song, n, start)
    m = est_meter(song["beats"], song["downs"])
    b, db = targets(song, start, n)
    return dict(ph=ph, logtempo=math.log(phidot), m=m, b=b, db=db)


class Mini(nn.Module):
    """Simplest bar pointer sharing VBPM's structure."""
    def __init__(self, h_dim=H_DIM, hidden=128, k=K, init_pool="mean"):
        super().__init__()
        self.k = k; self.init_pool = init_pool
        self.gru = nn.GRU(h_dim, hidden, batch_first=True, bidirectional=True)
        self.ctx = nn.Linear(2 * hidden, hidden)
        self.init_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(),
                                       nn.Linear(hidden, 2 + 2 + k))  # u,v | lv_mu,lv_logsig | meter
        self.rho_head = nn.Linear(hidden, 1)
        self.decoder = nn.Sequential(nn.Linear(3 + k, 128), nn.Tanh(), nn.Linear(128, 2))
        # posterior branch (rung F only)
        self.post_gru = nn.GRU(h_dim + 1, hidden, batch_first=True, bidirectional=True)
        self.post_ctx = nn.Linear(2 * hidden, hidden)
        self.post_head = nn.Sequential(nn.Linear(hidden + 3 + k, hidden), nn.Tanh(),
                                       nn.Linear(hidden, 2 + 2 + k + 1))
        self.z0 = nn.Parameter(torch.zeros(3 + k))
        # tempo bias initialised at the physically-correct scale for 120bpm 4/4 @50fps
        self.lv_bias = nn.Parameter(torch.tensor(math.log(0.5 * TWO_PI / FPS)))

    def enc(self, h):
        o, _ = self.gru(h); return torch.tanh(self.ctx(o))

    def enc_post(self, h, b):
        o, _ = self.post_gru(torch.cat([h, b.unsqueeze(-1)], -1)); return torch.tanh(self.post_ctx(o))

    def init_params(self, c):
        pooled = c.mean(1) if self.init_pool == "mean" else c[:, 0]
        v = self.init_head(pooled)
        phi0 = torch.atan2(v[:, 1], v[:, 0]) % TWO_PI
        lv_mu = v[:, 2] + self.lv_bias
        lv_sig = F.softplus(v[:, 3]) + 1e-3
        return phi0, lv_mu, lv_sig, v[:, 4:]

    def rho(self, c):
        return torch.sigmoid(self.rho_head(c).squeeze(-1)) * (1 - 1e-4)


def zfeat(phi, logt, meter):
    return torch.cat([torch.cos(phi)[..., None], torch.sin(phi)[..., None],
                      logt[..., None].clamp(-12, 6), meter], -1)


# ------------------------------------------------------------------ rung config
CFG = {
    #            phase src      tempo src   meter src  stoch  kl   post
    "A":       dict(phase="oracle", tempo="oracle", meter="oracle", stoch=False, kl=False, post=False),
    "B1_sup":  dict(phase="learn",  tempo="oracle", meter="oracle", stoch=False, kl=False, post=False, sup=True),
    "B2_sup":  dict(phase="oracle", tempo="learn",  meter="oracle", stoch=False, kl=False, post=False, sup=True),
    "B3_sup":  dict(phase="learn",  tempo="learn",  meter="oracle", stoch=False, kl=False, post=False, sup=True),
    "B3_recon":dict(phase="learn",  tempo="learn",  meter="oracle", stoch=False, kl=False, post=False, sup=False),
    "C_sup":   dict(phase="learn",  tempo="learn",  meter="oracle", stoch=True,  kl=False, post=False, sup=True),
    "D_sup":   dict(phase="learn",  tempo="learn",  meter="learn",  stoch=True,  kl=False, post=False, sup=True),
    "E_sup":   dict(phase="learn",  tempo="learn",  meter="learn",  stoch=True,  kl=True,  post=False, sup=True),
    "E_recon": dict(phase="learn",  tempo="learn",  meter="learn",  stoch=True,  kl=True,  post=False, sup=False),
    "F":       dict(phase="learn",  tempo="learn",  meter="learn",  stoch=True,  kl=True,  post=True,  sup=False),
}


def rollout(model, h, b, orc, cfg, temp=0.5):
    """Returns (recon_loss, kl, aux_sup_loss, diag). orc = dict of oracle tensors."""
    B, T, _ = h.shape
    c = model.enc(h)
    phi0_hat, lv_mu, lv_sig, meter_logits = model.init_params(c)
    aux = h.new_zeros(())
    if cfg.get("sup"):
        aux = aux + (1 - torch.cos(phi0_hat - orc["phi0"])).mean() \
                  + ((lv_mu - orc["logtempo"]) ** 2).mean() * 10.0
        if cfg["meter"] == "learn":
            aux = aux + F.cross_entropy(meter_logits, orc["m_idx"])

    logt = orc["logtempo"] if cfg["tempo"] == "oracle" else lv_mu
    if cfg["meter"] == "oracle":
        meter = orc["m_onehot"]
    else:
        meter = gumbel_softmax(meter_logits, temp)

    kl = h.new_zeros(B)
    if cfg["post"]:
        pc = model.enc_post(h, b)

    if cfg["phase"] == "oracle" and not cfg["stoch"]:
        phi = orc["ph"]                                       # [B,T]
        z = zfeat(phi, logt[:, None].expand(-1, T), meter[:, None].expand(-1, T, -1))
    else:
        phis, zs = [], []
        phi_prev = phi0_hat if cfg["phase"] == "learn" else orc["ph"][:, 0]
        meter_t = meter
        z_prev = zfeat(phi_prev, logt, meter_t)
        for t in range(T):
            if t == 0:
                phi = phi_prev
            else:
                adv = (phi_prev + torch.exp(logt.clamp(-12, 6)))
                cross = (adv >= TWO_PI)
                p_mu = adv % TWO_PI
                if cfg["stoch"]:
                    rho_p = model.rho(c[:, t])
                    if cfg["post"]:
                        pv = model.post_head(torch.cat([pc[:, t], z_prev], -1))
                        q_mu = torch.atan2(pv[:, 1], pv[:, 0]) % TWO_PI
                        q_rho = torch.sigmoid(pv[:, 4 + K]) * (1 - 1e-4)
                    else:
                        q_mu = orc["ph"][:, t] if cfg.get("kl") else p_mu
                        q_rho = torch.full_like(rho_p, 0.99)
                    phi = sample_wrapped_cauchy(q_mu, q_rho) if cfg.get("kl") or cfg["post"] \
                          else sample_wrapped_cauchy(p_mu, rho_p)
                    if cfg["kl"]:
                        kl = kl + kl_wrapped_cauchy(q_mu, q_rho, p_mu, rho_p)
                else:
                    phi = p_mu
                if cfg["meter"] == "learn":
                    meter_t = torch.where(cross[:, None], gumbel_softmax(meter_logits, temp), meter_t)
            zs.append(zfeat(phi, logt, meter_t)); phis.append(phi)
            z_prev = zs[-1]; phi_prev = phi
        z = torch.stack(zs, 1)
    if cfg["kl"]:   # level KL (t=1 only, as in the real model's init head) + meter KL
        kl = kl + kl_log_normal(lv_mu, lv_sig, torch.zeros_like(lv_mu),
                                torch.full_like(lv_sig, 1.0))
    out = model.decoder(z)
    rb = F.binary_cross_entropy_with_logits(out[..., 0], orc["b"], reduction="none").sum(1)
    rd = F.binary_cross_entropy_with_logits(out[..., 1], orc["db"], reduction="none").sum(1)
    diag = dict(recon_b=float(rb.mean()), kl=float(kl.mean()),
                lt_err=float((lv_mu - orc["logtempo"]).abs().mean()),
                phi_err=float(((phi0_hat - orc["phi0"] + math.pi) % TWO_PI - math.pi).abs().mean()))
    return (rb + rd).mean(), kl.mean(), aux, diag


@torch.no_grad()
def free_eval(model, cfg, songs):
    """Deploy path, EXACTLY the shape of vbpm.elbo.free_run's mean chain: constant-tempo ramp."""
    model.eval()
    rows = []
    for s in songs:
        T = min(s["T"], MAXF)
        h = torch.from_numpy(dirac_h(s, 0, T)).unsqueeze(0).to(DEV)
        c = model.enc(h)
        phi0_hat, lv_mu, _, ml = model.init_params(c)
        _, phidot_true, phi0_true = oracle_const_phase(s, T)
        if cfg["phase"] == "oracle": phi0_hat = torch.tensor([phi0_true], device=DEV)
        if cfg["tempo"] == "oracle": lv_mu = torch.tensor([math.log(phidot_true)], device=DEV)
        ph = ((phi0_hat[0] + torch.arange(T, device=DEV) * torch.exp(lv_mu[0])) % TWO_PI).cpu().numpy()
        ref = s["beats"][s["beats"] < T / FPS]; dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 2: continue
        m_true = est_meter(ref, dref)
        m_use = m_true if cfg["meter"] == "oracle" else max(2, min(int(ml.argmax(-1)[0]) + 1, 4))
        rows.append(dict(
            beat=f_measure(ref, beats_from_barphase(ph, m_use, FPS)),
            db=(f_measure(dref, downbeats_from_barphase(ph, FPS)) if len(dref) >= 2 else np.nan),
            lt_err=float(lv_mu[0]) - math.log(phidot_true),
            m_use=m_use, m_true=m_true))
    model.train()
    agg = lambda k: float(np.nanmean([r[k] for r in rows]))
    return dict(beat_F=agg("beat"), db_F=agg("db"),
                lt_abs_err=float(np.mean([abs(r["lt_err"]) for r in rows])),
                lt_bias=float(np.mean([r["lt_err"] for r in rows])),
                meter_acc=float(np.mean([r["m_use"] == r["m_true"] for r in rows])))


def run(name, seed=0, steps=STEPS, init_pool="mean"):
    cfg = CFG[name]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = Mini(init_pool=init_pool).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    t0 = time.time(); last = {}
    for step in range(1, steps + 1):
        temp = 1.0 + (0.3 - 1.0) * step / steps
        beta = min(1.0, step / (steps // 2))
        H, Bb, Db, PH, LT, PHI0, MI = [], [], [], [], [], [], []
        for _ in range(BS):
            s = train[rng.integers(len(train))]
            if s["T"] <= CROP: continue
            st = int(rng.integers(0, s["T"] - CROP))
            o = bundle(s, st, CROP)
            H.append(dirac_h(s, st, CROP, rng)); Bb.append(o["b"]); Db.append(o["db"])
            PH.append(o["ph"]); LT.append(o["logtempo"]); PHI0.append(o["ph"][0]); MI.append(o["m"] - 1)
        tt = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(DEV)
        orc = dict(ph=tt(PH), logtempo=tt(LT), phi0=tt(PHI0), b=tt(Bb), db=tt(Db),
                   m_idx=torch.tensor(MI, device=DEV),
                   m_onehot=F.one_hot(torch.tensor(MI, device=DEV), K).float())
        h = tt(H)
        opt.zero_grad()
        rec, kl, aux, diag = rollout(model, h, orc["b"], orc, cfg, temp)
        loss = rec + (beta * kl if cfg["kl"] else 0.0) + aux * 100.0
        if not torch.isfinite(loss): print(f"  !! NaN at step {step}"); break
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        last = diag
    r = free_eval(model, cfg, ev)
    r.update(train_recon_b=last.get("recon_b", float("nan")), sec=time.time() - t0)
    return r


if __name__ == "__main__":
    order = ["A", "B1_sup", "B2_sup", "B3_sup", "B3_recon", "C_sup", "D_sup", "E_sup", "E_recon", "F"]
    only = sys.argv[1:] or order
    res = {}
    for nm in only:
        r = run(nm)
        res[nm] = r
        print(f"{nm:10s} beat_F={r['beat_F']:.3f} db_F={r['db_F']:.3f} "
              f"|logtempo_err|={r['lt_abs_err']:.4f} bias={r['lt_bias']:+.3f} "
              f"meter_acc={r['meter_acc']:.2f} recon_b={r['train_recon_b']:.1f} ({r['sec']:.0f}s)",
              flush=True)
    print(json.dumps(res, indent=1))
