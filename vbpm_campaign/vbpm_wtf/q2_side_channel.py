"""Q2: where does the beat decoder's information actually come from?

Runs the REAL elbo_b forward on train crops, keeps Z, then:
  * reports prior/posterior phase concentration rho  (is phase noise?)
  * reports log_tempo statistics + its mutual dependence with the beat target
  * ABLATES each z component and re-measures recon_beat / recon_obs
"""
import math, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms")
import variant_b as VB
from vbpm.distributions import TWO_PI, gumbel_softmax, sample_wrapped_cauchy, sample_student_t
from audit_common import load_split, FPS
from common import targets

DEV = "cuda:0"
ARMS = "/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms"
TAG, HDIM = sys.argv[1] if len(sys.argv) > 1 else "i_bern", None
HDIM = 768 if TAG == "i_bern" else 2


@torch.no_grad()
def rollout(model, h, b, temp=0.3):
    """Same recursion as elbo_b, but return Z + the parameter traces."""
    B, T, _ = h.shape
    post_ctx = model.encode_posterior(h, b)
    prior_ctx = model.encode_prior(h)
    dof = model.tempo_dof()
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
        model.post_head(torch.cat([post_ctx[:, 0], z0], -1)))
    meter = gumbel_softmax(q_m, temp)
    phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
    level = sample_student_t(dof, q_lv_mu, q_lv_s)
    dev = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
    lt = level + dev
    a_lv = model.level_ar(); anchor = level
    Zs, QRHO, PRHO, LT = [], [], [], []
    Zs.append(model.z_features(meter, phi, lt)); QRHO.append(q_ph_rho); LT.append(lt)
    mp, pp, lp, dp, ltp = meter, phi, level, dev, lt
    for t in range(1, T):
        zpf = model.z_features(mp, pp, ltp)
        q_m, q_ph_mu, q_ph_rho, q_lv_mu, q_lv_s, q_dv_mu, q_dv_s = model.unpack(
            model.post_head(torch.cat([post_ctx[:, t], zpf], -1)))
        adv = pp + torch.exp(ltp.clamp(-12, 6))
        cross = (adv >= TWO_PI).float()
        p_rho = model.prior_phase_conc(prior_ctx[:, t])
        phi = sample_wrapped_cauchy(q_ph_mu, q_ph_rho)
        level = sample_student_t(dof, q_lv_mu, q_lv_s)
        dev = q_dv_mu + q_dv_s * torch.randn_like(q_dv_mu)
        lt = level + dev
        meter = torch.where(cross.unsqueeze(-1) > 0.5, gumbel_softmax(q_m, temp), mp)
        Zs.append(model.z_features(meter, phi, lt)); QRHO.append(q_ph_rho)
        PRHO.append(p_rho); LT.append(lt)
        mp, pp, lp, dp, ltp = meter, phi, level, dev, lt
    return (torch.stack(Zs, 1), torch.stack(QRHO, 1), torch.stack(PRHO, 1),
            torch.stack(LT, 1))


def bce(logit, tgt):
    return F.binary_cross_entropy_with_logits(logit, tgt, reduction="none").sum(1).mean()


def main():
    ck = torch.load(f"{ARMS}/arm_i_{TAG}.pt", map_location=DEV)
    cfg = ck["config"]
    model = VB.BarPointerVAE_B(h_dim=HDIM, hidden=cfg["hidden"], num_meters=4,
                               obs_dim=2, obs_type="bern").to(DEV)
    model.load_state_dict(ck["model"]); model.eval()
    merge = None
    if TAG == "i_bern":
        from arm_i import LayerMerge
        merge = LayerMerge().to(DEV); merge.load_state_dict(ck["merge"]); merge.eval()

    train = load_split("train", with_feats=True, cap=40)
    d = np.load(f"{ARMS}/act_train.npz", allow_pickle=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    FR, BS = 256, 16
    fe, bb, dd, oo = [], [], [], []
    while len(fe) < BS:
        s = train[rng.integers(len(train))]
        T = s["feats"].shape[1]
        if T <= FR: continue
        st = int(rng.integers(0, T - FR))
        fe.append(torch.from_numpy(s["feats"][:, st:st+FR, :].astype(np.float32)))
        bt, dt = targets(s["beats"], s["downs"], st, FR)
        bb.append(torch.from_numpy(bt)); dd.append(torch.from_numpy(dt))
        a = np.clip(np.asarray(d[s["stem"] + "|act"], np.float32)[st:st+FR], 1e-4, 1-1e-4)
        oo.append(torch.from_numpy(a))
    f = torch.stack(fe).to(DEV); b = torch.stack(bb).to(DEV)
    db = torch.stack(dd).to(DEV); o = torch.stack(oo).to(DEV)
    h = merge(f) if merge is not None else (o if TAG == "ii_bern" else None)
    if TAG == "ii_bern":
        h = o.clone()
    print(f"h {tuple(h.shape)}  obs {tuple(o.shape)}  beat rate {float(b.mean()):.4f}")

    with torch.no_grad():
        Z, qrho, prho, LT = rollout(model, h, b)
    print(f"\nPHASE CONCENTRATION (wrapped Cauchy rho; 0 = UNIFORM, 1 = Dirac)")
    print(f"  posterior q rho: mean {float(qrho.mean()):.5f}  median {float(qrho.median()):.5f} "
          f"max {float(qrho.max()):.5f}")
    print(f"  prior     p rho: mean {float(prho.mean()):.5f}  median {float(prho.median()):.5f} "
          f"max {float(prho.max()):.5f}")
    print(f"\nLOG-TEMPO (z dim 2)  mean {float(LT.mean()):.3f}  sd {float(LT.std()):.3f} "
          f"min {float(LT.min()):.3f} max {float(LT.max()):.3f}")
    ltb = LT[b > 0.5]; ltn = LT[b < 0.5]
    print(f"  log_tempo at BEAT frames   mean {float(ltb.mean()):+.3f} sd {float(ltb.std()):.3f}")
    print(f"  log_tempo at NONbeat frames mean {float(ltn.mean()):+.3f} sd {float(ltn.std()):.3f}")
    print(f"  point-biserial corr(log_tempo, beat) = "
          f"{float(np.corrcoef(LT.flatten().cpu(), b.flatten().cpu())[0,1]):+.4f}")
    ph = torch.atan2(Z[..., 1], Z[..., 0]) % TWO_PI
    dphi = ((ph[:, 1:] - ph[:, :-1] + math.pi) % TWO_PI) - math.pi
    print(f"  phase increment: mean {float(dphi.mean()):+.4f} sd {float(dphi.std()):.4f} "
          f"frac_neg {float((dphi<0).float().mean()):.3f}   (true rate ~+0.065 rad/frame)")
    # circular concentration of realised phase (is it uniform?)
    R = float(torch.sqrt(torch.cos(ph).mean()**2 + torch.sin(ph).mean()**2))
    print(f"  realised phase circular R = {R:.4f}  (0 = uniform over the circle)")

    print("\nABLATION of recon (per 256-frame crop, nats):")
    base_rate_b = float(F.binary_cross_entropy(b.mean().expand_as(b), b, reduction="none").sum(1).mean())
    base_rate_d = float(F.binary_cross_entropy(db.mean().expand_as(db), db, reduction="none").sum(1).mean())
    def score(Zx, name):
        lg = model.decoder(Zx)
        rb, rd = float(bce(lg[..., 0], b)), float(bce(lg[..., 1], db))
        ro = float(-model.obs_logp(Zx.reshape(-1, 7), o.reshape(-1, 2)).reshape(Z.shape[0], -1).sum(1).mean())
        print(f"  {name:34s} rec_b={rb:7.2f}  rec_db={rd:6.2f}  rec_obs={ro:7.2f}")
    print(f"  {'BASE RATE (constant predictor)':34s} rec_b={base_rate_b:7.2f}  rec_db={base_rate_d:6.2f}")
    score(Z, "FULL z")
    Za = Z.clone(); pr = torch.rand_like(ph) * TWO_PI
    Za[..., 0], Za[..., 1] = torch.cos(pr), torch.sin(pr)
    score(Za, "phase -> UNIFORM RANDOM")
    Zb = Z.clone(); Zb[..., 0], Zb[..., 1] = 1.0, 0.0
    score(Zb, "phase -> CONSTANT 0")
    Zc = Z.clone(); Zc[..., 2] = Z[..., 2].mean()
    score(Zc, "log_tempo -> its global mean")
    Zd = Z.clone(); Zd[..., 2] = Z[..., 2].mean(1, keepdim=True)
    score(Zd, "log_tempo -> per-crop mean")
    Ze = Z.clone(); Ze[..., 2] = Z[..., 2][torch.randperm(Z.shape[0])][:, torch.randperm(Z.shape[1])]
    score(Ze, "log_tempo -> shuffled")
    Zf = Z.clone(); Zf[..., 3:] = 0.25
    score(Zf, "meter -> flat 0.25")
    Zg = Z.clone(); Zg[..., 0], Zg[..., 1] = 1.0, 0.0; Zg[..., 3:] = 0.25
    score(Zg, "ONLY log_tempo (phase+meter const)")


if __name__ == "__main__":
    main()
