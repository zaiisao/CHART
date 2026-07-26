"""How informative is the learned observation likelihood p_theta(h_t|z_t) about bar phase?

For each eval song and each frame t we evaluate log p(o_t | phi) on a grid of phase values
(log_tempo and meter held at their typical values) and report:
  * contrast    : per-frame std over phi of log p  (nats). ~0 => the filter weights are flat
                  and the particle filter degenerates back to an open-loop prior rollout.
  * phase_corr  : circular correlation between argmax_phi log p(o_t|phi) and the TRUE bar
                  phase (from GT downbeats + beats). This is the honest measure of whether the
                  generative observation model actually locates the bar pointer in the audio.
"""
import sys, math, argparse, json
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
import numpy as np, torch, torch.nn as nn

from vbpm_fix.variant_b import BarPointerVAE_B, dirac_obs
from vbpm_fix.common import load_split, dirac_h, FPS
from vbpm.evaluate import _estimate_meter

DEV = "cuda:0"
NGRID = 64


def true_bar_phase(song, T, fps=FPS):
    """Bar phase at each frame by linear interpolation between GT downbeats."""
    d = song["downs"]
    d = d[d < T / fps]
    if len(d) < 2:
        return None
    ph = np.full(T, np.nan)
    for i in range(len(d) - 1):
        a, b = int(round(d[i] * fps)), int(round(d[i + 1] * fps))
        if b <= a or b > T:
            continue
        ph[a:b] = np.linspace(0, 2 * math.pi, b - a, endpoint=False)
    return ph


@torch.no_grad()
def sharpness(model, obs, m, log_tempo=-2.66):
    T = obs.shape[0]
    grid = torch.linspace(0, 2 * math.pi, NGRID + 1, device=DEV)[:NGRID]
    mo = torch.zeros(NGRID, model.K, device=DEV); mo[:, min(m, model.K) - 1] = 1.0
    zf = torch.cat([torch.cos(grid)[:, None], torch.sin(grid)[:, None],
                    torch.full((NGRID, 1), log_tempo, device=DEV), mo], 1)
    lp = []
    for t0 in range(0, T, 256):
        o = obs[t0:t0 + 256]                                   # [n,obs_dim]
        n = o.shape[0]
        z = zf.unsqueeze(0).expand(n, -1, -1).reshape(-1, zf.shape[1])
        oo = o.unsqueeze(1).expand(-1, NGRID, -1).reshape(-1, o.shape[1])
        lp.append(model.obs_logp(z, oo).reshape(n, NGRID))
    lp = torch.cat(lp, 0)                                      # [T,NGRID]
    return lp.cpu().numpy(), grid.cpu().numpy()


def circ_corr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 10: return float("nan")
    am = math.atan2(np.sin(a).mean(), np.cos(a).mean())
    bm = math.atan2(np.sin(b).mean(), np.cos(b).mean())
    num = (np.sin(a - am) * np.sin(b - bm)).sum()
    den = math.sqrt((np.sin(a - am) ** 2).sum() * (np.sin(b - bm) ** 2).sum())
    return float(num / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dirac", "mert"], required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--max_frames", type=int, default=1600)
    a = ap.parse_args()

    if a.mode == "dirac":
        ev = load_split("eval", a.n)
        model = BarPointerVAE_B(h_dim=8, hidden=128, num_meters=4, obs_dim=2, obs_type="bern").to(DEV)
        model.load_state_dict(torch.load(
            "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/varB_dirac.pt", map_location=DEV))
        model.eval()
        get_obs = lambda s, T: dirac_obs(torch.from_numpy(
            dirac_h(s["beats"], s["downs"], 0, T, rng=np.random.default_rng(0))).unsqueeze(0).to(DEV))[0]
    else:
        sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
        from vbpm_fix.run_mert import FixedProj, LayerMerge, fit_pca, song_feats, OBS_DIM
        tr = load_split("train", with_feats=True)
        mu, comps = fit_pca(tr, np.random.default_rng(0))
        proj = FixedProj(mu, comps).to(DEV)
        ev = load_split("eval", a.n, with_feats=True)
        ck = torch.load("/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/varB_mert.pt", map_location=DEV)
        model = BarPointerVAE_B(h_dim=768, hidden=128, num_meters=4,
                                obs_dim=OBS_DIM, obs_type="gauss").to(DEV)
        model.load_state_dict(ck["model"]); model.eval()
        get_obs = lambda s, T: proj(song_feats(s, T))[0]

    contrasts, corrs = [], []
    for s in ev:
        T = min(s.get("T", s["feats"].shape[1] if "feats" in s else 0), a.max_frames)
        ref = s["beats"][s["beats"] < T / FPS]; dref = s["downs"][s["downs"] < T / FPS]
        if len(ref) < 2 or len(dref) < 2: continue
        m = _estimate_meter(ref, dref)
        obs = get_obs(s, T)
        lp, grid = sharpness(model, obs, m)
        contrasts.append(float(lp.std(1).mean()))
        est_ph = grid[lp.argmax(1)]
        tp = true_bar_phase(s, T)
        if tp is not None:
            corrs.append(circ_corr(est_ph, tp))
    print(json.dumps({"mode": a.mode, "n": len(contrasts),
                      "obs_logp_phase_contrast_nats": float(np.mean(contrasts)),
                      "circ_corr_argmax_vs_true_barphase": float(np.nanmean(corrs))}, indent=2))


if __name__ == "__main__":
    main()
