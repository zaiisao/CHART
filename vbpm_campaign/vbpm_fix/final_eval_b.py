"""FINAL, CONTROL-CHECKED evaluation of VARIANT B, following vbpm_fix/YARDSTICK.md.

Everything the yardstick demands is reported:
  * ALL 79 eval songs (not the 30/30-ballroom subset), plus the eval[:30] subset for
    comparability with the previously quoted baselines.
  * beat_F, downbeat_F AND n_est/n_true.
  * floors under the SAME protocol: 120-BPM metronome, and a blind constant-spacing grid
    matched to the variant's OWN emission density (an over-emitting estimator scores well
    above the metronome floor for free).
  * the perfect-open-loop reference (oracle tempo + oracle start phase), which an entirely
    audio-blind path can reach.
  * the TIME-ROLL leak/tracking control (features slide +1000 frames, labels stay put,
    identical seed) -- the only test that shows evidence actually reaches the state.
"""
import sys, json, math, argparse
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
import numpy as np, torch

from vbpm.elbo import free_run
from vbpm.evaluate import beats_from_barphase, downbeats_from_barphase, metronome, f_measure, _estimate_meter
from vbpm_fix.variant_b import BarPointerVAE_B, dirac_obs, particle_filter
from vbpm_fix.common import load_split, dirac_h, score_phase, smooth_phase, agg, ratio, FPS

DEV = "cuda:0"


def blind_grid_floor(songs, cap, density, seed=0):
    """Constant-spacing grid at `density` x the true beat density, random start phase."""
    rng = np.random.default_rng(seed); out = []
    for s in songs:
        T = min(s.get("T", 0) or s["feats"].shape[1], cap)
        ref = s["beats"][s["beats"] < T / FPS]
        if len(ref) < 2: continue
        n = max(int(round(len(ref) * density)), 2)
        step = (T / FPS) / n
        out.append(f_measure(ref, np.arange(n) * step + rng.random() * step))
    return float(np.mean(out))


def perfect_open_loop(songs, cap):
    """Oracle tempo + oracle start phase, zero audio afterwards (the audio-blind ceiling)."""
    out = []
    for s in songs:
        T = min(s.get("T", 0) or s["feats"].shape[1], cap)
        ref = s["beats"][s["beats"] < T / FPS]
        if len(ref) < 2: continue
        ibi = np.median(np.diff(ref))
        out.append(f_measure(ref, np.arange(ref[0], T / FPS, ibi)))
    return float(np.mean(out))


def trajectories(phase_fn, songs, cap):
    """Run the deploy path ONCE per song and cache the trajectory (PF is the expensive part)."""
    out = []
    for i, s in enumerate(songs):
        T = min(s.get("T", 0) or s["feats"].shape[1], cap)
        ref = s["beats"][s["beats"] < T / FPS]
        if len(ref) < 2: continue
        ph = phase_fn(s, T, i)
        if ph is None: continue
        out.append((s, T, np.asarray(ph)[:T]))
    return out


def score(trajs, smooth=0):
    return [score_phase(ph, s, T, smooth=smooth) for s, T, ph in trajs]


def evaluate(phase_fn, songs, cap, smooth=0):
    return score(trajectories(phase_fn, songs, cap), smooth)


def line(tag, rows):
    return (f"{tag:42s} beat_F={agg(rows,'beat_F'):.3f} db_F={agg(rows,'db_F'):.3f} "
            f"n_est/n_true={ratio(rows):.3f} (N={len([r for r in rows if r])})")


def rec(rows):
    return {"beat_F": agg(rows, "beat_F"), "db_F": agg(rows, "db_F"), "n_ratio": ratio(rows),
            "N": len([r for r in rows if r])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dirac", "mert"], required=True)
    ap.add_argument("--cap", type=int, default=1600)
    ap.add_argument("--K", type=int, default=400)
    ap.add_argument("--roll", type=int, default=1000)
    ap.add_argument("--n_roll", type=int, default=25)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or f"/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/final_{a.mode}.json"

    if a.mode == "dirac":
        ev = load_split("eval")
        model = BarPointerVAE_B(h_dim=8, hidden=128, num_meters=4, obs_dim=2, obs_type="bern").to(DEV)
        model.load_state_dict(torch.load(
            a.ckpt or "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/varB_dirac.pt", map_location=DEV))
        model.eval()

        def get_h(s, T, roll=0):
            h = dirac_h(s["beats"], s["downs"], 0, T, rng=np.random.default_rng(0))
            if roll: h = np.roll(h, roll, axis=0)
            return torch.from_numpy(h).unsqueeze(0).to(DEV)

        def pf_phase(s, T, i, roll=0):
            torch.manual_seed(1234 + i)
            h = get_h(s, T, roll)
            return particle_filter(model, h, dirac_obs(h), K=a.K)["phase_mean"].numpy()

        def free_phase(s, T, i, roll=0):
            torch.manual_seed(1234 + i)
            return free_run(model, get_h(s, T, roll))["phase_mu"][0, :T].cpu().numpy()
    else:
        from vbpm_fix.run_mert import FixedProj, LayerMerge, song_feats, OBS_DIM
        ck = torch.load(a.ckpt or "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/varB_mert.pt",
                        map_location=DEV)
        assert "pca_comps" in ck, "checkpoint predates the pinned-PCA fix"
        proj = FixedProj(ck["pca_mean"], ck["pca_comps"]).to(DEV)
        merge = LayerMerge().to(DEV); merge.load_state_dict(ck["merge"]); merge.eval()
        model = BarPointerVAE_B(h_dim=768, hidden=128, num_meters=4,
                                obs_dim=OBS_DIM, obs_type="gauss").to(DEV)
        model.load_state_dict(ck["model"]); model.eval()
        ev = load_split("eval", with_feats=True)

        def get_f(s, T, roll=0):
            f = torch.from_numpy(s["feats"][:, :T, :].astype(np.float32)).unsqueeze(0)
            if roll: f = torch.roll(f, roll, dims=2)
            return f.to(DEV)

        def pf_phase(s, T, i, roll=0):
            torch.manual_seed(1234 + i)
            f = get_f(s, T, roll)
            return particle_filter(model, merge(f), proj(f), K=a.K)["phase_mean"].numpy()

        def free_phase(s, T, i, roll=0):
            torch.manual_seed(1234 + i)
            return free_run(model, merge(get_f(s, T, roll)))["phase_mu"][0, :T].cpu().numpy()

    res = {"mode": a.mode, "cap": a.cap, "K": a.K}
    with torch.no_grad():
        for name, songs in [("ALL79", ev), ("eval[:30]", ev[:30])]:
            print(f"\n===== {name}  (cap {a.cap} frames = {a.cap/FPS:.0f} s) =====", flush=True)
            fr = evaluate(free_phase, songs, a.cap)
            tj = trajectories(pf_phase, songs, a.cap)
            pf = score(tj, 0); pfs = score(tj, 5); pfs9 = score(tj, 9)
            metro = float(np.mean([r["metronome"] for r in fr if r]))
            print(line("free_run (open loop, same model)", fr), flush=True)
            print(line(f"PARTICLE FILTER (K={a.K})", pf), flush=True)
            print(line("PARTICLE FILTER + smooth5 read-out", pfs), flush=True)
            print(line("PARTICLE FILTER + smooth9 read-out", pfs9), flush=True)
            print(f"{'FLOOR 120-BPM metronome':42s} beat_F={metro:.3f}", flush=True)
            for tag, rr in [("PF", pf), ("PF+sm5", pfs), ("PF+sm9", pfs9)]:
                d = ratio(rr)
                print(f"{'FLOOR blind grid @ ' + tag + ' density %.2fx' % d:42s} "
                      f"beat_F={blind_grid_floor(songs, a.cap, d):.3f}", flush=True)
            print(f"{'REF perfect open loop (oracle tempo+phase)':42s} "
                  f"beat_F={perfect_open_loop(songs, a.cap):.3f}", flush=True)
            res[name] = {"free_run": rec(fr), "pf": rec(pf), "pf_smooth5": rec(pfs),
                         "pf_smooth9": rec(pfs9),
                         "blind_grid_at_pfsmooth9_density": blind_grid_floor(songs, a.cap, ratio(pfs9)),
                         "metronome": metro,
                         "blind_grid_at_pf_density": blind_grid_floor(songs, a.cap, ratio(pf)),
                         "blind_grid_at_pfsmooth_density": blind_grid_floor(songs, a.cap, ratio(pfs)),
                         "perfect_open_loop": perfect_open_loop(songs, a.cap)}

        # ---------- TIME-ROLL CONTROL (yardstick section F) ----------
        print(f"\n===== TIME-ROLL CONTROL (+{a.roll} frames = {a.roll/FPS:.0f} s, labels fixed) =====", flush=True)
        sub = ev[:a.n_roll]
        for tag, fn in [("PF", pf_phase), ("free_run", free_phase)]:
            al = evaluate(lambda s, T, i: fn(s, T, i, 0), sub, a.cap)
            ro = evaluate(lambda s, T, i: fn(s, T, i, a.roll), sub, a.cap)
            metro = float(np.mean([r["metronome"] for r in al if r]))
            drop = agg(al, "beat_F") - agg(ro, "beat_F")
            print(f"  {tag:10s} aligned={agg(al,'beat_F'):.3f} rolled={agg(ro,'beat_F'):.3f} "
                  f"metro={metro:.3f} drop={drop:+.3f}", flush=True)
            res[f"roll_{tag}"] = {"aligned": agg(al, "beat_F"), "rolled": agg(ro, "beat_F"),
                                  "metronome": metro, "drop": drop}

    json.dump(res, open(out_path, "w"), indent=2)
    print("\nWROTE " + out_path, flush=True)


if __name__ == "__main__":
    main()
