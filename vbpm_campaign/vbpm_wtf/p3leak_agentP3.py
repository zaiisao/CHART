"""PROBE 3 follow-up: is log_tempo a LEAK wire for b, and is it phase or tempo the
decoders read?  Runs on both arms."""
from __future__ import annotations
import argparse, math, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix")
sys.path.insert(0, "/home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf")
import variant_b as VB
from vbpm.distributions import TWO_PI
from vbpm.evaluate import f_measure
from audit_common import load_split, banner, FPS
from p3align_agentP3 import (LayerMerge, posterior_replay, make_crops, build_obs_cache, DEV, ARMS)


def auc(score, label):
    o = np.argsort(score); l = np.asarray(label)[o]
    n1 = l.sum(); n0 = len(l) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.arange(1, len(l) + 1)
    return float((ranks[l > 0.5].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="i")
    ap.add_argument("--n", type=int, default=48); a = ap.parse_args()
    ck = f"{ARMS}/arm_i_{'i_bern' if a.arm=='i' else 'ii_bern'}.pt"
    sd = torch.load(ck, map_location="cpu"); h_dim = 768 if a.arm == "i" else 2
    merge = LayerMerge().to(DEV); merge.load_state_dict(sd["merge"]); merge.eval()
    model = VB.BarPointerVAE_B(h_dim=h_dim, hidden=128, num_meters=4,
                               obs_dim=2, obs_type="bern").to(DEV)
    model.load_state_dict(sd["model"]); model.eval()
    banner(f"LEAK PROBE arm={a.arm}")
    ev = load_split("eval", with_feats=True)
    oev = build_obs_cache(ev, f"{ARMS}/act_eval.npz")
    torch.manual_seed(11); rng = np.random.default_rng(11)
    crops = make_crops(ev, oev, rng, a.n, 256)
    R = {k: [] for k in ("real", "bzero", "bshuf", "auc", "ltF", "cc_lt")}
    for i in range(0, len(crops), 8):
        ch = crops[i:i + 8]
        f = torch.from_numpy(np.stack([c["feats"] for c in ch])).to(DEV)
        b = torch.from_numpy(np.stack([c["b"] for c in ch])).to(DEV)
        d = torch.from_numpy(np.stack([c["d"] for c in ch])).to(DEV)
        o = torch.from_numpy(np.stack([c["obs"] for c in ch])).to(DEV)
        h = merge(f) if a.arm == "i" else o
        for tag, bin_ in (("real", b), ("bzero", torch.zeros_like(b)),
                          ("bshuf", b[:, torch.randperm(b.shape[1], device=DEV)])):
            torch.manual_seed(3)
            r = posterior_replay(model, h, bin_, d)
            lg = model.decoder(r["Z"])
            rb = F.binary_cross_entropy_with_logits(lg[..., 0], b, reduction="none").sum(1)
            R[tag] += rb.tolist()
            if tag == "real":
                lt = r["log_tempo"].cpu().numpy()
                for j, c in enumerate(ch):
                    R["auc"].append(auc(lt[j], c["b"]))
                    # peak-pick log_tempo directly as if it were a beat activation
                    x = lt[j]; thr = x.mean() + 0.8 * x.std()
                    pk = [t for t in range(1, len(x) - 1)
                          if x[t] > thr and x[t] >= x[t - 1] and x[t] > x[t + 1]]
                    est = (np.asarray(pk) + 0.5) / FPS
                    ref = (np.nonzero(c["b"])[0] + 0.5) / FPS
                    R["ltF"].append(f_measure(ref, est) if len(ref) > 1 else float("nan"))
        del f, b, d, o, h
    A = lambda k: float(np.nanmean(R[k]))
    print(f"  rec_beat with TRUE b in posterior : {A('real'):.2f} nats/crop")
    print(f"  rec_beat with b:=0   in posterior : {A('bzero'):.2f} nats/crop")
    print(f"  rec_beat with b SHUFFLED          : {A('bshuf'):.2f} nats/crop")
    print(f"  AUC( log_tempo_t  ->  b_t )       : {A('auc'):.4f}   (0.5 = no info)")
    print(f"  beat F from PEAK-PICKING log_tempo: {A('ltF'):.4f}")


if __name__ == "__main__":
    main()
