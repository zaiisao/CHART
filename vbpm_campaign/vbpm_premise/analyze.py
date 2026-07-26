"""Premise-4 read-out: response curves, per-dataset split, paired bootstrap CIs, lambda_needed."""
import json, math, sys
import numpy as np

OUT='/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise'
S2=json.load(open(f'{OUT}/sweep2_main2.json'))
KL=json.load(open(f'{OUT}/kl_to_physical.json'))
DS=('ballroom','beatles','hainsworth')

def boot(dv, B=20000, seed=0):
    r=np.random.default_rng(seed); dv=np.asarray(dv)
    m=np.array([r.choice(dv,len(dv),replace=True).mean() for _ in range(B//200)])
    idx=r.integers(0,len(dv),(B,len(dv)))
    m=dv[idx].mean(1)
    return float(dv.mean()), float(np.percentile(m,2.5)), float(np.percentile(m,97.5))

for op in ('pub','cell'):
    rows=[e for e in S2 if e['op']==op]
    print('='*118)
    print(f'OPERATING POINT {op}  (alpha={0.25 if op=="pub" else 1.0}, K={600 if op=="pub" else 300}) '
          f' | emission fit on 147 TRAIN songs, scored on 79 EVAL (fold-0) songs')
    for axis in ('rev_p','cauchy_rho','gauss_sphi','sigma_lt'):
        ax=[e for e in rows if e['axis']==axis]
        base=ax[0]
        bs=base['per_song_beatF']
        print(f'\n--- axis {axis} ---   (baseline = first row; paired bootstrap CI over 79 songs)')
        print(f'{"x":>9} {"KL/frame":>9} {"beat_F":>7} {"blind":>6} {"MARGIN":>7} {"db_F":>6} '
              f'{"frac_neg":>8} {"logZ/fr":>8} {"dF_paired":>10} {"95%CI":>17} '
              + ' '.join(f'{d[:4]:>6}' for d in DS))
        for e in ax:
            klv=KL[axis].get(str(e['x']), float('nan'))
            d=e['pf_meter_path']
            stems=[k for k in bs if k in e['per_song_beatF']]
            dv=np.array([e['per_song_beatF'][k]-bs[k] for k in stems])
            m,lo,hi=boot(dv)
            byds=' '.join(f"{e['pf_meter_path_by_ds'].get(dd,{}).get('beat_F',float('nan')):6.3f}" for dd in DS)
            print(f"{e['x']:>9} {klv:9.3g} {d['beat_F']:7.4f} {d['blind_best']:6.3f} "
                  f"{d['margin']:+7.4f} {d['db_F']:6.3f} {d['frac_neg']:8.3f} {d['logZpf']:+8.4f} "
                  f"{m:+10.4f} [{lo:+6.3f},{hi:+6.3f}] {byds}")
        # lambda_needed to move the optimum from the sloppiest point to each x
        sl=ax[-1]
        print(f'  lambda_needed = [logZ(sloppiest) - logZ(x)] / [KL(sloppiest) - KL(x)]   '
              f'(nats/frame ratio; >0 means anchoring must outbid the data term)')
        klS=KL[axis].get(str(sl['x']),float("nan")); lzS=sl['pf_meter_path']['logZpf']
        for e in ax:
            klv=KL[axis].get(str(e['x']),float('nan'))
            if not np.isfinite(klv) or abs(klS-klv)<1e-9: continue
            lam=(lzS-e['pf_meter_path']['logZpf'])/(klS-klv)
            print(f"    x={e['x']:<8} dlogZ={lzS-e['pf_meter_path']['logZpf']:+.4f}  "
                  f"dKL={klS-klv:8.3f}  lambda_needed={lam:+.5f}  "
                  f"beat_F {sl['pf_meter_path']['beat_F']:.3f} -> {e['pf_meter_path']['beat_F']:.3f}")
