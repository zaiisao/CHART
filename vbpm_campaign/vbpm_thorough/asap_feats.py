"""X2(b) features: audio-blind history + audio(causal/full) features on ASAP,
reusing vbpm_premise/histfeat.hist_feats-equivalent and feats.song_feats VERBATIM,
with row order guaranteed consistent across the three sets (same ks per song)."""
import sys, numpy as np
from multiprocessing import Pool
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough')
from core import prep
from feats import song_feats
from asap_data import build_asap

def hist_rows(d, ks):
    u=d['u']; F=[]
    for k in ks:
        up=u[k-1]
        lag=lambda j: u[k-j] if k-j>=0 else up
        past=u[max(0,k-8):k]; past2=u[max(0,k-4):k]; allp=u[:k]
        de=np.diff(allp) if len(allp)>1 else np.array([0.0])
        F.append([up, up-lag(2), lag(2)-lag(3), lag(3)-lag(4),
                  float(np.mean(past))-up, float(np.mean(past2))-up,
                  float(np.mean(np.abs(np.diff(past)))) if len(past)>2 else 0.0,
                  float(np.mean(np.abs(de))), float(np.std(de)), float(np.median(np.abs(de))),
                  float(np.mean(allp))-up, float(np.std(allp)), float(k),
                  float(d['meter']==2),float(d['meter']==3),float(d['meter']==4)])
    return np.asarray(F,np.float64)

def one_song(d):
    Xf, ksf = song_feats(d, 300, 'full')
    Xc, ksc = song_feats(d, 300, 'causal')
    assert ksf == ksc
    ks = ksf
    if not len(ks): return None
    Xh = hist_rows(d, ks)
    G = dict(u_prev=d['u'][np.array(ks)-1], u=d['u'][np.array(ks)],
             ulo=d['ulo'][np.array(ks)], uhi=d['uhi'][np.array(ks)],
             meter=np.full(len(ks), d['meter']),
             stem=np.array([d['stem']]*len(ks)))
    return Xh, Xc, Xf, G

for sp in ('train','eval'):
    D = prep(build_asap(sp))
    with Pool(16) as p:
        res = [r for r in p.map(one_song, D) if r is not None]
    Xh = np.concatenate([r[0] for r in res],0)
    Xc = np.concatenate([r[1] for r in res],0)
    Xf = np.concatenate([r[2] for r in res],0)
    G  = {k: np.concatenate([r[3][k] for r in res],0) for k in res[0][3]}
    np.savez(f'hist_asap_{sp}.npz', X=Xh, **G)
    np.savez(f'feat_causal_asap_{sp}.npz', X=Xc, **G)
    np.savez(f'feat_full_asap_{sp}.npz', X=Xf, **G)
    print(sp, 'hist', Xh.shape, 'causal', Xc.shape, 'full', Xf.shape, flush=True)
print('FEATS DONE')
