"""STRAND X2: ASAP data builder mirroring vbpm_premise/data.py's schema exactly.

Songs = 473 ASAP performances with both beat_this annotations and FOLD-HONEST
Beat This activations (percussion_bias/asap_foldhonest_activations.npz, logits,
fps=50, alignment verified: sigmoid 0.358 at beat frames vs 0.008 elsewhere).

Split: BY PIECE (stem minus performer token) to kill same-piece leakage across
performers: rng(0) permutation, 70% pieces -> train, 30% -> eval.

meter = mode of beats-per-bar between annotated downbeats (unclamped, unlike
_estimate_meter's [2,4]; meter only enters u via a per-song constant, so
increments are unaffected)."""
import sys, math, os, numpy as np
from collections import Counter
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')

FPS = 50.0
TWO_PI = 2*math.pi
ANN = '/disk1/jaehoon/dataset_store/beat_this_annotations/asap/annotations/beats'
ACT = '/home/sogang/jaehoon/VBPM/percussion_bias/asap_foldhonest_activations.npz'

def _sig(x):
    return 1.0/(1.0+np.exp(-x.astype(np.float64)))

def piece_of(stem):
    return '_'.join(stem.split('_')[:-1])

def split_stems():
    A = np.load(ACT, allow_pickle=True)
    stems = sorted(s[:-6] for s in os.listdir(ANN) if s[:-6] in set(A.files))
    pieces = sorted(set(piece_of(s) for s in stems))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pieces))
    ntr = int(round(0.7*len(pieces)))
    trp = set(pieces[i] for i in perm[:ntr])
    tr = [s for s in stems if piece_of(s) in trp]
    ev = [s for s in stems if piece_of(s) not in trp]
    return tr, ev

def build_asap(split):
    tr, ev = split_stems()
    stems = tr if split=='train' else ev
    A = np.load(ACT, allow_pickle=True)
    out=[]
    for stem in stems:
        rows=[l.split() for l in open(f'{ANN}/{stem}.beats')]
        b=np.array([float(r[0]) for r in rows])
        c=np.array([int(float(r[1])) for r in rows])
        d=b[c==1]
        if len(b) < 8: continue
        # meter: mode of beats-per-bar
        if len(d) >= 2:
            bpb=[int(((b>=d[i])&(b<d[i+1])).sum()) for i in range(len(d)-1)]
            m = Counter(bpb).most_common(1)[0][0]
            m = max(2, min(int(m), 12))
        else:
            m = 4
        logits = A[stem]                       # (2,T)
        a = np.clip(_sig(logits.T), 1e-4, 1-1e-4).astype(np.float32)  # [T,2]
        T = len(a)
        keepb = (b >= 0) & (b < T/FPS)
        b2, c2 = b[keepb], c[keepb]
        d2 = b2[c2==1]
        if len(b2) < 8: continue
        I = np.diff(b2)
        ok = I > 1e-3
        w = TWO_PI/(m*I*FPS)
        u = np.log(w)
        e = np.diff(u)
        bib = np.zeros(len(b2), int)
        if len(d2) >= 1:
            for i,t in enumerate(b2):
                prev = d2[d2 <= t+1e-6]
                if len(prev)==0: bib[i] = -1
                else:
                    j = np.searchsorted(b2, prev[-1]-1e-6)
                    bib[i] = int(round(i-j))
        out.append(dict(stem=stem, dataset='asap', meter=m, T=T,
                        beats=b2, downs=d2, act=a[:T], I=I, u=u, e=e, bib=bib, ok=ok,
                        col2=c2))
    return out

if __name__=='__main__':
    from data import stats
    for sp in ('train','eval'):
        D=build_asap(sp)
        e=np.concatenate([d['e'] for d in D])
        print(sp,'songs',len(D),'pieces',len(set(piece_of(d["stem"]) for d in D)),
              'beats',sum(len(d['u']) for d in D))
        print('  ',stats('e (log-tempo incr / beat)',e))
        bpm=np.concatenate([60.0/d['I'] for d in D])
        print('   bpm median',np.median(bpm),'q01',np.quantile(bpm,0.01),'q99',np.quantile(bpm,0.99))
