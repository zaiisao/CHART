import sys, numpy as np, collections
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
from common import load_labels, FPS

for sp in ("train","eval"):
    L = load_labels(sp)
    print(f"=== {sp}: {len(L)} songs")
    c = collections.Counter(r["dataset"] for r in L)
    print("  datasets:", dict(c))
    cm = collections.Counter((r["dataset"], r["meter"]) for r in L)
    print("  (dataset,meter):", dict(sorted(cm.items())))
    # annotation monotonicity / dup checks
    nb_bad=nd_bad=0; ndup_b=0; nlen=[]
    for r in L:
        b,d = r["beats"], r["downs"]
        b0 = np.asarray(np.load(r["path"],allow_pickle=True)["beats"],float)
        if not np.all(np.diff(b0)>0): nb_bad+=1
        if np.any(np.diff(b0)<=0): ndup_b+=1
        if len(d)>1 and not np.all(np.diff(d)>0): nd_bad+=1
        nlen.append((len(b),len(d),r["T"]/FPS))
    print(f"  songs w/ non-increasing raw beat times: {nb_bad}; non-incr downbeats: {nd_bad}")
    nb=np.array([x[0] for x in nlen]); nd=np.array([x[1] for x in nlen]); dur=np.array([x[2] for x in nlen])
    print(f"  beats/song med {np.median(nb):.0f} min {nb.min()}; downs/song med {np.median(nd):.0f} min {nd.min()}; dur med {np.median(dur):.1f}s total {dur.sum()/3600:.2f}h")
