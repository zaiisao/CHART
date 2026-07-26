import sys, math, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')
from audit_common import load_split, FPS
from common import targets
ARMS='/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms'
import json
print(json.dumps(json.load(open(f'{ARMS}/act_head_report.json'))['summary'],indent=1,default=float)[:1500])

for split in ['eval','train']:
    songs=load_split(split)
    d=np.load(f'{ARMS}/act_{split}.npz',allow_pickle=True)
    lags_b=[]; lags_d=[]; peaks=[]; means=[]; base=[]
    for s in songs:
        a=np.asarray(d[s['stem']+'|act'],np.float32)
        T=a.shape[0]
        assert T==s['T'],(s['stem'],T,s['T'])
        b,db=targets(s['beats'],s['downs'],0,T)
        for ch,tgt,store in ((0,b,lags_b),(1,db,lags_d)):
            x=a[:,ch]-a[:,ch].mean(); y=tgt-tgt.mean()
            L=25
            cc=np.array([np.dot(x[max(0,l):T+min(0,l)], y[max(0,-l):T-max(0,l)]) for l in range(-L,L+1)])
            store.append(np.arange(-L,L+1)[cc.argmax()])
        means.append(a.mean(0)); base.append([b.mean(),db.mean()])
    lags_b=np.array(lags_b); lags_d=np.array(lags_d)
    print(f'{split}: n={len(songs)}  beat-channel peak lag (frames): median {np.median(lags_b)} mean {lags_b.mean():.2f} frac|lag|<=1 {np.mean(np.abs(lags_b)<=1):.3f}')
    print(f'{split}: downbeat-channel peak lag: median {np.median(lags_d)} mean {lags_d.mean():.2f} frac|lag|<=1 {np.mean(np.abs(lags_d)<=1):.3f}')
    m=np.mean(means,0); bs=np.mean(base,0)
    print(f'{split}: mean activation per channel {m}  label density {bs}  (=> base-rate BCE per 256fr: beat {256*(-bs[0]*np.log(bs[0])-(1-bs[0])*np.log(1-bs[0])):.1f} db {256*(-bs[1]*np.log(bs[1])-(1-bs[1])*np.log(1-bs[1])):.1f})')
