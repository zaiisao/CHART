"""X2(c): does the PHYSICAL fixed transition (fitted on the steady corpus) hold on ASAP?
1) transfer nats/beat: steady-fitted law scored on ASAP held-out pairs vs ASAP-fitted law
2) frame-level ideal bar-pointer law on TRUE beat-linear phase: frac_neg + phase-advance R2
   (persistence: predict this interval's rate by the previous interval's mean rate)
3) beat-level persistence R2 of u (how much tempo variance persistence explains)"""
import sys, math, json, numpy as np
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise')
sys.path.insert(0,'/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough')
from core import prep, pairs, gather, score_rw, logmass
from data import build
from asap_data import build_asap
from phases import phase_beatlinear, inside_mask, frame_t, wrap, TWO_PI

OUT={}
# ---------- 1) transfer of the steady-fitted law ----------------------------------
steady = json.load(open('/home/sogang/jaehoon/VBPM_reintegration/vbpm_premise/step1.json'))
asap   = json.load(open('asap_step1.json'))
Dev = prep(build_asap('eval')); Gev2 = gather(pairs(Dev,'second'))
Sev = prep(build('eval'));      Gs2  = gather(pairs(Sev,'second'))
print('ASAP HO pairs', Gev2['n'], '| steady HO pairs', Gs2['n'])
tr={}
for nm,fam,nu in [('gauss_RW','gauss',0),('laplace_RW','laplace',0),('t2.0_RW','t',2.0)]:
    Mst=dict(th=np.array(steady[nm]['th']), ou=False, fam=fam, nu=nu)
    ll_transfer = score_rw(Mst, Gev2).mean()
    ll_native   = asap[nm]['eval2']
    ll_steady_on_steady = steady[nm]['eval2']
    print(f"{nm:12s} steady-law on ASAP {ll_transfer:+.4f} | ASAP-fitted on ASAP {ll_native:+.4f} "
          f"(room {ll_native-ll_transfer:+.4f} nats/beat) | steady-law on steady {ll_steady_on_steady:+.4f}")
    tr[nm]=dict(transfer=float(ll_transfer), native=float(ll_native),
                room=float(ll_native-ll_transfer), steady_home=float(ll_steady_on_steady),
                scale_steady=float(math.exp(steady[nm]['th'][-1])), scale_asap=float(math.exp(asap[nm]['th'][-1])))
OUT['transfer']=tr

# ---------- 2) frame-level ideal bar-pointer law on TRUE phase --------------------
def framelaw(D, name):
    fn_list=[]; r2_list=[]; pooled_num=0.0; pooled_den=0.0; nneg=0; ntot=0
    for d in D:
        T=d['T']; ph=phase_beatlinear(d, T)
        if ph is None: continue
        msk=inside_mask(d, T)
        if msk.sum()<100: continue
        dph=wrap(np.diff(ph))
        m2=msk[1:]&msk[:-1]
        b=d['beats']; t=frame_t(T)[1:]
        # per-frame persistence prediction: mean advance over the PREVIOUS beat interval
        k=np.searchsorted(b, t, side='right')-1        # interval index of each frame
        pred=np.full(len(dph), np.nan)
        # mean advance of interval j = 2pi/(m_bar) per beat / (I_j*FPS) -> just average dph over that interval
        # compute interval means of dph
        valid=m2&(k>=1)&(k<len(b)-1)
        # interval mean advance from TRUE phase (rate of interval j)
        rates={}
        for j in range(0,len(b)-1):
            sel=(k==j)&m2
            if sel.sum()>0: rates[j]=dph[sel].mean()
        for j in range(1,len(b)-1):
            if j-1 in rates:
                sel=(k==j)&valid
                pred[sel]=rates[j-1]
        ok=valid&~np.isnan(pred)
        if ok.sum()<50: continue
        y=dph[ok]; p=pred[ok]
        num=((y-p)**2).sum(); den=((y-y.mean())**2).sum()
        pooled_num+=num; pooled_den+=den
        r2_list.append(1-num/max(den,1e-12))
        nneg+=int((dph[m2]<0).sum()); ntot+=int(m2.sum())
        fn_list.append(float((dph[m2]<0).mean()))
    res=dict(pooled_R2=float(1-pooled_num/pooled_den), median_song_R2=float(np.median(r2_list)),
             frac_neg=float(nneg/max(ntot,1)), median_song_frac_neg=float(np.median(fn_list)),
             n_songs=len(r2_list))
    print(f"{name:12s} frame-level: pooled R2={res['pooled_R2']:+.4f} median-song R2={res['median_song_R2']:+.4f} "
          f"frac_neg={res['frac_neg']:.4f} (median song {res['median_song_frac_neg']:.4f}) n={res['n_songs']}")
    return res
OUT['framelaw_asap']  =framelaw(Dev,'ASAP eval')
OUT['framelaw_steady']=framelaw(Sev,'steady eval')

# ---------- 3) beat-level persistence R2 of u -------------------------------------
def beatR2(G, name):
    e=G['u']-G['u_prev']
    # pooled within-song: subtract per-song mean of u
    from collections import defaultdict
    idx=defaultdict(list)
    for i,s in enumerate(G['stem']): idx[s].append(i)
    num=0.0; den=0.0
    for s,ii in idx.items():
        ii=np.array(ii); u=G['u'][ii]
        num+=float((e[ii]**2).sum()); den+=float(((u-u.mean())**2).sum())
    r2=1-num/den
    print(f"{name:12s} beat-level persistence R2 of u (within-song) = {r2:+.4f}")
    return float(r2)
OUT['persistR2_asap']  =beatR2(Gev2,'ASAP HO')
OUT['persistR2_steady']=beatR2(Gs2,'steady HO')
json.dump(OUT, open('asap_physical.json','w'), indent=1)
print('DONE')
