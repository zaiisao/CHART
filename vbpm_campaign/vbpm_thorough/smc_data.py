"""SMC data builder mirroring vbpm_premise/data.py + frozen act-head activations.

SMC has beat-only annotations (no downbeats) -> meter is set to 4 for every song.
This is HARMLESS for the increment analyses: u_k = log(2pi/(m I_k FPS)) shifts by the
per-song constant -log(m), so increments e_k = u_k - u_{k-1} are meter-independent;
only the OU level term sees it, and fitted a ~= 1 makes that negligible.
bib is -1 everywhere (unknown), downs empty.
"""
import sys, os, glob, math
import numpy as np

sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration')
sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix')

FPS = 50.0
TWO_PI = 2*math.pi
HERE = '/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough'
CACHE_DIRS = ['/home/sogang/jaehoon/VBPM/percussion_bias/mert_cache',
              f'{HERE}/mert_cache_extra']
ANN = '/disk1/jaehoon/dataset_store/beat_this_annotations/smc/annotations/beats'
ACT_NPZ = f'{HERE}/act_smc.npz'


def smc_ids():
    return sorted(os.path.basename(p).split('.')[0].split('_')[1]
                  for p in glob.glob(f'{ANN}/*.beats'))


def feat_path(i):
    for d in CACHE_DIRS:
        p = f'{d}/smc_{i}.npy'
        if os.path.exists(p):
            return p
    return None


def build_activations(dev='cuda:0'):
    """Run the FROZEN shared act head (vbpm_ground/act_head_shared.pt) on SMC MERT caches."""
    import torch
    sys.path.insert(0, '/home/sogang/jaehoon/VBPM_reintegration/vbpm_arms')
    from build_act_head import ActHead
    ck = torch.load('/home/sogang/jaehoon/VBPM_reintegration/vbpm_ground/act_head_shared.pt',
                    map_location='cpu', weights_only=False)
    net = ActHead(ck['config']).to(dev)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    out = {}
    for i in smc_ids():
        p = feat_path(i)
        if p is None:
            continue
        f = np.load(p).astype(np.float32)          # [13,T,768]
        with torch.no_grad():
            A = torch.sigmoid(net(torch.from_numpy(f).unsqueeze(0).to(dev)))[0].cpu().numpy()
        out[f'smc_{i}|act'] = A.astype(np.float16)
    np.savez_compressed(ACT_NPZ, **out)
    print('saved', len(out), 'song activations ->', ACT_NPZ)


def load_smc_act():
    d = np.load(ACT_NPZ, allow_pickle=True)
    return {k[:-4]: np.clip(np.asarray(d[k], np.float32), 1e-4, 1-1e-4)
            for k in d.files if k.endswith('|act')}


def build_smc():
    """Mirror of data.build() for SMC."""
    A = load_smc_act()
    out = []
    for i in smc_ids():
        stem = f'smc_{i}'
        if stem not in A:
            continue
        b = np.loadtxt(f'{ANN}/{stem}.beats', ndmin=2)[:, 0].astype(float)
        if len(b) < 8:
            continue
        a = A[stem]
        T = len(a)
        b = b[(b >= 0) & (b < T/FPS)]
        if len(b) < 8:
            continue
        m = 4
        I = np.diff(b)
        ok = I > 1e-3
        w = TWO_PI/(m*I*FPS)
        u = np.log(w)
        e = np.diff(u)
        bib = -np.ones(len(b), int)
        out.append(dict(stem=stem, dataset='smc', meter=m, T=T,
                        beats=b, downs=np.zeros(0), act=a[:T], I=I, u=u, e=e,
                        bib=bib, ok=ok))
    return out


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'act':
        build_activations()
    D = build_smc()
    e = np.concatenate([d['e'] for d in D])
    print('smc songs', len(D), 'beats', sum(len(d['u']) for d in D), 'increments', len(e))
    print('e kurt', float(((e-e.mean())**4).mean()/e.var()**2))
