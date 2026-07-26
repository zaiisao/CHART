"""Extract MERT all-layer features for SMC songs missing from percussion_bias cache.
Identical recipe to /home/sogang/jaehoon/VBPM/percussion_bias/mert_drift_probe.py PHASE A:
MERT-v1-95M, 30s crops, layer stack [13,T,768] fp16, linear interp onto 50fps mid-frame grid.
Writes to vbpm_thorough/mert_cache_extra/ (NEVER touches the original cache dir)."""
import os, sys, glob
import numpy as np, torch, soundfile, librosa
from transformers import AutoModel

FPS, MERT_SR, CROP = 50, 24000, 30
OUT = '/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough/mert_cache_extra'
os.makedirs(OUT, exist_ok=True)
missing = np.load('/home/sogang/jaehoon/VBPM_reintegration/vbpm_thorough/missing_ids.npy')
wav_of = {os.path.basename(p).split('.')[0].split('_')[1]: p
          for p in glob.glob('/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Audio/*.wav')}
DEVICE = 'cuda:0'
mert = AutoModel.from_pretrained('m-a-p/MERT-v1-95M', trust_remote_code=True).to(DEVICE).eval()
for p in mert.parameters(): p.requires_grad_(False)

def load(p):
    w, sr = soundfile.read(str(p), dtype='float32', always_2d=True); w = w.mean(1)
    return (librosa.resample(w, orig_sr=sr, target_sr=MERT_SR) if sr != MERT_SR else w), None

@torch.no_grad()
def extract(path):
    w, _ = load(path)
    n = int(round(len(w)/MERT_SR*FPS)) + 1          # 40.00s -> 2001, matches cached files
    step = CROP*MERT_SR; ch = []; tm = []
    for i in range(0, len(w), step):
        seg = w[i:i+step]
        if len(seg) < MERT_SR//2: break
        h = mert(torch.from_numpy(seg).float().unsqueeze(0).to(DEVICE),
                 output_hidden_states=True).hidden_states
        f = torch.stack(h)[:, 0].float().cpu().numpy(); ni, dur = f.shape[1], len(seg)/MERT_SR
        ch.append(f); tm.append(i/MERT_SR + (np.arange(ni)+0.5)*(dur/ni))
    feats = np.concatenate(ch, axis=1); tsrc = np.concatenate(tm)
    tdst = (np.arange(n)+0.5)/FPS
    return np.stack([np.stack([np.interp(tdst, tsrc, feats[L, :, c]) for c in range(768)], axis=1)
                     for L in range(feats.shape[0])]).astype(np.float16)

for k, s in enumerate(missing):
    dst = f'{OUT}/smc_{s}.npy'
    if os.path.exists(dst): continue
    np.save(dst, extract(wav_of[s]))
    if (k+1) % 10 == 0: print(f'{k+1}/{len(missing)}', flush=True)
print('done', flush=True)
