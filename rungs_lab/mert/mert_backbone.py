"""Frozen MERT-v1-95M backbone loader + all-layer extraction (chart env: transformers 4.44/torch 2.6).

Era decision recovered from git 853b8ad~1 (vbpm_campaign/vbpm/mert_extract.py + configs):
model = m-a-p/MERT-v1-95M, ALL 13 hidden-state layers, 24 kHz, ~75 Hz native, 30 s chunks,
linear interp onto a 50 fps center-of-frame grid, fp16 cache.

CRITICAL remap (era note "pos_conv weight-norm remap REQUIRED"): under torch>=2.x transformers
parametrizes weight norm; the HF checkpoint stores encoder.pos_conv_embed.conv.weight_g/weight_v,
which from_pretrained silently DROPS, leaving the positional conv randomly initialized.
load_mert() below copies them into parametrizations.weight.original0/original1 explicitly and
verifies the effective weight matches the checkpoint's weight-norm product.
"""
import numpy as np
import torch

MERT_ID = "m-a-p/MERT-v1-95M"
MERT_SR = 24000
FPS = 50
CROP_S = 30


def load_mert(device):
    from transformers import AutoModel
    from transformers.utils import cached_file
    model = AutoModel.from_pretrained(MERT_ID, trust_remote_code=True)
    sd = torch.load(cached_file(MERT_ID, "pytorch_model.bin"), map_location="cpu",
                    weights_only=True)
    pre = "encoder.pos_conv_embed.conv."
    g, v = sd[pre + "weight_g"], sd[pre + "weight_v"]
    with torch.no_grad():
        par = model.encoder.pos_conv_embed.conv.parametrizations.weight
        par.original0.copy_(g)
        par.original1.copy_(v)
        expected = g * (v / v.norm(2, dim=(0, 1), keepdim=True))
        assert torch.allclose(model.encoder.pos_conv_embed.conv.weight, expected, atol=1e-5), \
            "pos_conv remap failed"
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def mert_all_layers(model, wav24, device):
    """[13, n, 768] native-rate hidden states + [n] frame-center times (era chunking)."""
    fs, ts, step = [], [], CROP_S * MERT_SR
    for i in range(0, len(wav24), step):
        seg = wav24[i:i + step]
        if len(seg) < MERT_SR // 2:
            break
        h = model(torch.from_numpy(seg).float().unsqueeze(0).to(device),
                  output_hidden_states=True).hidden_states
        f = torch.stack(h)[:, 0]
        n = f.shape[1]
        dur = len(seg) / MERT_SR
        fs.append(f)
        ts.append(i / MERT_SR + (np.arange(n) + 0.5) * (dur / n))
    return torch.cat(fs, 1), np.concatenate(ts)


def interp_to_grid(x, t_src, n_frames, device, fps=FPS):
    """Linear interp [n,768] -> [n_frames,768] on the (k+0.5)/fps grid (era code verbatim)."""
    t_dst = torch.as_tensor((np.arange(n_frames) + 0.5) / fps, dtype=torch.float64, device=device)
    ts = torch.as_tensor(t_src, dtype=torch.float64, device=device)
    idx = torch.searchsorted(ts, t_dst).clamp(1, len(ts) - 1)
    t0, t1 = ts[idx - 1], ts[idx]
    w = ((t_dst - t0) / (t1 - t0).clamp(min=1e-9)).clamp(0, 1).float().unsqueeze(1)
    return x[idx - 1] * (1 - w) + x[idx] * w


def extract_song(model, audio_path, device, fps=FPS):
    """audio file -> fp16 numpy [13, T, 768] at `fps` (frame centers at (k+0.5)/fps)."""
    import librosa
    wav, _ = librosa.load(str(audio_path), sr=MERT_SR, mono=True)
    feats, tsrc = mert_all_layers(model, wav, device)
    n_frames = int(len(wav) / MERT_SR * fps)
    g = torch.stack([interp_to_grid(feats[l].to(device), tsrc, n_frames, device, fps)
                     for l in range(feats.shape[0])])
    return g.half().cpu().numpy()
