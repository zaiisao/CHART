"""A self-contained phase-vocoder time-stretch (torch STFT/ISTFT only) -- NOT librosa.effects.time_stretch.

Why not librosa: librosa is installable, but librosa.effects.time_stretch transitively imports
librosa.core.audio, which imports `soxr` -- and `soxr` fails to import in this conda env (numpy 2.x ABI
break, the exact same failure this project's own code already works around in several places, e.g.
extract_ood_features.py avoiding beat_this.inference for the same reason: `import soxr` ->
`ImportError: numpy.core.multiarray failed to import`). We don't need librosa's resampling (audio here is
already at 22050 Hz), so rather than touch the shared conda env's soxr/numpy versions (risky with sibling
jobs running), we implement a standard phase-vocoder stretch directly on torch.stft/istft (already a hard
dependency everywhere in this repo, no import-chain issue) -- the same algorithm librosa.effects.time_stretch
itself uses (phase_vocoder + resample-to-restore-original-duration), just without the soxr dependency for
the resample step (torchaudio.functional.resample, sinc-based, already used elsewhere in this project e.g.
data/feature_extractor.py's BeatThisFeatureExtractor path and the archived extract_smc_rich.py).
"""
from __future__ import annotations

import torch
import torchaudio


def time_stretch(waveform: torch.Tensor, rate: float, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
    """Pitch-preserving time stretch: waveform [num_samples] -> stretched waveform.

    rate > 1.0 speeds up (shortens); rate < 1.0 slows down (lengthens) -- matches librosa's convention
    (output_duration = input_duration / rate).

    Standard phase-vocoder recipe (same as librosa.effects.time_stretch / torchaudio.transforms.TimeStretch):
      1. STFT.
      2. torchaudio.functional.phase_vocoder: resample the STFT frames along time by 1/rate, correcting
         phase so the reconstructed signal doesn't sound phasy/robotic (standard PV phase accumulation).
      3. ISTFT back to a stretched-length waveform (this already changes the sample count by ~1/rate --
         no separate resample step needed, unlike a naive "resample to force a different rate" approach).
    """
    if rate == 1.0:
        return waveform
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window,
                      return_complex=True, center=True)                      # [freq, T]
    phase_advance = torch.linspace(0, torch.pi * hop_length, stft.shape[0],
                                   device=waveform.device, dtype=waveform.dtype).unsqueeze(-1)
    stretched_stft = torchaudio.functional.phase_vocoder(stft, rate, phase_advance)
    stretched = torch.istft(stretched_stft, n_fft=n_fft, hop_length=hop_length, window=window, center=True)
    return stretched
