"""Fresh Beat-This feature extraction for OOD eval sets (SMC, GTZAN), matching bt_train_rich's EXACT recipe.

Why "fresh" matters: cache/acts/smc_rich_heldout's pre-cached features turned out to use 8 DIFFERENT
per-fold Beat-This checkpoints (fold0..fold7, see the archived tests/extract_smc_rich.py), not "final0" --
the checkpoint bt_train_rich/bt_val_rich (and hence the trained KVAE model) actually used. Using
mismatched-checkpoint features would confound "genuine OOD audio difficulty" with "frontend weight
mismatch". This script re-derives features from RAW AUDIO with the same checkpoint + frame-rate
convention bt_train_rich used, for both SMC and GTZAN, so both OOD checks are apples-to-apples with the
in-domain training/eval features.

The bt_train_rich recipe (reconstructed from the archived training/extractors/beat_this_backend.py's
"resample" fps mode, cross-referenced against config.py's FRAMES_PER_SECOND = 22050/256 = 86.1328125):
  1. Load audio, resample to mono 22050 Hz.
  2. Beat This's OWN LogMelSpect (native hop_length=441, i.e. its pretrained 50 fps convention -- do NOT
     change this hop; the frontend was trained on spectrograms at this hop, changing it would feed the
     transformer out-of-distribution spectrograms).
  3. feat_native = transformer_blocks(frontend(spect))   # [T_native (~50 fps), 512]
  4. Linearly interpolate feat_native along time up to T_target frames, where
     T_target = round(num_audio_samples / 256)  (the 22050/256 = 86.1328125 fps grid), matching EXACTLY
     what training/extractors/beat_this_backend.py's compute_hidden_and_activations does in "resample"
     mode (torch.nn.functional.interpolate(..., mode="linear", align_corners=False)).
This is upsampling an already-computed representation, not re-running the frontend at a different hop --
faithful to how bt_train_rich/bt_val_rich were actually built.

Checkpoint: "final0" (matches data/feature_extractor.py's BeatThisFeatureExtractor default and
bt_train_rich's --beat_this_checkpoint final0), loaded via beat_this's own torch.hub URL (avoiding
beat_this.inference's soxr import, which is broken in this conda env -- same workaround the archived
extract_smc_rich.py used).

Usage as a library:
    from extract_ood_features import BeatThisResampledExtractor
    extractor = BeatThisResampledExtractor(device="cuda:0")
    features = extractor.extract_from_wav(wav_path)   # [T, 512] float32, at 86.1328125 fps
"""
from __future__ import annotations

import inspect
import os
import sys

import torch
import torch.nn.functional as F
import torchaudio

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BEAT_THIS_ROOT = os.path.join(_REPO_ROOT, "external", "beat_this")
if _BEAT_THIS_ROOT not in sys.path:
    sys.path.insert(0, _BEAT_THIS_ROOT)

from beat_this.model.beat_tracker import BeatThis                # noqa: E402
from beat_this.preprocessing import LogMelSpect                  # noqa: E402
from beat_this.utils import replace_state_dict_key                # noqa: E402

_CHECKPOINT_URL_BASE = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"
TARGET_SAMPLE_RATE = 22050
TARGET_HOP = 256                                    # 22050/256 = 86.1328125 fps -- bt_train_rich's grid
FRAMES_PER_SECOND = TARGET_SAMPLE_RATE / TARGET_HOP  # 86.1328125, matches config.FRAMES_PER_SECOND


def _load_beat_this_checkpoint(name: str, device) -> BeatThis:
    """Load a Beat This checkpoint WITHOUT importing beat_this.inference (its soxr import is broken in
    this conda env -- numpy 2.x ABI mismatch). Same workaround as the archived extract_smc_rich.py."""
    checkpoint = torch.hub.load_state_dict_from_url(
        f"{_CHECKPOINT_URL_BASE}/{name}.ckpt", file_name=f"beat_this-{name}.ckpt",
        map_location=device, check_hash=False)
    hyperparameters = {k: v for k, v in checkpoint["hyper_parameters"].items()
                       if k in set(inspect.signature(BeatThis).parameters)}
    model = BeatThis(**hyperparameters)
    model.load_state_dict(replace_state_dict_key(checkpoint["state_dict"], "model.", ""))
    return model.to(device).eval()


class BeatThisResampledExtractor:
    """Wraps Beat-This "final0" + its native LogMelSpect, exposing bt_train_rich's 86.13fps feature grid."""

    def __init__(self, checkpoint_name: str = "final0", device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = _load_beat_this_checkpoint(checkpoint_name, self.device)
        self.core = getattr(self.model, "_orig_mod", self.model)  # unwrap torch.compile if present
        self.spect = LogMelSpect(device=self.device)               # native hop_length=441 (Beat This default)

    @torch.no_grad()
    def extract_from_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """waveform: mono [num_samples] at any sample_rate -> features [T, 512] at 86.1328125 fps."""
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
        waveform = waveform.to(self.device)

        spect = self.spect(waveform)                                        # [T_native(~50fps), 128]
        feat_native = self.core.transformer_blocks(self.core.frontend(spect.unsqueeze(0)))  # [1, T_native, 512]

        target_frames = round(waveform.shape[-1] / TARGET_HOP)               # bt_train_rich's frame grid
        feat_resampled = F.interpolate(
            feat_native.transpose(1, 2), size=target_frames, mode="linear", align_corners=False,
        ).transpose(1, 2)                                                    # [1, T_target, 512]
        return feat_resampled[0].float().cpu()

    def extract_from_wav(self, wav_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform.mean(dim=0)  # mono
        return self.extract_from_waveform(waveform, sample_rate)
