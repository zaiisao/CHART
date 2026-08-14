"""Beat Transformer frontend (Zhao, Xia & Wang, ISMIR 2022; external/beat_transformer submodule)."""
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile
import torch

from . import Frontend

_BEAT_TRANSFORMER_ROOT = Path(__file__).resolve().parents[3] / "external" / "beat_transformer"
_DEMIX_SCRIPT = Path(__file__).resolve().parent / "beat_transformer_demix.py"
_DEMIX_PYTHON = "/home/sogang/mnt/db_2/anaconda3/envs/analyze-smc/bin/python"

# Their eight_fold_test.py hyperparameters -- must match the released checkpoints.
_MODEL_KWARGS = dict(attn_len=5, instr=5, ntoken=2, dmodel=256, nhead=8,
                     d_hid=1024, nlayers=9, norm_first=True)


class BeatTransformerFrontend(Frontend):
    """Official Beat Transformer inference (Spleeter demix subprocess); see module docstring."""

    OUTPUT_MODES = {"activations": 2, "features": 256, "features+activations": 258}

    ACTIVATION_FORM = "logit"
    BOUNDING = "none"          # their pipeline feeds madmom raw sigmoids, no clip/squeeze,
                               # decorrelation floor exactly 0 (ours: 1e-12 for log safety)
    FPS = 44100 / 1024         # ~= 43.066; Spleeter's hop, a fact not a knob

    def __init__(self, checkpoint: str = "fold_0", device: str = "cuda",
                 output: str = "activations",
                 demix_python: str = _DEMIX_PYTHON, spleeter_model_path: Optional[str] = None):
        if output not in self.OUTPUT_MODES:
            raise KeyError(f"unknown output mode {output!r} for {self.name} "
                           f"(have: {sorted(self.OUTPUT_MODES)})")
        self.output = output

        code_dir = str(_BEAT_TRANSFORMER_ROOT / "code")
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from DilatedTransformer import Demixed_DilatedTransformerModel

        checkpoint_path = _BEAT_TRANSFORMER_ROOT / "checkpoint" / f"{checkpoint}_trf_param.pt"
        if not checkpoint_path.exists():
            available = sorted(p.name.split("_trf_")[0]
                               for p in (_BEAT_TRANSFORMER_ROOT / "checkpoint").glob("*.pt"))
            raise KeyError(f"unknown checkpoint {checkpoint!r} (have: {available})")

        self._model = Demixed_DilatedTransformerModel(**_MODEL_KWARGS)
        state = torch.load(str(checkpoint_path), map_location="cpu")["state_dict"]
        self._model.load_state_dict(state)
        self._model.to(device).eval()
        # features = the instrument-pooled representation entering out_linear (dmodel wide);
        # read the width off the model so the declared modes stay honest.
        self.OUTPUT_MODES = {**self.OUTPUT_MODES, "features": self._model.dmodel,
                             "features+activations": self._model.dmodel + 2}

        self.checkpoint = checkpoint
        self.device = device
        self.demix_python = demix_python
        self.spleeter_model_path = spleeter_model_path

    @torch.no_grad()
    def get_features(self, signal, sample_rate: int) -> torch.Tensor:
        """[num_samples] mono audio -> [num_frames, num_channels] at FPS (~43.07)."""
        x = self._demix(signal, sample_rate)                                  # [5, T, 128]
        batch = torch.from_numpy(x).transpose(0, 1).unsqueeze(0)              # [1, T, 5, 128]
        return self.forward_features(batch)[0].cpu()                          # [T@43fps, C]

    def prepare_input(self, signal, sample_rate: int) -> np.ndarray:
        """[num_samples] mono audio -> [T, 5, 128] demixed log-mel stack, TIME-FIRST."""
        stack = self._demix(signal, sample_rate)                              # [5, T, 128]
        return np.ascontiguousarray(stack.transpose(1, 0, 2), dtype=np.float32)

    @torch.no_grad()
    def forward_features(self, batch) -> torch.Tensor:
        """[B, T, 5, 128] demixed-mel windows -> [B, T, num_channels]."""
        x = batch.to(self.device).permute(0, 2, 1, 3)                         # [B, 5, T, 128]
        captured = []
        handle = self._model.out_linear.register_forward_hook(
            lambda module, inputs, output: captured.append((inputs[0], output)))
        try:
            self._model(x)
        finally:
            handle.remove()
        features, logits = (t.float() for t in captured[0])
        if self.output == "features":
            return features
        if self.output == "activations":
            return logits[:, :, :2]
        return torch.cat([features, logits[:, :, :2]], dim=-1)

    def _demix(self, signal, sample_rate: int) -> np.ndarray:
        """[5, T, 128] log-compressed demixed mel via the Spleeter subprocess (see module doc)."""
        with tempfile.TemporaryDirectory(prefix="bt_demix_") as tmp:
            wav_path = Path(tmp) / "in.wav"
            npz_path = Path(tmp) / "out.npz"
            soundfile.write(str(wav_path), np.asarray(signal, dtype=np.float32), sample_rate)
            command = [self.demix_python, str(_DEMIX_SCRIPT), str(wav_path), str(npz_path)]
            if self.spleeter_model_path:
                command += ["--model-path", self.spleeter_model_path]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"spleeter demixing failed "
                                   f"({self.demix_python}):\n{result.stderr[-2000:]}")
            return np.load(str(npz_path))["x"]


FRONTEND = BeatTransformerFrontend  # the module's class, for by-name selection
