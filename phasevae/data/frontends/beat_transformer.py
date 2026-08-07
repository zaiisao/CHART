"""Beat Transformer frontend (Zhao, Xia & Wang, ISMIR 2022; external/beat_transformer submodule).

Why this frontend matters for the ladder: Beat Transformer was DESIGNED around the madmom DBN --
its published results are activation + DBN -- whereas Beat This performs best without one. Pairing
the DBN-family rungs with the frontend that was co-designed with the DBN is the fairer baseline
comparison.

Wraps the OFFICIAL code: Demixed_DilatedTransformerModel (code/DilatedTransformer.py) with their
released fold checkpoints, fed their exact input -- Spleeter 5-stem demixed, log-compressed mel
spectrograms. Spleeter needs TensorFlow, which does not coexist with this env's torch stack, so
demixing runs in a SUBPROCESS under a Spleeter-equipped interpreter (see
frontends/beat_transformer_demix.py; default: the analyze-smc env, whose 5-stem weights are
already cached). The torch model itself runs in-process.

Properties of what this emits:
  * the grid is 44100/1024 ~= 43.066 fps (Spleeter's STFT hop) -- FPS is a fact of the
    hop, and every consumer reads it off the class rather than assuming a global grid.
  * activations are LOGITS (ACTIVATION_FORM="logit"); Beat Transformer's own pipeline applies
    sigmoid and feeds madmom with NO bounding (BOUNDING="none") and decorrelation floor 0
    (eight_fold_test.py: np.maximum(beat - downbeat, 0)).
  * their shipped decode differs from Beat This's madmom call: observation_lambda=6,
    num_tempi=None, threshold=0.2 (and correct=True, madmom's default). To run R0/R1 as
    Beat-Transformer-ships-it, pass those explicitly -- our shipped defaults are Beat This's.
  * the model takes the WHOLE piece in one forward (dilated self-attention; no chunking in their
    inference code).

Checkpoints: "fold_0".."fold_7" -- their 8-fold split over ballroom/hainsworth/carnatic/
harmonix/smc (seed-0 shuffle; NOT Beat This's folds). GTZAN was held out of ALL folds, so any
fold is honest on GTZAN; on the training datasets, fold-honesty needs their fold mapping.
"""
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
        """[num_samples] mono audio -> [num_frames, num_channels] at FPS (~43.07).

        Channels per output mode: (beat, downbeat) LOGITS; the instrument-pooled
        pre-out_linear features [T, dmodel]; or [features ⊕ activations].
        """
        x = self._demix(signal, sample_rate)                                  # [5, T, 128]
        batch = torch.from_numpy(x).transpose(0, 1).unsqueeze(0)              # [1, T, 5, 128]
        return self.forward_features(batch)[0].cpu()                          # [T@43fps, C]

    def prepare_input(self, signal, sample_rate: int) -> np.ndarray:
        """[num_samples] mono audio -> [T, 5, 128] demixed log-mel stack, TIME-FIRST.

        The Spleeter demix + per-stem power_to_db -- this frontend's whole preprocessing
        demand -- runs here (subprocess, separate env). Stored time-first per the
        Frontend contract; forward_features permutes back to their (instr, time) layout.
        """
        stack = self._demix(signal, sample_rate)                              # [5, T, 128]
        return np.ascontiguousarray(stack.transpose(1, 0, 2), dtype=np.float32)

    @torch.no_grad()
    def forward_features(self, batch) -> torch.Tensor:
        """[B, T, 5, 128] demixed-mel windows -> [B, T, num_channels].

        One whole-window forward -- dilated self-attention has no chunk-size demand.
        Features come off a forward hook on out_linear (its input is the instrument-
        pooled representation, its output the logits), so one pass serves every mode.
        """
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
