"""Beat This frontend (Foscarin, Schlueter & Widmer, ISMIR 2024; external/beat_this submodule)."""
import torch

from . import Frontend


class BeatThisFrontend(Frontend):
    """Official Beat This inference behind the Frontend surface; see module docstring."""

    OUTPUT_MODES = {"activations": 2, "features": 512, "features+activations": 514}

    ACTIVATION_FORM = "logit"
    BOUNDING = "squeeze"       # Beat This's published convention: sigmoid(x)*(1-eps) + eps/2.
                               # With this, Tracker(frontend, "madmom_dbn") is EVENT-IDENTICAL to
                               # the official Audio2Beats(dbn=True) (verified 8/8 GTZAN songs);
                               # with "clip" it diverged on 1/8 -- the convention is NOT always
                               # event-neutral, though mean F is identical to 4 decimals.
    FPS = 50.0                 # 22050 Hz / hop 441: a fact of their spectrogram, not a knob

    def __init__(self, checkpoint: str = "final0", device: str = "cuda",
                 float16: bool = False, output: str = "activations"):
        if output not in self.OUTPUT_MODES:
            raise KeyError(f"unknown output mode {output!r} for {self.name} "
                           f"(have: {sorted(self.OUTPUT_MODES)})")
        self.output = output
        # beat_this is editable-installed from external/beat_this (see environment.yml)
        from beat_this.inference import Audio2Frames

        self._audio2frames = Audio2Frames(checkpoint_path=checkpoint, device=device,
                                          float16=float16)
        # The features width is the loaded checkpoint's transformer_dim (the frontend stack's final
        # linear projects into it); read it off the model so small checkpoints declare honestly.
        transformer_dim = self._audio2frames.model.frontend.linear.out_features
        self.OUTPUT_MODES = {**self.OUTPUT_MODES, "features": transformer_dim,
                             "features+activations": transformer_dim + 2}
        self.checkpoint = checkpoint
        self.device = device

    @torch.no_grad()
    def get_features(self, signal, sample_rate: int) -> torch.Tensor:
        """[num_samples] mono audio -> [num_frames, num_channels] at FPS (50)."""
        if self.output == "activations":
            beat_logits, downbeat_logits = self._audio2frames(signal, sample_rate)
            out = torch.stack([beat_logits, downbeat_logits], dim=-1)         # [T@50fps, 2]
        elif self.output == "features+activations":
            features = self._penultimate(signal, sample_rate)                 # [T@50fps, C]
            heads = self._audio2frames.model.task_heads(features.unsqueeze(0))
            out = torch.cat([features, heads["beat"][0, :, None],
                             heads["downbeat"][0, :, None]], dim=-1)          # [T, C+2]
        else:
            out = self._penultimate(signal, sample_rate)                      # [T@50fps, C]
        return out.cpu()

    def prepare_input(self, signal, sample_rate: int):
        """[num_samples] mono audio -> [T, 128] float32 log-mel at 50 fps (their recipe)."""
        import numpy as np
        spect = self._audio2frames.signal2spect(signal, sample_rate)
        return np.ascontiguousarray(spect.cpu().numpy(), dtype=np.float32)

    def forward_features(self, batch) -> torch.Tensor:
        """[B, T, 128] mel windows -> [B, T, num_channels] in the instance's output mode."""
        batch = batch.to(self.device)
        num_frames = batch.shape[1]
        if num_frames <= self._CHUNK_SIZE:
            return self._head_channels(self._features_forward(batch))

        from beat_this.inference import split_piece
        _chunks, starts = split_piece(batch[0], self._CHUNK_SIZE,
                                      border_size=self._BORDER_SIZE, avoid_short_end=True)
        border, chunk_size = self._BORDER_SIZE, self._CHUNK_SIZE
        # split_piece's starts begin at -border with zero-PADDED first/last chunks;
        # padding the whole batch once makes every chunk the slice
        # padded[:, start+border : start+border+chunk_size], exactly their zeropad.
        padded = torch.nn.functional.pad(batch, (0, 0, border, border))
        out = None
        for start in reversed(list(starts)):                                  # keep_first
            window = padded[:, start + border:start + border + chunk_size]
            piece = self._head_channels(self._features_forward(window))[:, border:-border]
            if out is None:
                out = piece.new_zeros(batch.shape[0], num_frames, piece.shape[-1])
            # their aggregate_prediction's write region, verbatim
            out[:, start + border:start + border + piece.shape[1]] = piece
        return out

    def _features_forward(self, batch: torch.Tensor) -> torch.Tensor:
        """[B, <=1500, 128] -> [B, T, transformer_dim] via the transformer_blocks hook."""
        model = self._audio2frames.model
        captured = []
        handle = model.transformer_blocks.register_forward_hook(
            lambda module, inputs, output: captured.append(output))
        try:
            with torch.autocast(enabled=self._audio2frames.float16,
                                device_type=self._audio2frames.device.type):
                model(batch)
        finally:
            handle.remove()
        return captured[0].float()

    def _head_channels(self, features: torch.Tensor) -> torch.Tensor:
        """[B, T, C] features -> the instance's output mode ([B, T, 2/C/C+2])."""
        if self.output == "features":
            return features
        heads = self._audio2frames.model.task_heads(features)
        activations = torch.cat([heads["beat"][..., None],
                                 heads["downbeat"][..., None]], dim=-1)
        if self.output == "activations":
            return activations
        return torch.cat([features, activations], dim=-1)

    # spect2frames' chunking constants (external/beat_this/beat_this/inference.py) -- the features
    # path MUST split/aggregate identically or feature frames misalign with the logits pipeline.
    _CHUNK_SIZE = 1500
    _BORDER_SIZE = 6

    def _penultimate(self, signal, sample_rate: int) -> torch.Tensor:
        """[T, transformer_dim] transformer_blocks output."""
        from beat_this.inference import split_piece

        model = self._audio2frames.model
        spect = self._audio2frames.signal2spect(signal, sample_rate)
        captured = []
        handle = model.transformer_blocks.register_forward_hook(
            lambda module, inputs, output: captured.append(output))
        try:
            with torch.inference_mode():
                with torch.autocast(enabled=self._audio2frames.float16,
                                    device_type=self._audio2frames.device.type):
                    chunks, starts = split_piece(spect, self._CHUNK_SIZE,
                                                 border_size=self._BORDER_SIZE,
                                                 avoid_short_end=True)
                    for chunk in chunks:
                        model(chunk.unsqueeze(0))
        finally:
            handle.remove()
        feature_chunks = [chunk[0].float() for chunk in captured]             # each [chunk_T, C]

        piece = torch.zeros((spect.shape[0], feature_chunks[0].shape[1]), device=spect.device)
        border, chunk_size = self._BORDER_SIZE, self._CHUNK_SIZE
        for start, chunk in reversed(list(zip(starts, feature_chunks))):      # keep_first
            piece[max(start + border, 0):start + chunk_size - border] = chunk[border:-border]
        return piece


FRONTEND = BeatThisFrontend        # the module's class, for by-name selection
