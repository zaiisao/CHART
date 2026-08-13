"""Feature extractors (frontends): audio -> [num_frames, num_channels] activations/features."""


class Frontend:
    """Interface. A frontend turns audio into [num_frames, num_channels] in its output mode."""

    OUTPUT_MODES: dict = {"activations": 2}
    ACTIVATION_FORM: str = "probability"
    BOUNDING: str = "clip"
    FPS: float                 # the frame tempo this frontend's output ticks at -- fixed
                               # by its STFT hop, not chosen. Everything downstream
                               # (targets, scoring) reads THIS; there is no global grid.

    output: str = "activations"

    @property
    def name(self) -> str:
        """Frontend name, derived from the defining module — never declared per class."""
        module_name = type(self).__module__.rsplit(".", 1)[-1]

        if module_name == "__main__":
            import inspect
            from pathlib import Path
            try:
                return Path(inspect.getfile(type(self))).stem
            except TypeError:        # class defined interactively; nothing better to derive
                pass

        return module_name

    @property
    def num_channels(self) -> int:
        """Channel count of the constructed output mode."""
        return self.OUTPUT_MODES[self.output]

    def get_features(self, signal, sample_rate: int):
        """[num_samples] mono audio -> [num_frames, num_channels] in the instance's output mode."""
        raise NotImplementedError

    def prepare_input(self, signal, sample_rate: int):
        """[num_samples] mono audio -> np.float32 model input, TIME-FIRST at self.FPS."""
        raise NotImplementedError

    def forward_features(self, batch):
        """[B, T, ...] batched ``prepare_input`` windows -> [B, T, num_channels]."""
        raise NotImplementedError
