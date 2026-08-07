"""Feature extractors (frontends): audio -> [num_frames, num_channels] activations/features.

One script per frontend (beat_this.py, beat_transformer.py, later mert.py, ...). A frontend wraps
the official upstream model behind a small property surface -- WHAT it emits (`fps`, `output`,
`ACTIVATION_FORM`) and HOW to get it (`get_features(signal, sample_rate) ->
[num_frames, num_channels]` -- frontends are feature extractors, and the [T, 2] activations are
just the most compressed feature). Each module declares its class as ``FRONTEND``, so callers
select one by module name exactly the way run.py selects a variant:
``importlib.import_module(f"phasevae.data.frontends.{name}").FRONTEND``.

Output modes: a frontend can usually emit at more than one depth of its network. The classic cut is
the FINAL layer -- [T, 2] (beat, downbeat) activations, what the HMM-family rungs consume -- vs the
PENULTIMATE layer -- rich features (e.g. [T, 512]), what a latent-variable rung conditions on
(deleting the final linear compression). Each frontend class declares its modes in OUTPUT_MODES
(mode name -> num_channels) and is constructed in exactly one mode; the Tracker checks the emitted
channel count against the rung's declared INPUT_CHANNELS, and the config layer additionally demands
the frontend's `output` and the bar-pointer's `input` be declared together (see track.py).

Deliberately simple (a resurrected, slimmed version of the archived
data/feature_extractor.py + configs/frontends/*.yaml system): properties live on the wrapper class,
not in YAML, until we have enough frontends to need config files again.
"""


class Frontend:
    """Interface. A frontend turns audio into [num_frames, num_channels] in its output mode.

    Two routes to the same channels, and both are part of the contract:

      * ``get_features(signal, sample_rate)`` -- audio in, features out, one song at a
        time. The certified single-song path (feature caches, eval, demos).
      * ``prepare_input`` + ``forward_features`` -- the TRAINING-LOOP factorization.
        ``prepare_input`` is the frontend's frozen, checkpoint-independent preprocessing
        (its mel recipe, its demixing, whatever it demands), returning a TIME-FIRST
        array at the frontend's FPS that the excerpt dataset caches per song and slices windows
        from; ``forward_features`` is the frozen model forward over a BATCH of such
        windows. Everything frontend-specific -- axis layout, sample rate, chunking
        limits, subprocess demixing -- lives inside the frontend class, so the dataset
        and training loop never branch on which frontend they hold.
    """

    OUTPUT_MODES: dict = {"activations": 2}
    ACTIVATION_FORM: str = "probability"
    BOUNDING: str = "clip"
    FPS: float                 # the frame rate this frontend's output ticks at -- fixed
                               # by its STFT hop, not chosen. Everything downstream
                               # (targets, scoring) reads THIS; there is no global grid.

    output: str = "activations"

    @property
    def name(self) -> str:
        """Frontend name, derived from the defining module — never declared per class.

        A frontend's identity IS its module under the dotted-path loader:
        frontends.beat_this -> "beat_this". When the module runs as a script
        (__main__), fall back to its file stem.
        """
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
        """[num_samples] mono audio -> np.float32 model input, TIME-FIRST at self.FPS.

        The frozen preprocessing this frontend demands of its input (log-mel spectrogram,
        demixed mel stack, ...). Checkpoint-INDEPENDENT by contract: this is what the
        excerpt dataset caches per song -- cache what is model-free, run the model live.
        Axis 0 is time so the dataset can slice windows without knowing the layout.
        """
        raise NotImplementedError

    def forward_features(self, batch):
        """[B, T, ...] batched ``prepare_input`` windows -> [B, T, num_channels].

        The frozen model forward in the instance's output mode. Frontend-specific
        demands (chunk-size limits, axis permutations, autocast) are handled HERE,
        inside the class -- callers hand over windows and get channels back.
        """
        raise NotImplementedError
