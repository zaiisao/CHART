# VBPM — phase-only bar-pointer VAE (branch `phase-min`)

This branch is the phase-only bar-phase VAE: one continuous latent (bar phase, one turn
per bar), the bar period given per crop, downbeats read off the phase trajectory by the
deterministic rule g. Architecture follows the tutorial's §7 configuration (encoder-only
deployment, fixed physical prior, no psi); every knob's measured rationale is in
`docs/phasevae_decisions.md`.

Stage-0 (meter latent) and the earlier campaign surfaces are deleted here — recover them
from `master` or git history. v1 lives in `vbpm-campaign-2026-07-26`.

## Layout

| path | what |
|---|---|
| `phasevae/` | the model (`model.py`), training CLI (`run.py`), the config schema (`config.py` + `config_schema.json`), hooks modules (`variants/`), data (`data/`), pre-flight controls and scoring (`scoring/`), tests (`tests/`) |
| `docs/phasevae_decisions.md` | measured rationale behind every flag and recorded deviations |
| `vbpm/data.py` | fold-honest frontend feature pass (the single authority) |
| `data/songs.py` | song catalog: Beat This annotations + official 8-fold splits + local audio |
| `frontends/` | Beat This / Beat Transformer wrappers over `external/` submodules |
| `logs/phasevae/` | full training logs of the 2026-08 campaign |

## Run

    PYTHONPATH=. python -m phasevae.run --config phasevae/configs/anchor_k.yaml --gpu 1

The recipe is the config's business; the CLI carries only run mechanics (device, seed,
paths). Override one key for one run with `--set`, repeatable:

    PYTHONPATH=. python -m phasevae.run --config phasevae/configs/baseline.yaml \
        --set epochs=2 --set emission=cosine --save-dir checkpoints/<name>

Every mainline key, its default, its type and why it has that value:
`phasevae/config_schema.json` (a variant's extra keys are in its own module's `DEFAULTS`).
Anything else in a config -- or a value of the wrong type, or outside the declared range --
refuses at parse time.

    PYTHONPATH=. python -m pytest phasevae/tests -q     # 69 tests, CPU, ~2 s
