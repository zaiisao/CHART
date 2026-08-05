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
| `phasevae/` | the model (`model.py`), training CLI (`run.py`), data/batching (`loading.py`), pre-flight controls (`controls.py`), scoring (`evaluation.py`), tests (`tests/`, 69 blind-written) |
| `docs/phasevae_decisions.md` | measured rationale behind every flag and recorded deviations |
| `vbpm/data.py` | fold-honest frontend feature pass (the single authority) |
| `data/songs.py` | song catalog: Beat This annotations + official 8-fold splits + local audio |
| `frontends/` | Beat This / Beat Transformer wrappers over `external/` submodules |
| `logs/phasevae/` | full training logs of the 2026-08 campaign |

## Run

    PYTHONPATH=. python -m phasevae.run --gpu 1 --epochs 60 --seeds 0 1 2 \
        --emission triangle --drift-bound 0.01 --crop-cache /disk4/jaehoon/phasevae_dedup.pkl \
        --gtzan-checkpoint fold7 --save-dir checkpoints/<name>

    PYTHONPATH=. python -m pytest phasevae/tests -q     # 69 tests, CPU, ~2 s
