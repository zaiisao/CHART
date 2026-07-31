# VBPM v2 — Variational Bar-Pointer Model

Built staged, from scratch, against `docs/SPEC.md` (normative). v1 lives in git history
(`master`, `vbpm-campaign-2026-07-26`); nothing here depends on it.

**Stage 0** (current): one latent, `z = m` (beats per bar). Emission / conditional prior /
encoder / exact-enumeration ELBO. Stage 1 adds bar phase `φ`; Stage 2 adds tempo `φ̇`.

## Layout

| path | what |
|---|---|
| `docs/SPEC.md` | the spec — model §4, training §5, data §6, evaluation §8, Appendix A interface |
| `vbpm/` | the library: `stage0.py` (model+fit), `reducers.py`, `data.py` (crops), `fitting.py` (vectorized training, CV protocol, scoring) |
| `train.py` | THE entry point: fold-honest CV training + §8 report on the real corpora |
| `tests/synth_bench.py` | sanity self-check on the §6.4 synthetic bench (not pytest-collected) |
| `tests/` | the acceptance suite: reference oracle + 16 mutants + property checks |
| `experiments/` | one-off measurements (synthetic-h causal control, rich features, e2e evidence head, downbeat decode) |
| `frontends/` | frozen feature extractors (Beat This, Beat Transformer) — VBPM trains no part of them |
| `data/songs.py` | annotation + audio catalog, Beat This 8-fold splits structurally enforced |
| `logs/stage0_*` | committed result records, 2026-07-31 campaign |

## Run

```bash
# environment: conda env "chart" (no bare python; .venv lacks torch)
PY=/disk4/anaconda3/envs/chart/bin/python

$PY -m pytest tests --impl=vbpm -q        # acceptance: 142 passed, 2 skipped
$PY train.py                              # fold-honest CV on real corpora (GPU + data)
$PY tests/synth_bench.py                  # sanity: Stage 0 on the synthetic bench
```

## State of play (2026-07-31)

Suite green. On real data (18,902 bar-aligned 8-bar crops): best deployable meter accuracy
0.595 balanced (e2e evidence head on rich features); the pipeline's ceiling is 0.99
(proven by synthetic-h intervention); the remaining gap is frontend evidence, dataset-dependent.
Grid-constrained downbeat decode beats raw peak-picking on every dataset (F@±70 ms).
Details: `logs/stage0_*`.
