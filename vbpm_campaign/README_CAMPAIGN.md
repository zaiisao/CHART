# VBPM verdict campaign (2026-07-25/26) — consolidated working tree

Snapshot of the adversarial investigation that closed the VBPM-as-VAE question.
Regenerable caches (>5MB npz/npy; MERT features live at /disk1/jaehoon/vbpm_mert_cache) are excluded;
checkpoints backing claims are included.

## Headline results (79 eval songs = fold 0, ballroom/beatles/hainsworth; density-matched blind controls throughout)
- PAPER-STANDARD NEGATIVE: learned/variational conditioning of the bar-pointer TRANSITION cannot beat a
  well-tuned fixed heavy-tailed kernel (trained anchored best 0.646 < hand 0.751; causal-mean audio
  conditioning dead on steady/SMC/ASAP; all deploy wins came from inference).
- POSITIVE: MASK2 (per-meter PF posterior phase mask x activation -> peak-pick) beats the frozen head on
  BOTH channels: beats 0.8285 vs 0.8115 (sign p=1.9e-5), downbeats 0.559 vs 0.534, CMLt 0.638 vs 0.564,
  AMLt 0.880 vs 0.664. Shape: fixed physics + supervised evidence + particle inference wins.
- Tempo increment law = Student-t(nu~2) replicated on steady+SMC+ASAP (Gaussian loses >=0.3 nats/step).
- PF-vs-peakpick inversion was METER-IDENTITY failure; per-meter PFs + logZ selection fix it (0.633->0.722).
- Frontend is OOD-blind on SMC (head 0.441 vs 0.8115 in-domain) — the binding constraint for the SMC mission.

## Directory map
- vbpm/            reality-adjusted VBPM package (latent-only likelihood, OU tempo, all 5 certified fixes)
- faithful/        strict-ELBO baseline (untouched)
- vbpm_debug/      Dirac/audio-blind root-cause probes (open-loop metronome proof)
- vbpm_fix/        variant B (p(h|z) + particle filter) + conditioning fixes
- vbpm_arms/       shared activation head + rich-h vs activation-only arms
- vbpm_final/      supervised emission + PF pipeline (the working 0.751 system) + E1/E3 decompositions
- vbpm_premise/    premise tests: P1 tempo law, P2 audio-vs-history, lambda response curves
- vbpm_anchor/     physical prior fitting (rho=0.9994, per-meter lt means)
- vbpm_wtf/        paradox probes (tempo side-channel confirmation)
- vbpm_thorough/   final thoroughness: V1/V2 verification, X1 SMC, X2 ASAP, F1/F2 (MASK2 win)
- component_tests/ oracle unit tests (83/83 distributions etc.)
- reality_checks/  spec-vs-reality measurements
- alternatives/    certified drop-in alternatives (Student-t, wrapped Cauchy, ...)
- docs/            ELBO_for_DBN.md + professor tutorial notes (north star)

Full session narrative: memory files project_vbpm_final_verdict.md and antecedents.
