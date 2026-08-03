"""Phase-tracking VAE for downbeat placement on a GIVEN beat grid.

Kept out of ``vbpm/`` deliberately: this is a separate build with its own latent
(continuous bar phase per frame) and its own objective, and it must not inherit
Stage 0's meter-classification surface by accident.
"""
