"""The strong, learned activation emission -- the make-or-break piece per the project diagnosis.

Historical context (see the task brief / MEMORY): an EARLIER FIVO attempt in this project
(archived as `dbn_vae.py`) used a hand-crafted onset envelope (spectral flux of a log-mel) as the
particle filter's observation and only reached free-run beat-F ~0.37-0.40. The diagnosis was that
the emission was too weak to carry real beat/downbeat information. The "neural-madmom" number this
prototype targets (real beat ~0.72 / downbeat ~0.42) instead used a TRAINED, expressive activation
(Beat-This's own per-frame beat/downbeat probability head) as the particle filter's observation.

We do not have that head's weights in this cache (only its [T,512] penultimate features are cached),
so we reproduce the same *spirit*: a small supervised MLP that maps the frozen [T,512] Beat-This
features to per-frame (beat, downbeat) PROBABILITIES, trained by plain BCE against the ground-truth
targets. This is exactly the "small learned MLP on top of the [512] features" option the diagnosis
names explicitly. It is pretrained (and optionally fine-tuned jointly with FIVO) SEPARATELY from the
particle filter's dynamics -- the emission's job is only to be a good beat/downbeat detector; the
particle filter's job is to turn that detector's per-frame evidence into a temporally consistent,
metrically structured phase trajectory.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ActivationEmissionHead(nn.Module):
    """[*, feature_dim] -> per-frame (p_beat, p_downbeat) in (0, 1). A "neural-madmom" activation."""

    def __init__(self, feature_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        """features [..., feature_dim] -> logits [..., 2] for (beat, downbeat)."""
        return self.net(features)

    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(features))


def pretrain_emission(model: ActivationEmissionHead, songs, config, steps: int, device: str,
                      pos_weight_beat: float = 5.0, pos_weight_downbeat: float = 5.0,
                      lr: float = 1e-3, log_every: int = 200) -> ActivationEmissionHead:
    """Supervised pretraining of the emission head: plain BCE against beat/downbeat targets.

    Beats/downbeats are rare (~5-10% of frames), so a modest positive-class weight keeps the head
    from collapsing to "always predict no-beat" -- this is the SAME positive-weighting mechanism
    already used by the baseline's `divergence_beat_pos_weight` (losses.py), just applied here to a
    standalone supervised detector instead of the VAE's ELBO reconstruction term.
    """
    from data.dataset import sample_training_batch

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([pos_weight_beat, pos_weight_downbeat], device=device)
    model.train()
    for step in range(1, steps + 1):
        features, beat_targets, downbeat_targets = sample_training_batch(
            songs, config.crop_length_frames, config.batch_size, device)
        logits = model.logits(features)
        targets = torch.stack([beat_targets, downbeat_targets], dim=-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                beat_acc = ((probs[..., 0] > 0.5).float() == beat_targets).float().mean()
            print(f"  [emission pretrain] step {step:5d} | bce {loss.item():.4f} | beat_frame_acc {beat_acc.item():.3f}", flush=True)
    model.eval()
    return model
