"""R3 -- audio-conditioned PER-FRAME transition tolerance, trained by the exact time-varying forward.

R2 learned ONE global transition_lambda; tonight's experiments showed the F-optimal kernel is
"stiff on steady passages, yielding only at genuine tempo changes" -- which a single scalar cannot
be. R3 changes EXACTLY that one thing: lambda becomes lambda_t = lambda_base * exp(net(x_t)), a
per-frame stiffness produced by a small temporal conv net reading the activations, so the kernel
can loosen precisely where the audio signals an imminent tempo change and tighten elsewhere. The
net is initialized to output ~0, so lambda_t starts at lambda_base (R2's optimum) everywhere and
learns deviations from it.

Everything else is R2's chassis: same state space / DP / read-out (the deployment interface is
unchanged), so any R2-vs-R3 difference is attributable to per-frame conditioning alone. The
structured DP already accepts a per-frame [T, V, V] kernel (see rungs/bar_pointer/structured_dp.py
_kernel_at); R3 is what produces it.
"""
import numpy as np
import torch
from torch import nn

from rungs.r1_2016_dbn import DBN2016


class R3ConditionedFactors(nn.Module):
    """Training-side owner of the per-frame tempo kernel. Conditions on [T, 2] activations."""

    def __init__(self, fps: float, beats_per_bar=(3, 4), lambda_base: float = 100.0,
                 observation_lambda: int = 6, device: str = "cuda",
                 min_bpm: float = 55.0, max_bpm: float = 215.0, hidden: int = 32):
        super().__init__()
        self.chassis = DBN2016(fps=fps, min_bpm=min_bpm, max_bpm=max_bpm,
                               beats_per_bar=beats_per_bar, num_tempi=None,
                               threshold=0.0, correct=False,
                               observation_lambda=observation_lambda,
                               dtype=torch.float32, device=device)
        self.device = device
        self.lambda_base = lambda_base
        self._min_interval = int(self.chassis.state_spaces[0].interval_frames[0])
        self._max_interval = int(self.chassis.state_spaces[0].interval_frames[-1])

        # |ratio - 1| over the tempo grid, precomputed [V, V]
        intervals = torch.from_numpy(
            self.chassis.state_spaces[0].interval_frames.astype(np.float32)).to(device)
        self.register_buffer("abs_ratio_dev",
                             (intervals[None, :] / intervals[:, None] - 1.0).abs())

        # conditioning net: [T, 2] activations -> [T] per-frame log-multiplier on lambda_base.
        # Temporal convs (odd kernels, 'same' padding) see local activation shape; final layer
        # zero-initialized so lambda_t == lambda_base at start (begin at R2's global optimum).
        self.net = nn.Sequential(
            nn.Conv1d(2, hidden, 7, padding=3), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 7, padding=3), nn.ReLU(),
            nn.Conv1d(hidden, 1, 1),
        ).to(device)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def log_lambda_per_frame(self, activations: torch.Tensor) -> torch.Tensor:
        """[T, 2] activations -> [T] log lambda_t (natural log)."""
        x = activations.t().unsqueeze(0)                      # [1, 2, T]
        delta = self.net(x)[0, 0]                             # [T]
        return np.log(self.lambda_base) + delta

    def per_frame_kernel(self, activations: torch.Tensor) -> torch.Tensor:
        """[T, V, V] log tempo kernel, one per frame, differentiable in the conditioning net."""
        lam = self.log_lambda_per_frame(activations).exp()    # [T]
        scores = -lam[:, None, None] * self.abs_ratio_dev[None]   # [T, V, V]
        return scores - torch.logsumexp(scores, dim=2, keepdim=True)

    # --- R2 machinery reused verbatim (path construction + densities) ---
    def annotated_state_path(self, beat_frames, beat_in_bar, beats_per_bar):
        meters = tuple(self.chassis.beats_per_bar)
        if int(beats_per_bar) not in meters:
            return None
        mi = meters.index(int(beats_per_bar))
        space = self.chassis.state_spaces[mi]
        gaps = np.diff(beat_frames)
        if gaps.min() < self._min_interval or gaps.max() > self._max_interval:
            return None
        if not np.all((beat_in_bar[1:] - beat_in_bar[:-1]) % int(beats_per_bar) == 1):
            return None
        path = np.concatenate([space.first_states[beat_in_bar[i], gap - self._min_interval]
                               + np.arange(gap) for i, gap in enumerate(gaps)])
        return path.astype(np.int64), mi

    def log_class_densities(self, activations: torch.Tensor) -> torch.Tensor:
        eps = self.chassis.eps
        beat = activations[:, 0].clamp(eps, 1 - eps)
        downbeat = activations[:, 1].clamp(eps, 1 - eps)
        bnd = (beat - downbeat).clamp(min=eps)
        n_nb = self.chassis.observation_lambda - 1
        no_beat = (1.0 - bnd - downbeat).clamp(min=1e-12)
        return torch.stack([torch.log(no_beat / n_nb), torch.log(bnd), torch.log(downbeat)], dim=1)

    def crf_nll(self, activations, state_path, meter_index):
        """-log p(annotated path | activations) with the PER-FRAME kernel. Differentiable in the
        conditioning net AND the activations (e2e)."""
        chassis, dp = self.chassis, self.chassis.dynamic_programs[meter_index]
        space = chassis.state_spaces[meter_index]
        densities = self.log_class_densities(activations)
        kernel = self.per_frame_kernel(activations)               # [T, V, V]
        s2c = chassis.state_to_classes[meter_index]
        log_init = chassis.log_initial_distributions[meter_index]

        path = torch.from_numpy(state_path).to(self.device)
        emission = densities[torch.arange(len(state_path), device=self.device),
                             s2c[path].long()].sum()
        intervals = space.state_interval_frames[state_path]
        boundaries = np.where(np.isin(state_path[1:], space.first_states.reshape(-1)))[0]
        from_idx = torch.from_numpy(intervals[boundaries] - self._min_interval).long().to(self.device)
        to_idx = torch.from_numpy(intervals[boundaries + 1] - self._min_interval).long().to(self.device)
        # each boundary crossing occurs at frame boundaries+1 -> use that frame's kernel
        # frame of each boundary crossing: state_path[b+1] is a first_state, produced by the
        # forward step frame=b+1 which uses kernel[frame-1]=kernel[b]. (Was boundaries+1 -- an
        # off-by-one that made the NLL invalid; caught by adversarial review 2026-07-18.)
        frame_idx = torch.from_numpy(boundaries).long().to(self.device)
        transition = kernel[frame_idx, from_idx, to_idx].sum()
        path_score = log_init[path[0]] + emission + transition

        log_z = dp.forward_log_likelihood(log_init, kernel, densities, state_to_class=s2c)
        return log_z - path_score

    @torch.no_grad()
    def decode(self, activations: torch.Tensor, meter_index: int = None) -> dict:
        """Time-varying Viterbi + read-out -> events. Picks the best-scoring meter (like R1)."""
        from rungs.bar_pointer.readout import state_path_to_events
        densities = self.log_class_densities(activations)
        kernel = self.per_frame_kernel(activations)
        meters = range(len(self.chassis.state_spaces)) if meter_index is None else [meter_index]
        best = None
        for mi in meters:
            dp = self.chassis.dynamic_programs[mi]
            path, score = dp.viterbi(self.chassis.log_initial_distributions[mi], kernel, densities,
                                     state_to_class=self.chassis.state_to_classes[mi],
                                     return_log_score=True)
            if best is None or score > best[0]:
                best = (score, path.cpu().numpy(), self.chassis.state_spaces[mi])
        _, path, space = best
        return state_path_to_events(path, space, self.chassis.fps)
