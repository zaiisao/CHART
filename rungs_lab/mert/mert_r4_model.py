"""MERT-R4 -- COPY of r4/r4_model.py with one change: `input_dim` is an explicit constructor
arg, so the trunk can ingest [BT penultimate 256 ; MERT winners k*768] concatenated+standardized
(input_mode "featsmert"). Everything else (chassis, heads, objective, decode) is r4 verbatim.

Original header:
R4 -- transformer-conditioned bar-pointer (tempo side, v1).

Trunk: small pre-norm transformer with ROTARY position encoding (absolute PE banned: June lesson
-- position alone must not be able to emit periodic structure). Input: frontend activations [T, 2]
or penultimate features [T, 256] (standardized), chosen by `input_mode`.

Heads (all zero-init linears off the trunk, so at init the model IS the R2mix global optimum):
  * initial tempo prior  p(s_0|x): mean-pooled trunk -> [V] logits -> softmax, mixed with a
    uniform floor (blind-spot-bias regularizer: priors modulate, never zero).
  * per-frame transition kernel: component weights over {hold(dither), drift-up, drift-down,
    half-time, double-time} + a width delta on the drift kernels, mixed per frame, plus the same
    uniform floor on every kernel row.

Objective: UNSUPERVISED exact meter-marginal log-likelihood of the activations through the
time-varying forward (structured_dp), exactly R2/R3's objective. Emission is always the Böck
plug-in on the [T, 2] activations regardless of input_mode.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rungs.r1_2016_dbn import DBN2016
from rungs.bar_pointer.readout import state_path_to_events
from rungs.deployment import threshold_crop

KERNEL_COMPONENTS = ("hold", "drift_up", "drift_down", "half_time", "double_time")
OCTAVE_SHARPNESS = 8.0          # fixed peakedness of the half/double-time component kernels
UNIFORM_FLOOR = 0.05            # fixed eps: p = (1-eps)*model + eps*uniform (prior AND kernel rows)


def _rope_cache(seq_len, head_dim, device, dtype):
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = positions[:, None] * inv_freq[None]                       # [T, head_dim/2]
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _apply_rope(x, cos, sin):
    """x: [batch, heads, T, head_dim]"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class RoPEBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def _qkv(self, x, cos, sin):
        T = x.shape[1]
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        shape = (x.shape[0], T, self.num_heads, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))   # [B, H, T, hd]
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        return q, k, v

    def forward(self, x, cos, sin):
        q, k, v = self._qkv(x, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v)                 # flash/mem-efficient
        attn = attn.transpose(1, 2).reshape(x.shape)
        x = x + self.proj(attn)
        return x + self.mlp(self.norm2(x))

    @torch.no_grad()
    def attention_weights(self, x, cos, sin):
        """Explicit [H, T, T] softmax weights (probes only; materializes the matrix)."""
        q, k, _ = self._qkv(x, cos, sin)
        return torch.softmax(q @ k.transpose(-2, -1) / self.head_dim ** 0.5, dim=-1)[0]


class R4Trunk(nn.Module):
    def __init__(self, input_dim, d_model=128, num_layers=3, num_heads=4):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(RoPEBlock(d_model, num_heads) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head_dim = d_model // num_heads

    def forward(self, x):                                              # [T, input_dim]
        h = self.embed(x).unsqueeze(0)
        cos, sin = _rope_cache(x.shape[0], self.head_dim, x.device, h.dtype)
        for block in self.blocks:
            h = block(h, cos, sin)
        return self.norm(h)[0]                                         # [T, d_model]


class R4Conditioned(nn.Module):
    """Owns the chassis (state space/DP/emission), the trunk and the two heads."""

    def __init__(self, fps, input_mode="acts", device="cuda:1", observation_lambda=6,
                 lambda_base=104.43, w_hold0=0.39, d_model=128, num_layers=3, num_heads=4,
                 beats_per_bar=(3, 4), input_dim=None):
        super().__init__()
        self.device, self.input_mode = device, input_mode
        self.lambda_base = float(lambda_base)
        self.chassis = DBN2016(fps=fps, beats_per_bar=beats_per_bar, num_tempi=None,
                               threshold=0.0, correct=False,
                               observation_lambda=observation_lambda,
                               dtype=torch.float32, device=device)
        space = self.chassis.state_spaces[0]
        intervals = space.interval_frames.astype(np.float64)           # shared tempo grid
        self.num_tempi = len(intervals)
        ratio = torch.from_numpy(intervals[None, :] / intervals[:, None]).float()
        self.register_buffer("abs_ratio_dev", (ratio - 1.0).abs().to(device))
        self.register_buffer("mask_up", (ratio <= 1.0).to(device))     # tempo up = shorter interval
        self.register_buffer("mask_down", (ratio >= 1.0).to(device))   # diagonal in both: no empty row
        # hold = the R2mix dither: uniform over |interval diff| <= 1
        adjacency = (torch.from_numpy(np.abs(intervals[None] - intervals[:, None])).float() <= 1.0)
        self.register_buffer("log_hold",
                             torch.log(adjacency / adjacency.sum(1, keepdim=True)).to(device))
        self.register_buffer("log_half",
                             torch.log_softmax(-OCTAVE_SHARPNESS * (ratio - 2.0).abs(), 1).to(device))
        self.register_buffer("log_double",
                             torch.log_softmax(-OCTAVE_SHARPNESS * (ratio - 0.5).abs(), 1).to(device))
        # state -> tempo index and tempo -> total union state count (both meters, same tempo grid)
        self._min_interval = int(intervals[0])
        self.state_tempo_index = [
            torch.from_numpy(s.state_interval_frames - self._min_interval).long().to(device)
            for s in self.chassis.state_spaces]
        total_bpb = sum(self.chassis.beats_per_bar)
        self.register_buffer("log_states_per_tempo",
                             torch.log(torch.from_numpy(total_bpb * intervals).float()).to(device))

        if input_dim is None:
            input_dim = 2 if input_mode == "acts" else 256
        self.trunk = R4Trunk(input_dim, d_model, num_layers, num_heads)
        self.prior_head = nn.Linear(d_model, self.num_tempi)
        self.kernel_head = nn.Linear(d_model, len(KERNEL_COMPONENTS) + 1)   # + width delta
        # SMALL (not zero) head-weight init: zero weights block all gradient into the trunk
        # (measured: trunk grad norm exactly 0.0 at init). std 1e-3 keeps the init within ~1e-3
        # of the R2mix optimum while opening the backprop path.
        nn.init.normal_(self.prior_head.weight, std=1e-3); nn.init.zeros_(self.prior_head.bias)
        nn.init.normal_(self.kernel_head.weight, std=1e-3)
        # component-weight biases: start AT the R2mix optimum (hold=w0, drift split, octaves tiny)
        w0 = float(w_hold0)
        init_weights = torch.tensor([w0, (1 - w0 - 0.02) / 2, (1 - w0 - 0.02) / 2, 0.01, 0.01])
        with torch.no_grad():
            self.kernel_head.bias.copy_(torch.cat([init_weights.log(), torch.zeros(1)]))
        self.to(device)
        self._ablate_input = False                                      # position-ablation switch
        self._zero_trunk = False                                        # degeneracy-check switch

    # --- heads --------------------------------------------------------------------------------
    def trunk_output(self, trunk_input):
        if self._ablate_input:
            trunk_input = torch.zeros_like(trunk_input)
        return self.trunk(trunk_input)

    def head_outputs(self, trunk_input):
        """(log_prior [V], log_kernel [T, V, V], diagnostics dict)."""
        h = self.trunk_output(trunk_input)                              # [T, d]
        if self._zero_trunk:
            h = torch.zeros_like(h)                                     # heads emit their biases only
        prior_logits = self.prior_head(h.mean(0))                       # [V]
        prior = (1 - UNIFORM_FLOOR) * torch.softmax(prior_logits, 0) + UNIFORM_FLOOR / self.num_tempi
        log_prior = prior.log()

        raw = self.kernel_head(h)                                       # [T, 6]
        component_weights = torch.softmax(raw[:, :len(KERNEL_COMPONENTS)], dim=1)     # [T, 5]
        lam = self.lambda_base * torch.exp(raw[:, len(KERNEL_COMPONENTS)])            # [T]
        drift_scores = -lam[:, None, None] * self.abs_ratio_dev[None]                 # [T, V, V]
        neg_inf = torch.finfo(drift_scores.dtype).min
        log_up = torch.log_softmax(drift_scores.masked_fill(~self.mask_up[None], neg_inf), 2)
        log_down = torch.log_softmax(drift_scores.masked_fill(~self.mask_down[None], neg_inf), 2)
        log_w = component_weights.log()                                               # [T, 5]
        T = trunk_input.shape[0]
        stacked = torch.stack([
            log_w[:, 0, None, None] + self.log_hold[None].expand(T, -1, -1),
            log_w[:, 1, None, None] + log_up,
            log_w[:, 2, None, None] + log_down,
            log_w[:, 3, None, None] + self.log_half[None].expand(T, -1, -1),
            log_w[:, 4, None, None] + self.log_double[None].expand(T, -1, -1)])
        log_mix = torch.logsumexp(stacked, dim=0)                                     # [T, V, V]
        log_kernel = torch.logsumexp(torch.stack([
            np.log(1 - UNIFORM_FLOOR) + log_mix,
            torch.full_like(log_mix, np.log(UNIFORM_FLOOR / self.num_tempi))]), dim=0)
        return log_prior, log_kernel, {"component_weights": component_weights, "lambda_t": lam,
                                       "prior": prior}

    def conditioned_log_inits(self, log_prior):
        """Per-meter [num_states] log initial distributions: uniform WITHIN a tempo's union states,
        p(tempo) from the head. Sums to 1 over the meter union, so cross-meter scores compare."""
        per_tempo = log_prior - self.log_states_per_tempo                              # [V]
        return [per_tempo[idx] for idx in self.state_tempo_index]

    # --- objective ----------------------------------------------------------------------------
    def marginal_ll(self, activations, trunk_input):
        """Exact meter-marginal log p(activations); differentiable in trunk + heads."""
        densities = self.chassis.log_class_densities(activations)
        log_prior, log_kernel, _ = self.head_outputs(trunk_input)
        log_inits = self.conditioned_log_inits(log_prior)
        per_meter = [dp.forward_log_likelihood(init, log_kernel, densities, state_to_class=s2c)
                     for dp, init, s2c in zip(self.chassis.dynamic_programs, log_inits,
                                              self.chassis.state_to_classes)]
        return torch.logsumexp(torch.stack(per_meter), dim=0)

    def path_nll(self, activations, trunk_input, state_path, meter_index):
        """SUPERVISED arm: exact conditional NLL of the CLAMPED annotated path,
        -log p(path | x, meter) = logZ_meter - path_score, both under the conditioned initial
        prior AND the per-frame mixture kernel (so prior + kernel heads train on annotations).
        state_path: numpy int64 [T] from chassis.annotated_state_path (jitter-smoothed beats).
        Boundary transition at frames b->b+1 uses kernel[b] (the r3 off-by-one lesson)."""
        chassis = self.chassis
        dp = chassis.dynamic_programs[meter_index]
        space = chassis.state_spaces[meter_index]
        s2c = chassis.state_to_classes[meter_index]
        densities = chassis.log_class_densities(activations)
        log_prior, log_kernel, _ = self.head_outputs(trunk_input)
        log_init = self.conditioned_log_inits(log_prior)[meter_index]

        path = torch.from_numpy(state_path).to(self.device)
        emission = densities[torch.arange(len(state_path), device=self.device),
                             s2c[path].long()].sum()
        intervals = space.state_interval_frames[state_path]
        boundaries = np.where(np.isin(state_path[1:], space.first_states.reshape(-1)))[0]
        from_idx = torch.from_numpy(intervals[boundaries] - self._min_interval).long().to(self.device)
        to_idx = torch.from_numpy(intervals[boundaries + 1] - self._min_interval).long().to(self.device)
        frame_idx = torch.from_numpy(boundaries).long().to(self.device)
        transition = log_kernel[frame_idx, from_idx, to_idx].sum()
        path_score = log_init[path[0]] + emission + transition
        log_z = dp.forward_log_likelihood(log_init, log_kernel, densities, state_to_class=s2c)
        return log_z - path_score

    def joint_path_nll(self, activations, trunk_input, state_path, meter_index):
        """SUPERVISED-GENERATIVE arm: -log p(path, obs) = -path_score (emission fixed, kernel
        rows normalized). No logZ term -> no discriminative normalization competition -> the
        probability-sink exploit (w_half inflation, gate finding 2026-08-09) is impossible by
        construction. This is exact MLE of prior + kernel heads on the clamped annotated
        statistics: the kernel matches empirical transitions, the prior predicts the ANNOTATED
        initial tempo (perceptual octave included)."""
        chassis = self.chassis
        space = chassis.state_spaces[meter_index]
        s2c = chassis.state_to_classes[meter_index]
        densities = chassis.log_class_densities(activations)
        log_prior, log_kernel, _ = self.head_outputs(trunk_input)
        log_init = self.conditioned_log_inits(log_prior)[meter_index]
        path = torch.from_numpy(state_path).to(self.device)
        emission = densities[torch.arange(len(state_path), device=self.device),
                             s2c[path].long()].sum()
        intervals = space.state_interval_frames[state_path]
        boundaries = np.where(np.isin(state_path[1:], space.first_states.reshape(-1)))[0]
        from_idx = torch.from_numpy(intervals[boundaries] - self._min_interval).long().to(self.device)
        to_idx = torch.from_numpy(intervals[boundaries + 1] - self._min_interval).long().to(self.device)
        frame_idx = torch.from_numpy(boundaries).long().to(self.device)
        transition = log_kernel[frame_idx, from_idx, to_idx].sum()
        return -(log_init[path[0]] + emission + transition)

    # --- decode -------------------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, activations, trunk_input, deploy=False, deploy_threshold=0.2):
        """Time-varying Viterbi with the conditioned prior + kernel; best meter wins.
        activations/trunk_input: aligned numpy [T, 2] / [T, input_dim]."""
        acts = np.ascontiguousarray(np.asarray(activations, dtype=np.float64))
        trunk_np = np.ascontiguousarray(np.asarray(trunk_input, dtype=np.float32))
        first = 0
        if deploy:
            acts, first = threshold_crop(acts, deploy_threshold)
            if not acts.size:
                return {"beats": np.array([]), "downbeats": np.array([])}
            trunk_np = trunk_np[first:first + acts.shape[0]]
        acts_t = torch.from_numpy(acts.astype(np.float32)).to(self.device)
        trunk_t = torch.from_numpy(trunk_np).to(self.device)
        densities = self.chassis.log_class_densities(acts_t)
        log_prior, log_kernel, _ = self.head_outputs(trunk_t)
        log_inits = self.conditioned_log_inits(log_prior)
        best = None
        for mi, dp in enumerate(self.chassis.dynamic_programs):
            path, score = dp.viterbi(log_inits[mi], log_kernel, densities,
                                     state_to_class=self.chassis.state_to_classes[mi],
                                     return_log_score=True)
            if best is None or score > best[0]:
                best = (score, path.cpu().numpy(), self.chassis.state_spaces[mi])
        _, path, space = best
        return state_path_to_events(path, space, self.chassis.fps, first_frame=first)
