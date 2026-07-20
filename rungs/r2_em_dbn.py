"""R2 -- learn the DBN's transition_lambda GENERATIVELY, by maximum likelihood of the marginal
p(observations | lambda), with exact EM (Baum-Welch). (Formerly "R1.5"; promoted when the
CRF-trained variant was demoted to experiments/bt_e2e/crf_baseline.py -- its conditional
objective p(z|x) is discriminative, off-program for a ladder of generative latent-variable
models.)

The ladder now separates HOW the same factor is learned:
    R1              lambda hand-set (madmom's 100)
    R2 (this file)  lambda by UNSUPERVISED generative MLE  (no annotations; meter marginalized)
    crf_baseline    lambda by SUPERVISED discriminative CRF (annotated path) -- comparison, not a rung

Two mathematically equivalent trainers are provided, deliberately:
  * em_step():   exact EM. E-step = expected tempo-transition counts under the exact posterior,
                 obtained by autograd on logZ (the classic identity d logZ / d logA_ij = E[#i->j]);
                 M-step = 1-D maximization of the expected complete-data transition log-likelihood
                 over lambda (the exponential-kernel family).
  * The marginal itself (marginal_log_likelihood) is differentiable, so direct gradient ascent on
    it is the SAME estimator -- Fisher's identity: grad log p(x) = E_posterior[grad log p(x,z)].
    Running both and watching them converge to the same lambda is the identity, demonstrated on
    the real model rather than a toy.

The observation model is the Böck plug-in likelihood on the activations (as in R1/R2); the meters
{3,4} are marginalized in the likelihood (union-uniform initial distribution + logsumexp over the
per-meter forwards), so learning uses NO annotation of any kind.
"""
import numpy as np
import torch

from rungs.r1_2016_dbn import DBN2016


class R2GenerativeLambda:
    def __init__(
        self,
        fps: float,
        beats_per_bar=(3, 4),
        init_transition_lambda: float = 100.0,
        observation_lambda: int = 6,
        device: str = "cuda",
        min_bpm: float = 55.0,
        max_bpm: float = 215.0,
        observation_candidates=(2, 4, 6, 8, 16, 32)
    ):
        self._chassis_kwargs = dict()
        self.observation_lambda = int(observation_lambda)
        self.chassis = DBN2016(
            observation_lambda=self.observation_lambda,
            fps=fps,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            beats_per_bar=beats_per_bar,
            num_tempi=None,
            threshold=0.0,
            correct=False,
            dtype=torch.float32,
            device=device
        )
        self._chassis_cache = {self.observation_lambda: self.chassis}
        self._obs_candidates = tuple(int(v) for v in observation_candidates)
        self.device = device
        self.transition_lambda = float(init_transition_lambda)

        intervals = self.chassis.state_spaces[0].interval_frames.astype(np.float32)
        ratio = intervals[None, :] / intervals[:, None]
        self._abs_ratio_dev = torch.from_numpy(np.abs(ratio - 1.0)).to(device)   # [V, V]

    def _chassis_for(self, observation_lambda: int) -> DBN2016:
        """The state space is independent of observation_lambda (it only re-partitions states
        into observation classes), so chassis per candidate are built once and cached."""
        if observation_lambda not in self._chassis_cache:
            self._chassis_cache[observation_lambda] = DBN2016(
                observation_lambda=observation_lambda, **self._chassis_kwargs)
        return self._chassis_cache[observation_lambda]

    # --- model pieces -------------------------------------------------------------------------
    def log_kernel(self, lam) -> torch.Tensor:
        """[V, V] log p(tempo_to | tempo_from) for a given lambda (row log-softmax of -lam*|r-1|)."""
        lam = lam if torch.is_tensor(lam) else torch.tensor(float(lam), device=self.device)
        scores = -lam * self._abs_ratio_dev
        return scores - torch.logsumexp(scores, dim=1, keepdim=True)

    def log_class_densities(self, activations: torch.Tensor,
                            chassis: DBN2016 = None) -> torch.Tensor:
        """Torch Böck densities (same as R1): [T, 2] probs -> [T, 3]."""
        chassis = chassis if chassis is not None else self.chassis
        eps = chassis.eps
        beat = activations[:, 0].clamp(eps, 1 - eps)
        downbeat = activations[:, 1].clamp(eps, 1 - eps)
        bnd = (beat - downbeat).clamp(min=eps)
        no_beat = (1.0 - bnd - downbeat).clamp(min=1e-12)
        return torch.stack([torch.log(no_beat / (chassis.observation_lambda - 1)),
                            torch.log(bnd), torch.log(downbeat)], dim=1)

    def marginal_log_likelihood(self, activations: torch.Tensor, log_kernel: torch.Tensor,
                                chassis: DBN2016 = None) -> torch.Tensor:
        """log p(activations | lambda, observation_lambda): meter-MARGINAL forward (union-uniform
        init makes the per-meter forwards components of one mixture; logsumexp combines them).
        Differentiable. `chassis` selects the observation model (default: the current one)."""
        chassis = chassis if chassis is not None else self.chassis
        densities = self.log_class_densities(activations, chassis)
        per_meter = [dp.forward_log_likelihood(init, log_kernel, densities, state_to_class=s2c)
                     for dp, init, s2c in zip(chassis.dynamic_programs,
                                              chassis.log_initial_distributions,
                                              chassis.state_to_classes)]
        return torch.logsumexp(torch.stack(per_meter), dim=0)

    # --- exact EM -----------------------------------------------------------------------------
    @staticmethod
    def _expected_counts(marginal_ll_fn, crops, log_kernel_value) -> torch.Tensor:
        """E-step: expected tempo-transition counts summed over crops.
        d logZ / d logA_ij = E[# i->j transitions] under the exact posterior."""
        leaf = log_kernel_value.detach().clone().requires_grad_(True)
        for acts in crops:
            marginal_ll_fn(acts, leaf).backward()          # grads ACCUMULATE on leaf
        return leaf.grad.detach()

    def m_step(self, counts: torch.Tensor) -> float:
        """argmax_lambda sum_ij C_ij log p_lambda(j|i): 1-D exact grid + parabolic refine."""
        grid = torch.logspace(np.log10(2.0), np.log10(600.0), 500, device=self.device)
        scores = -grid[:, None, None] * self._abs_ratio_dev[None]                  # [L, V, V]
        log_p = scores - torch.logsumexp(scores, dim=2, keepdim=True)
        objective = (counts[None] * log_p).sum(dim=(1, 2))                          # [L]
        best = int(objective.argmax())
        if 0 < best < len(grid) - 1:                                               # parabolic
            x = torch.log(grid[best - 1:best + 2]); y = objective[best - 1:best + 2]
            denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
            a = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
            b = (x[2]**2 * (y[0] - y[1]) + x[1]**2 * (y[2] - y[0]) + x[0]**2 * (y[1] - y[2])) / denom
            if a < 0:
                return float(torch.exp(-b / (2 * a)))
        return float(grid[best])

    def observation_step(self, crops) -> int:
        """Exact coordinate update for observation_lambda: argmax over the integer candidates of
        the MARGINAL log-likelihood at the current transition_lambda.

        observation_lambda is a discrete PARTITION parameter (it reassigns whole state blocks
        between observation classes), so the objective is piecewise-constant in it -- there is no
        gradient to follow and no smooth Q to maximize over a continuum. The exact M-analog for a
        finite candidate set is direct argmax of the marginal itself: this is coordinate ascent on
        the same ell(lambda, obs) the EM lambda-step climbs, so the combined loop stays monotone
        in the marginal likelihood."""
        log_kernel = self.log_kernel(self.transition_lambda)
        best_obs, best_ll = self.observation_lambda, None
        with torch.no_grad():
            for obs in self._obs_candidates:
                chassis = self._chassis_for(obs)
                total = sum(float(self.marginal_log_likelihood(acts, log_kernel, chassis))
                            for acts in crops)
                if best_ll is None or total > best_ll:
                    best_obs, best_ll = obs, total
        self.observation_lambda = best_obs
        self.chassis = self._chassis_cache[best_obs]
        return best_obs

    def em_step(self, crops, learn_observation_lambda: bool = False) -> float:
        """One exact EM iteration over the given activation crops. Returns the new lambda.
        With learn_observation_lambda=True, follows the lambda M-step with an exact discrete
        argmax over observation_lambda candidates (see observation_step)."""
        counts = self._expected_counts(self.marginal_log_likelihood, crops,
                                       self.log_kernel(self.transition_lambda))
        self.transition_lambda = self.m_step(counts)
        if learn_observation_lambda:
            self.observation_step(crops)
        return self.transition_lambda
