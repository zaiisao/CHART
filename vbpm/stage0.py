"""Stage 0 model + training, written from docs/SPEC.md §4 (model) and §5 (training).

Three parameter sets (§4.7, Sohn-standard):

    theta : emission  p_theta(y|m)   two scalars {alpha, beta}, bar offset marginalised
    psi   : prior     p_psi(m|h)     reads h only — the deployable path
    phi   : encoder   q_phi(m|h,y)   structured: prior logits + c * log p_theta(y|m)

All log-probabilities natural (C5), all tensors float64 (§5), everything exact — no
sampling anywhere (§4.6).
"""
from __future__ import annotations

import math

import numpy as np
import torch

DEFAULT_VALUES = (2, 3, 4)

_MASK_CACHE: dict = {}


def _offset_masks(n: int, m: int) -> torch.Tensor:
    """[m, n] mask: entry (r, i) = 1 iff beat i is a downbeat under bar offset r (§4.3)."""
    key = (n, m)
    hit = _MASK_CACHE.get(key)
    if hit is None:
        idx = torch.arange(n, dtype=torch.float64)
        hit = torch.stack(
            [(((idx - r) % m) == 0).to(torch.float64) for r in range(m)])
        _MASK_CACHE[key] = hit
    return hit


def reduce_h(h) -> torch.Tensor:
    """§4.4 default reducer: s(h) = concat[mean_t h, max_t h] ∈ R^{2D}.

    Swappable behind this one function, as §4.4 requires. Does not assume D.
    """
    h_t = torch.as_tensor(np.asarray(h, dtype=np.float64), dtype=torch.float64)
    return torch.cat([h_t.mean(dim=0), h_t.max(dim=0).values])


class Stage0:
    """One Stage-0 implementation of the Appendix-A surface."""

    def __init__(self, values=DEFAULT_VALUES, h_dim: int = 2, reducer=None,
                 s_dim: int = None):
        self.values = tuple(int(v) for v in values)
        K = len(self.values)
        # §4.4: the reducer is swappable behind one function; default s(h) = mean⊕max
        self.reducer = reducer if reducer is not None else reduce_h
        if s_dim is None:
            s_dim = 2 * h_dim
        # theta (§4.3): two learnable scalars
        self.alpha = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        self.beta = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)
        # psi (§4.4): logits = W · s(h) + b
        self.W = torch.zeros((K, s_dim), dtype=torch.float64, requires_grad=True)
        self.b = torch.zeros(K, dtype=torch.float64, requires_grad=True)
        # phi (§4.5): g_phi(y) = c · log p_theta(y|m); exact posterior at c = 1
        self.c = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

    # -- C1: count <-> index, explicit and one-way ---------------------------------
    def to_idx(self, m: int) -> int:
        m = int(m)
        if m not in self.values:
            raise ValueError(f"{m} is not a legal meter count in {self.values}")
        return self.values.index(m)

    def to_value(self, k: int) -> int:
        return int(self.values[int(k)])

    # -- theta: emission (§4.3) ----------------------------------------------------
    def emission_logp_all(self, y) -> torch.Tensor:
        """[K] log p_theta(y|m) = log[(1/m) Σ_r Π_i Bern(y_i; π_i^{(r)})]."""
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
        n = len(y_t)
        lsig = torch.nn.functional.logsigmoid
        on = y_t * lsig(self.alpha) + (1 - y_t) * lsig(-self.alpha)
        off = y_t * lsig(self.beta) + (1 - y_t) * lsig(-self.beta)
        out = []
        for m in self.values:
            M = _offset_masks(n, m)                              # [m, n]
            per_offset = (M * on + (1 - M) * off).sum(-1)        # [m]
            out.append(torch.logsumexp(per_offset, 0) - math.log(m))
        return torch.stack(out)

    # -- psi: conditional prior (§4.4) — the deployable path, reads h ONLY (C2) ----
    def prior_logp(self, h) -> torch.Tensor:
        logits = self.W @ self.reducer(h) + self.b
        return logits - torch.logsumexp(logits, -1)

    def predict(self, h) -> torch.Tensor:
        return self.prior_logp(h)

    # -- phi: encoder (§4.5), structured on the prior's logits ---------------------
    def q_logp(self, h, y) -> torch.Tensor:
        logits = self.prior_logp(h) + self.c * self.emission_logp_all(y)
        return logits - torch.logsumexp(logits, -1)

    # -- inference / objective (§4.6), exact over K terms --------------------------
    def exact_posterior(self, h, y) -> torch.Tensor:
        lp = self.emission_logp_all(y) + self.prior_logp(h)
        return lp - torch.logsumexp(lp, -1)

    def log_evidence(self, h, y) -> torch.Tensor:
        return torch.logsumexp(self.emission_logp_all(y) + self.prior_logp(h), dim=-1)

    def elbo(self, h, y) -> torch.Tensor:
        """Scalar E_q[log p_theta(y|m)] − KL(q ‖ p_psi)."""
        q_logp = self.q_logp(h, y)
        q = q_logp.exp()
        recon = (q * self.emission_logp_all(y)).sum()
        kl = (q * (q_logp - self.prior_logp(h))).sum()
        return recon - kl

    # -- training (§5) -------------------------------------------------------------
    def fit(self, songs, steps: int = 500, lr: float = 0.5, seed: int = 0) -> "Stage0":
        """Maximise the mean ELBO over crops: Adam, full batch, exact enumeration.

        Deterministic — the seed is accepted for interface parity but nothing samples.
        A tied parameter (e.g. beta := alpha) reaches the optimiser once (§5).
        """
        torch.manual_seed(seed)

        seen, params = set(), []
        for p in self.named_params().values():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        if not params:
            return self

        opt = torch.optim.Adam(params, lr=lr)
        cache = [(s["h"], s["y"]) for s in songs]

        for _ in range(steps):
            opt.zero_grad()
            loss = -torch.stack([self.elbo(h, y) for h, y in cache]).mean()
            loss.backward()
            opt.step()

        return self

    # -- introspection (§4.7 / §10.2) ----------------------------------------------
    def param_groups(self) -> dict:
        return {
            "theta": {"alpha": self.alpha, "beta": self.beta},
            "psi": {"W": self.W, "b": self.b},
            "phi": {"c": self.c},
        }

    def named_params(self) -> dict:
        out = {}
        for group in self.param_groups().values():
            out.update(group)
        return out
