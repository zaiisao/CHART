"""The Stage 0 model and its reference training loop, written from docs/SPEC.md.

Stage 0 keeps exactly one latent: m, the number of beats per bar, one value per crop.
The beat grid is given; the model's only job is to explain WHICH beats are downbeats.
Three parameter sets (the Sohn conditional-VAE split):

    theta : emission  p_theta(y|m)   two scalars {alpha, beta}, bar offset marginalised
    psi   : prior     p_psi(m|h)     reads audio features only — the deployable path
    phi   : encoder   q_phi(m|h,y)   the prior's logits plus c * log p_theta(y|m)

Everything is exact: with only 3 candidate meters, every expectation is a 3-term sum, so
nothing is sampled and two evaluations agree bit-for-bit. All log-probabilities are
natural logs; all tensors float64. The public surface below is pinned by docs/SPEC.md
(Appendix A) and certified by tests/ — do not rename or reshape it.
"""
from __future__ import annotations

import math

import numpy as np
import torch

DEFAULT_VALUES = (2, 3, 4)

_MASK_CACHE: dict = {}


def _offset_masks(n: int, m: int) -> torch.Tensor:
    """[m, n] mask: entry (r, i) = 1 iff beat i is a downbeat under bar offset r.

    A crop may start anywhere within a bar, so the emission considers every possible
    offset r of the downbeat pattern and (in emission_logp_all) averages over them.
    """
    key = (n, m)
    hit = _MASK_CACHE.get(key)
    if hit is None:
        idx = torch.arange(n, dtype=torch.float64)
        hit = torch.stack(
            [(((idx - r) % m) == 0).to(torch.float64) for r in range(m)])
        _MASK_CACHE[key] = hit
    return hit


def reduce_h(h) -> torch.Tensor:
    """The default reducer: s(h) = concat[mean over time, max over time], length 2D.

    The prior needs a fixed-size input but h is [T, D] with variable T, so some reduction
    must happen. This one is deliberately the simplest thing that could work — the spec
    marks it "a default, not a finding" and requires it be swappable behind one function
    (pass reducer= to Stage0), which is how the peak-summary variant plugs in. It makes
    no assumption about D. (Measured 2026-07-31: on real crops this default collapses to
    always-4; the swappability clause earned its keep.)
    """
    h_t = torch.as_tensor(np.asarray(h, dtype=np.float64), dtype=torch.float64)
    return torch.cat([h_t.mean(dim=0), h_t.max(dim=0).values])


class Stage0:
    """One Stage-0 implementation of the interface pinned in docs/SPEC.md Appendix A."""

    def __init__(self, values=DEFAULT_VALUES, h_dim: int = 2, reducer=None,
                 s_dim: int = None):
        self.values = tuple(int(v) for v in values)
        K = len(self.values)
        # the reducer is swappable behind this one attribute; default is mean⊕max
        self.reducer = reducer if reducer is not None else reduce_h
        if s_dim is None:
            s_dim = 2 * h_dim
        # theta — emission: just two learnable logit scalars (downbeat-slot / other-slot)
        self.alpha = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        self.beta = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)
        # psi — prior: linear logits on the reduced features, W @ s(h) + b
        self.W = torch.zeros((K, s_dim), dtype=torch.float64, requires_grad=True)
        self.b = torch.zeros(K, dtype=torch.float64, requires_grad=True)
        # phi — encoder: one scalar c scaling the emission term; at c = 1 the encoder
        # IS the exact posterior, so any amortization gap is a fitting result
        self.c = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

    # -- meter is a COUNT, never an index; the conversion is explicit and one-way ----
    # (v1 once consumed a count as an index and silently merged 3/4 with 4/4 for weeks,
    #  which is why these two tiny functions exist at all)
    def to_idx(self, m: int) -> int:
        """Beats-per-bar count -> position in ``values``; raises on an illegal count."""
        m = int(m)
        if m not in self.values:
            raise ValueError(f"{m} is not a legal meter count in {self.values}")
        return self.values.index(m)

    def to_value(self, k: int) -> int:
        """Position in ``values`` -> beats-per-bar count, the inverse of to_idx."""
        return int(self.values[int(k)])

    # -- theta: the emission --------------------------------------------------------
    def emission_logp_all(self, y) -> torch.Tensor:
        """[K] log p_theta(y|m) for every candidate meter, ordered by ``values``.

        Generative story per meter m: pick a bar offset r uniformly, then each beat is
        a downbeat with probability sigmoid(alpha) if it sits on the m-periodic comb at
        offset r, else sigmoid(beta). The offset is MARGINALISED (averaged, the 1/m),
        not maximised — no crop start position is privileged, which also makes the score
        invariant to cyclic shifts of y whenever m divides len(y).
        """
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

    # -- psi: the conditional prior — the ONLY path that survives to deployment -----
    def prior_logp(self, h) -> torch.Tensor:
        """[K] log p_psi(m|h): softmax of W @ reduce(h) + b.

        Reads the audio features only, never the annotations — anything that will run
        at deployment time must go through here.
        """
        logits = self.W @ self.reducer(h) + self.b
        return logits - torch.logsumexp(logits, -1)

    def predict(self, h) -> torch.Tensor:
        """The deployed predictor: identical to the prior by construction.

        At deployment only h exists (no labels), so predicting IS evaluating the prior.
        Kept as its own method so tests can pin that it never grows label access.
        """
        return self.prior_logp(h)

    # -- phi: the encoder, built ON TOP of the prior's logits ------------------------
    def q_logp(self, h, y) -> torch.Tensor:
        """[K] log q_phi(m|h,y): softmax of [prior logits + c * emission log-likelihood].

        Structured this way on purpose: the exact posterior is softmax(prior logits +
        emission), so writing q in the same shape means any gap between q and the true
        posterior is attributable to the single scalar c — not to q being unable to see
        the prior. At c = 1, q equals the exact posterior.
        """
        logits = self.prior_logp(h) + self.c * self.emission_logp_all(y)
        return logits - torch.logsumexp(logits, -1)

    # -- exact inference and the objective (3-term sums, nothing sampled) ------------
    def exact_posterior(self, h, y) -> torch.Tensor:
        """[K] log p(m|y,h) by Bayes: normalise emission log-likelihood + prior."""
        lp = self.emission_logp_all(y) + self.prior_logp(h)
        return lp - torch.logsumexp(lp, -1)

    def log_evidence(self, h, y) -> torch.Tensor:
        """Scalar log p(y|h): logsumexp over meters of emission + prior."""
        return torch.logsumexp(self.emission_logp_all(y) + self.prior_logp(h), dim=-1)

    def elbo(self, h, y) -> torch.Tensor:
        """Scalar evidence lower bound: E_q[log p_theta(y|m)] − KL(q ‖ p_psi(m|h)).

        Equals log_evidence exactly when q is the exact posterior; the slack is
        KL(q ‖ posterior). Training ascends this jointly in theta, psi, phi.
        """
        q_logp = self.q_logp(h, y)
        q = q_logp.exp()
        recon = (q * self.emission_logp_all(y)).sum()
        kl = (q * (q_logp - self.prior_logp(h))).sum()
        return recon - kl

    # -- the reference training loop --------------------------------------------------
    def fit(self, songs, steps: int = 500, lr: float = 0.5, seed: int = 0) -> "Stage0":
        """Maximise the mean ELBO over crops: Adam, full batch, exact enumeration.

        Deterministic — the seed parameter is part of the pinned interface but nothing
        here samples, so two fits from different seeds land on identical parameters.
        The defaults (500 steps, lr 0.5) are the spec's working values: weaker settings
        were shown to underfit a prior that must actually learn its features.
        For large real-data runs use vbpm.fitting.fit_vectorized, the fast twin that is
        verified equal to this loop before every use.
        """
        torch.manual_seed(seed)

        params = self.trainable_params()
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

    # -- introspection: exists so "did every parameter actually train" is a TEST -----
    # (v1 ran for weeks with half its parameters at exactly zero gradient; the audit
    #  surface below is what makes that a checkable property instead of a post-mortem)
    def param_groups(self) -> dict:
        """The theta/psi/phi split, as {group: {name: tensor}}.

        Only theta's parameterisation (two named scalars) is fixed by the spec; psi and
        phi are implementation choices, so outside code must reach them by GROUP and
        shape, never by field name — a test that hardcodes psi's shape binds the suite
        to one implementation.
        """
        return {
            "theta": {"alpha": self.alpha, "beta": self.beta},
            "psi": {"W": self.W, "b": self.b},
            "phi": {"c": self.c},
        }

    def named_params(self) -> dict:
        """Every trainable tensor, flat — the gradient-audit surface."""
        out = {}
        for group in self.param_groups().values():
            out.update(group)
        return out

    def trainable_params(self) -> list:
        """The optimiser set, with each TIED parameter included exactly once.

        Handing the same tensor to the optimiser twice silently doubles its learning
        rate — the meter-free null (beta tied to alpha) would hit exactly that.
        """
        seen, params = set(), []
        for p in self.named_params().values():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        return params
