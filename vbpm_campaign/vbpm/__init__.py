"""Reality-adjusted VBPM: the variational bar-pointer VAE with five certified in-model
fixes over ``faithful/`` (which remains the strict-ELBO A/B baseline).

This is a VAE (amortized q_phi(z|h), reparameterized latents, ELBO), NOT the HMM rungs.

Fixes (each real-data-justified, reparameterizable, with a computable ELBO KL term):
  1. tempo prior    : Student-t(nu) increments   (heavy tails; dBIC ~653k vs Gaussian)
  2. tempo dynamics : slow RW level + fast AR(1) deviation (mean-reversion; +12.4% pred LL)
  3. phase kernel   : wrapped Cauchy              (heavy circular tail; +1.62 nats/beat)
  4. meter          : bar-gated K x K Categorical (switch only at phase crossings; ~24x/bar)
  5. emission       : beat + downbeat two-Bernoulli (meter identifiability 0.000 -> 0.987)
"""
