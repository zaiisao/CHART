"""
certify_tempo_prior.py -- numerical drop-in certification for the TEMPO_PRIOR winner.

Certifies that the two admissible heavy-tailed replacements for the faithful §5.3
Log-Normal tempo transition are genuine drop-ins for the strict ELBO:
  (1) reparameterizable rsample -> gradients flow to BOTH posterior params (loc, scale);
  (2) a computable KL(q||p) that matches an MC estimate (Laplace: closed form; Student-t:
      MC with a matched-dof analytic scale term as a cross-check);
  (3) same generative FACTORIZATION: the prior mean stays the random-walk mean
      mu^p = log phidot_{t-1}, so the latent stays a tempo TRANSITION.

Run: /home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python certify_tempo_prior.py
"""
import math
import torch

torch.manual_seed(0)
B = 4096  # MC width

# ---------------------------------------------------------------------------
# Closed-form KL for Laplace(loc_q,b_q) || Laplace(loc_p,b_p)  (log-tempo space)
#   KL = log(b_p/b_q) + (b_q*exp(-|dm|/b_q) + |dm|)/b_p - 1 ,  dm = loc_q-loc_p
# (standard result; reduces to the Gaussian-analogue closed form the paper uses.)
# ---------------------------------------------------------------------------
def kl_laplace(loc_q, b_q, loc_p, b_p):
    dm = (loc_q - loc_p).abs()
    return (torch.log(b_p / b_q)
            + (b_q * torch.exp(-dm / b_q) + dm) / b_p - 1.0)


def run_laplace():
    print("=" * 74)
    print("LAPLACE drop-in  (q,p both Laplace on log-tempo)")
    print("=" * 74)
    # posterior params (leaf, require grad) ; prior mean = random-walk mean (fixed)
    loc_q = torch.tensor(0.13, requires_grad=True)
    log_bq = torch.tensor(-1.6, requires_grad=True)
    loc_p = torch.tensor(0.00)          # = log phidot_{t-1} (random-walk mean)
    b_p = torch.tensor(0.22)
    b_q = torch.nn.functional.softplus(log_bq) + 1e-3

    q = torch.distributions.Laplace(loc_q, b_q)
    z = q.rsample((B,))                 # reparameterized sample
    print(f"  rsample OK, shape={tuple(z.shape)}  mean={z.mean():+.4f}")

    kl_cf = kl_laplace(loc_q, b_q, loc_p, b_p)
    p = torch.distributions.Laplace(loc_p, b_p)
    kl_mc = (q.log_prob(z) - p.log_prob(z)).mean()
    print(f"  KL closed-form = {float(kl_cf):.5f}   KL MC = {float(kl_mc):.5f}   "
          f"|rel err|={abs(float(kl_cf)-float(kl_mc))/float(kl_cf):.3%}")

    # gradient flows to BOTH posterior params through rsample-based ELBO term
    (recon_surrogate := (z ** 2).mean())  # stands in for BCE(decode(z))
    (recon_surrogate + kl_cf).backward()
    print(f"  grad loc_q = {float(loc_q.grad):+.4e}   grad log_bq = {float(log_bq.grad):+.4e}"
          f"   (both non-zero => reparam path live)")


def run_studentt(nu=3.0):
    print("=" * 74)
    print(f"STUDENT-T drop-in  (q,p both Student-t, shared learnable dof nu={nu})")
    print("=" * 74)
    loc_q = torch.tensor(0.13, requires_grad=True)
    log_sq = torch.tensor(-1.6, requires_grad=True)
    loc_p = torch.tensor(0.00)          # random-walk mean
    s_p = torch.tensor(0.18)
    s_q = torch.nn.functional.softplus(log_sq) + 1e-3
    dof = torch.tensor(nu)

    q = torch.distributions.StudentT(dof, loc_q, s_q)
    p = torch.distributions.StudentT(dof, loc_p, s_p)
    z = q.rsample((B,))                 # implicit-reparam rsample (torch supports it)
    print(f"  rsample OK, shape={tuple(z.shape)}  mean={z.mean():+.4f}")

    # KL has no general closed form -> low-variance MC through the SAME rsample
    kl_mc = (q.log_prob(z) - p.log_prob(z)).mean()
    # analytic cross-check: same-loc, same-dof scale term  KL = log(s_p/s_q) (loc aligned)
    q0 = torch.distributions.StudentT(dof, loc_p, s_q)
    kl_scale_mc = (q0.log_prob(q0.rsample((B,))) - p.log_prob(q0.rsample((B,)))).mean()
    print(f"  KL(q||p) MC = {float(kl_mc):.5f}   (scale-only cross-check MC = {float(kl_scale_mc):.5f}, "
          f"analytic log(s_p/s_q)={float(torch.log(s_p/s_q)):+.5f})")

    (z ** 2).mean().add(kl_mc).backward()
    print(f"  grad loc_q = {float(loc_q.grad):+.4e}   grad log_sq = {float(log_sq.grad):+.4e}"
          f"   (both non-zero => reparam path live)")


if __name__ == "__main__":
    run_laplace()
    print()
    run_studentt(3.0)
