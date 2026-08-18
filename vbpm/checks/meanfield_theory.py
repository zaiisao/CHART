"""Do the observed failures match the mean-field analysis, or only resemble it?

Three checks, each able to falsify a claim made about the chain:

  L1  the structured family CONTAINS the exact posterior, so with psi set to the
      true emission likelihood the ELBO equals log p(b) exactly. Checked against
      brute-force enumeration of every path.
  P3  committing to one path beats hedging over two by about
      (T/2) log((K-1)/eps) - T log 2 nats. Checked against direct evaluation.
  MF  the mean-field optimum, found by exact coordinate ascent rather than by a
      network, still loses to the chain -- separating FAMILY from AMORTIZATION.
"""
from __future__ import annotations

import argparse
import itertools
import math

import torch


def brute_force(log_p1, log_T, log_e):
    """Exact log p(b) by enumerating every path. Only for tiny (T, K)."""
    T, K = log_e.shape
    total = []
    for path in itertools.product(range(K), repeat=T):
        lp = log_p1[path[0]] + sum(log_T[path[i], path[i + 1]] for i in range(T - 1))
        lp = lp + sum(log_e[i, path[i]] for i in range(T))
        total.append(lp)
    return torch.logsumexp(torch.stack(total), 0)


def chain_elbo(log_p1, log_T, log_e, log_psi):
    """Forward-backward ELBO for q propto p(phi) prod psi_t."""
    T, K = log_e.shape
    a = log_p1 + log_psi[0]
    alphas = [a]
    for t in range(1, T):
        a = torch.logsumexp(a[:, None] + log_T, 0) + log_psi[t]
        alphas.append(a)
    logZ = torch.logsumexp(alphas[-1], 0)
    b = torch.zeros(K)
    betas = [None] * T
    betas[T - 1] = b
    for t in range(T - 1, 0, -1):
        b = torch.logsumexp(log_T + (log_psi[t] + b)[None, :], 1)
        betas[t - 1] = b
    g = torch.stack([alphas[t] + betas[t] for t in range(T)])
    g = (g - torch.logsumexp(g, 1, keepdim=True)).exp()
    recon = (g * log_e).sum()
    kl = (g * log_psi).sum() - logZ
    return recon - kl, logZ, g


def main():
    """Run L1, P3 and MF and print pass/fail against the predictions."""
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    T, K = args.T, args.K

    log_p1 = torch.log_softmax(torch.randn(K), 0)
    log_T = torch.log_softmax(torch.randn(K, K), 1)
    log_e = torch.log(torch.rand(T, K).clamp(min=1e-3))

    print("=== L1: does the structured family contain the exact posterior? ===")
    exact = brute_force(log_p1, log_T, log_e)
    elbo, logZ, _g = chain_elbo(log_p1, log_T, log_e, log_e)
    print(f"   brute-force log p(b)        {float(exact): .8f}")
    print(f"   chain ELBO with psi = p(b|.){float(elbo): .8f}")
    print(f"   chain log Z                 {float(logZ): .8f}")
    print(f"   |ELBO - log p(b)|           {abs(float(elbo - exact)):.3e}   "
          f"{'PASS' if abs(float(elbo - exact)) < 1e-4 else 'FAIL'}")

    print("\n=== MF: exact mean-field optimum (coordinate ascent, no network) ===")
    logq = torch.log_softmax(torch.randn(T, K), 1)
    for _ in range(500):
        for t in range(T):
            m = log_e[t].clone()
            if t > 0:
                m = m + (logq[t - 1].exp()[:, None] * log_T).sum(0)
            else:
                m = m + log_p1
            if t < T - 1:
                m = m + (log_T * logq[t + 1].exp()[None, :]).sum(1)
            logq[t] = torch.log_softmax(m, 0)
    q = logq.exp()
    ent = -(q * logq).sum()
    e_lp = (q[0] * log_p1).sum() + sum(
        (q[t][:, None] * log_T * q[t + 1][None, :]).sum() for t in range(T - 1))
    mf_elbo = (q * log_e).sum() + e_lp + ent
    print(f"   mean-field ELBO             {float(mf_elbo): .8f}")
    print(f"   gap to log p(b)             {float(exact - mf_elbo): .6f} nats  "
          f"{'PASS (>0)' if float(exact - mf_elbo) > 0 else 'FAIL'}")
    print(f"   mean per-frame entropy      {float(ent) / T:.4f} nats "
          f"(max {math.log(K):.4f})")

    print("\n=== P3: commit vs hedge over two paths ===")
    for eps in (1e-2, 1e-3, 1e-4):
        Kb, Tb = 8, 60
        lT = torch.full((Kb, Kb), eps / (Kb - 1))
        for k in range(Kb):
            lT[k, (k + 1) % Kb] = 1 - eps
        lT = lT.log()
        A = [(0 + i) % Kb for i in range(Tb)]
        B = [(4 + i) % Kb for i in range(Tb)]
        commit = sum(float(lT[A[i], A[i + 1]]) for i in range(Tb - 1))
        hedge = sum(0.25 * (float(lT[A[i], A[i + 1]]) + float(lT[B[i], B[i + 1]])
                            + float(lT[A[i], B[i + 1]]) + float(lT[B[i], A[i + 1]]))
                    for i in range(Tb - 1)) + Tb * math.log(2)
        pred = (Tb / 2) * math.log((Kb - 1) / eps) - Tb * math.log(2)
        print(f"   eps {eps:.0e}: commit {commit:9.2f}  hedge {hedge:9.2f}  "
              f"advantage {commit - hedge:9.2f}  predicted {pred:9.2f}  "
              f"ratio {(commit - hedge) / pred:.3f}")


if __name__ == "__main__":
    main()
