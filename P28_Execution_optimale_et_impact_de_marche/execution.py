"""
execution.py
=============

Optimal execution with market impact. Implements the Almgren & Chriss
(2001, "Optimal Execution of Portfolio Transactions", Journal of Risk 3)
closed-form mean-variance liquidation trajectory, plus a general
quadratic-program solver that handles any symmetric positive-definite
impact kernel K, not just Almgren-Chriss's memoryless temporary-impact
cost. In particular this covers the transient (decaying) impact kernel
of Obizhaeva & Wang (2013, "Optimal Trading Strategy and Supply/Demand
Dynamics", Journal of Financial Markets 16(1)) and Gatheral (2010,
"No-Dynamic-Arbitrage and Market Impact", Quantitative Finance 10(7)).

The Almgren-Chriss closed form is recovered as the special diagonal
case of the general solver -- that's the main correctness check used
here (see __main__ below).

Setup: discrete time, liquidating X > 0 shares over N intervals of
length tau = T/N.
  - n_k : shares traded in interval k (k = 1..N), sum_k n_k = X
  - x_k : remaining holdings after trade k, x_0 = X, x_N = 0,
    x_k = X - sum_{i<=k} n_i
  - Permanent impact (linear, price-moving, persists forever) contributes
    a constant cost (gamma/2) X^2, independent of trajectory shape, for
    a linear permanent-impact model -- a standard Almgren-Chriss result.
    It therefore never affects the optimal SHAPE of the trajectory, and
    is added back only as a constant in the reported total cost, not
    inside the optimization.
  - Impact cost (temporary/transient) is a quadratic form in the trade
    vector n: Cost_impact(n) = 0.5 * n^T K n. For Almgren-Chriss,
    K = diag(eta/tau): each trade only costs against itself, impact
    fully reverts before the next trade. For Obizhaeva-Wang / Gatheral,
    K_{jk} = kappa0 * exp(-rho * |t_j - t_k|), a trade's impact decays
    but hasn't fully reverted by the time of later trades -- coupling
    every pair of trades through a Toeplitz exponential kernel.
  - Risk is the variance of the implementation-shortfall cost from
    holding the (volatile) remaining position while it's being unwound:
    Var(n) = sigma^2 * tau * sum_k x_k^2 = sigma^2 * tau * ||X*e - L n||^2,
    where L is the lower-triangular cumulative-sum ("running total")
    matrix, L[k, j] = 1 if j <= k else 0, and e = (1,...,1).

Mean-variance objective (Almgren-Chriss's own risk-aversion
formulation):

    J(n) = 0.5 n^T K n + lambda * sigma^2 * tau * || X e - L n ||^2,
    subject to 1^T n = X

This is a linear-equality-constrained convex QP. A closed-form solution
follows directly from its KKT (Karush-Kuhn-Tucker) linear system,
solved here with a single np.linalg.solve call -- no external QP
library needed.
"""

from __future__ import annotations
import numpy as np


# Lower-triangular running-sum matrix L, L[k,j] = 1 for j <= k (0-indexed).
def cumulative_matrix(N: int) -> np.ndarray:
    return np.tril(np.ones((N, N)))


# Solves the equality-constrained QP
#   min_n  0.5 n^T K n + lam * sigma^2 * tau * || X*e - L n ||^2
#   s.t.   1^T n = X
# via its KKT linear system. Returns the optimal trade vector n (N,).
def solve_optimal_trajectory(K: np.ndarray, X: float, lam: float, sigma: float, tau: float) -> np.ndarray:
    N = K.shape[0]
    L = cumulative_matrix(N)
    e = np.ones(N)

    # Hessian of the (unconstrained) quadratic, then the linear term
    # (gradient of the X*e part).
    H = K + 2.0 * lam * sigma ** 2 * tau * (L.T @ L)
    b = 2.0 * lam * sigma ** 2 * tau * (L.T @ (X * e))

    # KKT system: [H, 1; 1^T, 0] [n; mu] = [b; X]
    A = np.zeros((N + 1, N + 1))
    A[:N, :N] = H
    A[:N, N] = 1.0
    A[N, :N] = 1.0
    rhs = np.zeros(N + 1)
    rhs[:N] = b
    rhs[N] = X

    sol = np.linalg.solve(A, rhs)
    return sol[:N]


# x_0=X, x_1,...,x_N from cumulative sums of trades. Returns a length
# N+1 array including x_0.
def holdings_from_trades(n: np.ndarray, X: float) -> np.ndarray:
    N = len(n)
    L = cumulative_matrix(N)
    x_tail = X - L @ n
    return np.concatenate([[X], x_tail])


# Almgren-Chriss memoryless temporary-impact kernel: K = diag(eta/tau).
def diagonal_kernel(N: int, eta: float, tau: float) -> np.ndarray:
    return np.diag(np.full(N, eta / tau))


# Obizhaeva-Wang / Gatheral transient-impact kernel:
# K_{jk} = kappa0 * exp(-rho*|t_j-t_k|), evaluated at the midpoints of
# each trading interval (t_mid, length N). As rho -> infinity this
# converges to a diagonal matrix -- impact reverts instantly, i.e. the
# Almgren-Chriss limit -- which is checked numerically below.
def exponential_kernel(t_mid: np.ndarray, kappa0: float, rho: float) -> np.ndarray:
    diff = np.abs(t_mid[:, None] - t_mid[None, :])
    return kappa0 * np.exp(-rho * diff)


# ----------------------------------------------------------------------
# Almgren-Chriss closed-form solution (for validation)
# ----------------------------------------------------------------------
# Classical closed-form AC trajectory (their 2001 paper, section on the
# discrete-time solution). Remaining holdings:
#
#   x_j = X * sinh(kappa*(T - t_j)) / sinh(kappa*T),   j = 0..N
#
# where kappa solves cosh(kappa*tau) = 1 + tau^2 * kappa_tilde^2 / 2,
# and kappa_tilde^2 = lam * sigma^2 / eta_tilde,
# eta_tilde = eta*(1 - gamma*tau/(2*eta)).
# Returns holdings x_0..x_N (length N+1).
def almgren_chriss_closed_form(X: float, T: float, N: int, eta: float, gamma: float,
                                sigma: float, lam: float) -> np.ndarray:
    tau = T / N
    t = np.linspace(0, T, N + 1)
    eta_tilde = eta * (1.0 - gamma * tau / (2.0 * eta))
    kappa_tilde2 = lam * sigma ** 2 / eta_tilde

    # Solve cosh(kappa*tau) = 1 + tau^2*kappa_tilde2/2 for kappa.
    rhs = 1.0 + tau ** 2 * kappa_tilde2 / 2.0
    kappa = np.arccosh(rhs) / tau if rhs > 1.0 else np.sqrt(kappa_tilde2)  # rhs->1 limit: kappa->sqrt(kappa_tilde2)

    if lam == 0.0:
        # Risk-neutral limit: linear liquidation, a well-known AC
        # special case.
        return X * (1 - t / T)

    x = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
    return x


if __name__ == "__main__":
    # Validation 1: with a diagonal kernel (pure Almgren-Chriss
    # temporary impact, no transience), the general QP solver must
    # reproduce the classical closed-form sinh trajectory exactly.
    X, T, N = 100_000.0, 1.0, 20
    eta, gamma, sigma, lam = 0.05, 0.001, 0.3, 2e-6
    tau = T / N

    K_diag = diagonal_kernel(N, eta, tau)
    n_qp = solve_optimal_trajectory(K_diag, X, lam, sigma, tau)
    x_qp = holdings_from_trades(n_qp, X)
    x_closed = almgren_chriss_closed_form(X, T, N, eta, gamma, sigma, lam)

    rel_err = np.max(np.abs(x_qp - x_closed)) / X
    print("Validation 1 -- general QP solver vs Almgren-Chriss closed form (diagonal kernel):")
    print(f"  max abs difference in holdings (as fraction of X): {rel_err:.2e}")
    assert rel_err < 1e-6, "QP solver does not match Almgren-Chriss closed form!"
    print("  PASSED\n")

    # Validation 2: risk-neutral limit (lambda=0) with a diagonal
    # kernel should give uniform trading. Minimizing sum n_k^2 subject
    # to sum=X is minimized, by Cauchy-Schwarz, at n_k = X/N for all k.
    n_neutral = solve_optimal_trajectory(K_diag, X, 0.0, sigma, tau)
    uniform = np.full(N, X / N)
    rel_err2 = np.max(np.abs(n_neutral - uniform)) / (X / N)
    print("Validation 2 -- risk-neutral (lambda=0) limit gives uniform trading (TWAP):")
    print(f"  max relative deviation from X/N: {rel_err2:.2e}")
    assert rel_err2 < 1e-6, "Risk-neutral limit is not uniform trading!"
    print("  PASSED\n")

    # Validation 3: transient (exponential) kernel with rho -> infinity
    # should converge to the diagonal Almgren-Chriss kernel's solution,
    # since impact reverts before the next trade "sees" it.
    t_mid = (np.arange(N) + 0.5) * tau
    kappa0 = eta / tau  # match AC's diagonal scale so the rho->inf limit lines up exactly
    for rho in [1.0, 10.0, 100.0, 10000.0]:
        K_exp = exponential_kernel(t_mid, kappa0, rho)
        n_exp = solve_optimal_trajectory(K_exp, X, lam, sigma, tau)
        diff = np.max(np.abs(n_exp - n_qp)) / (X / N)
        print(f"  rho={rho:>8.1f}: max relative trade-size deviation from AC diagonal solution = {diff:.4f}")
    print("  (should shrink towards 0 as rho grows -- fast decay ~ instantaneous reversion ~ AC)")
