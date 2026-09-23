"""
hawkes.py
==========

Multivariate mutually-exciting Hawkes process with an exponential kernel
and a single, shared decay rate beta across all (source, target) pairs:

    lambda_m(t) = mu_m + sum_n sum_{t_k^n < t} alpha[m,n] * beta * exp(-beta*(t - t_k^n))

alpha[m,n] is the expected number of type-m "child" events directly
triggered by one type-n event (a branching-ratio matrix entry). The
process is stationary iff the spectral radius of alpha is < 1, in which
case the theoretical stationary mean intensity vector is the classical
Hawkes branching formula,

    Lambda = (I - alpha)^{-1} mu

used below as a closed-form check on both the simulator and the MLE fit.

Two references for the exact algorithms implemented here:
- Ogata (1981), "On Lewis' simulation method for point processes", IEEE
  Transactions on Information Theory 27(1), for the thinning algorithm.
- Bacry, Mastromatteo & Muzy (2015), "Hawkes Processes in Finance",
  Market Microstructure and Liquidity 1(1), arXiv:1502.04592, for the
  exact recursive log-likelihood for exponential kernels (their eq. for
  R_i), which avoids the naive O(N^2) double sum over event pairs.

Recursive state: because the decay rate beta is shared, the whole
M-dimensional kernel "memory" collapses to a single M-vector,

    R_n(t) = sum_{t_k^n < t} exp(-beta*(t - t_k^n))

updated in O(1) at every event (R decays by exp(-beta*dt), then the +1
from the new event is added to its own dimension's entry). So both
simulation and log-likelihood evaluation are O(N), not O(N^2) -- this
recursive trick is what makes exact simulation and MLE tractable at
realistic event counts.
"""

from __future__ import annotations
import numpy as np


class MultivariateHawkes:
    def __init__(self, mu: np.ndarray, alpha: np.ndarray, beta: float):
        self.mu = np.asarray(mu, dtype=float)
        self.alpha = np.asarray(alpha, dtype=float)
        self.beta = float(beta)
        self.M = len(mu)

    def branching_ratio_spectral_radius(self) -> float:
        return np.max(np.abs(np.linalg.eigvals(self.alpha)))

    # Lambda = (I - alpha)^{-1} mu, only valid if the process is stationary.
    def theoretical_mean_intensity(self) -> np.ndarray:
        I = np.eye(self.M)
        return np.linalg.solve(I - self.alpha, self.mu)

    # ------------------------------------------------------------------
    # Ogata's modified thinning algorithm. Returns (times, dims): times
    # is a sorted (N,) array of event times in [0, T], dims the
    # corresponding (N,) array of dimension indices in {0,...,M-1}.
    def simulate(self, T: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        M = self.M
        t = 0.0
        R = np.zeros(M)  # R[n] = recursive decayed sum of past type-n events
        times, dims = [], []

        while True:
            lam_vec = self.mu + self.beta * (self.alpha @ R)
            lam_bar = lam_vec.sum()
            if lam_bar <= 0:
                lam_bar = self.mu.sum()  # degenerate guard, shouldn't trigger with mu>0
            dt = rng.exponential(1.0 / lam_bar)
            t_candidate = t + dt
            if t_candidate > T:
                break

            R_decayed = R * np.exp(-self.beta * dt)
            lam_vec_c = self.mu + self.beta * (self.alpha @ R_decayed)
            lam_c_total = lam_vec_c.sum()

            u = rng.uniform()
            if u <= lam_c_total / lam_bar:
                probs = lam_vec_c / lam_c_total
                m = rng.choice(M, p=probs)
                times.append(t_candidate)
                dims.append(m)
                R = R_decayed
                R[m] += 1.0
            else:
                R = R_decayed
            t = t_candidate

        return np.array(times), np.array(dims, dtype=int)

    # ------------------------------------------------------------------
    # Exact log-likelihood via the recursive R_n(t) state, O(N).
    def log_likelihood(self, times: np.ndarray, dims: np.ndarray, T: float) -> float:
        M = self.M
        R = np.zeros(M)
        t_prev = 0.0
        ll_events = 0.0

        for ti, mi in zip(times, dims):
            dt = ti - t_prev
            R = R * np.exp(-self.beta * dt)
            lam_vec = self.mu + self.beta * (self.alpha @ R)
            lam_mi = max(lam_vec[mi], 1e-300)  # numerical floor, avoids log(0)
            ll_events += np.log(lam_mi)
            R[mi] += 1.0
            t_prev = ti

        # Integral term: sum_m mu_m*T + sum_n (sum_m alpha[m,n]) * S_n,
        # where S_n = sum_{t_k^n < T} (1 - exp(-beta*(T - t_k^n))).
        S = np.zeros(M)
        for n in range(M):
            tk = times[dims == n]
            S[n] = np.sum(1.0 - np.exp(-self.beta * (T - tk)))
        colsum_alpha = self.alpha.sum(axis=0)
        integral = self.mu.sum() * T + colsum_alpha @ S

        return ll_events - integral


# Maximum-likelihood estimation of (mu, alpha, beta) via L-BFGS-B on the
# negative log-likelihood. Positivity is enforced through simple lower
# bounds (mu, alpha, beta >= small epsilon) rather than a
# reparametrization -- kept simple and adequate at this scale, a handful
# of parameters and thousands of events.
def fit_mle(times: np.ndarray, dims: np.ndarray, T: float, M: int, x0=None, seed: int = 0):
    from scipy.optimize import minimize

    n_alpha = M * M
    eps = 1e-6

    def unpack(x):
        mu = x[:M]
        alpha = x[M:M + n_alpha].reshape(M, M)
        beta = x[-1]
        return mu, alpha, beta

    def neg_ll(x):
        mu, alpha, beta = unpack(x)
        model = MultivariateHawkes(mu, alpha, beta)
        return -model.log_likelihood(times, dims, T)

    if x0 is None:
        rng = np.random.default_rng(seed)
        mu0 = np.full(M, len(times) / (M * T) * 0.5)
        alpha0 = np.full((M, M), 0.1) + rng.uniform(-0.02, 0.02, size=(M, M))
        beta0 = 2.0
        x0 = np.concatenate([mu0, alpha0.ravel(), [beta0]])

    bounds = [(eps, None)] * M + [(0.0, None)] * n_alpha + [(eps, None)]
    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                    options=dict(maxiter=500, ftol=1e-12, gtol=1e-8))
    mu_hat, alpha_hat, beta_hat = unpack(res.x)
    return mu_hat, alpha_hat, beta_hat, res


if __name__ == "__main__":
    # Validation 1: univariate Hawkes (M=1). Simulate, then check the
    # empirical event rate against the closed-form stationary mean
    # intensity Lambda = mu / (1 - alpha).
    mu1, alpha1, beta1 = np.array([0.5]), np.array([[0.6]]), 2.0
    model1 = MultivariateHawkes(mu1, alpha1, beta1)
    T1 = 20000.0
    t1, d1 = model1.simulate(T1, seed=42)
    empirical_rate = len(t1) / T1
    theoretical_rate = model1.theoretical_mean_intensity()[0]
    print("Validation 1 -- univariate Hawkes: empirical vs theoretical mean rate")
    print(f"  branching ratio (spectral radius) = {model1.branching_ratio_spectral_radius():.3f}  (must be < 1 for stationarity)")
    print(f"  empirical event rate   = {empirical_rate:.4f} events/unit time")
    print(f"  theoretical Lambda     = {theoretical_rate:.4f} events/unit time  (mu/(1-alpha))")
    print(f"  relative error         = {abs(empirical_rate-theoretical_rate)/theoretical_rate:.2%}\n")

    # Validation 2: MLE recovers the true univariate parameters from a
    # long simulated sample.
    print("Validation 2 -- MLE recovery, univariate Hawkes (true mu=0.5, alpha=0.6, beta=2.0):")
    mu_hat, alpha_hat, beta_hat, res = fit_mle(t1, d1, T1, M=1, seed=1)
    print(f"  recovered: mu={mu_hat[0]:.4f}  alpha={alpha_hat[0,0]:.4f}  beta={beta_hat:.4f}")
    print(f"  optimizer converged: {res.success}, nit={res.nit}\n")
