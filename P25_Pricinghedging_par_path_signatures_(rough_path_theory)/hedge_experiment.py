"""
hedge_experiment.py
====================

Main experiment of the project: does using the (truncated) path signature
as a feature set for the hedge ratio reduce out-of-sample hedging
variance, relative to classical Black-Scholes delta hedging, when the
true world has rough stochastic volatility (rough Bergomi)?

Method: global linear-in-signature hedging
-------------------------------------------
Following Lyons, Nejad & Perez Arribas (2020, "Non-parametric pricing and
hedging of exotic derivatives", Applied Mathematical Finance 27(6),
arXiv:1905.01345) and the related linear-functionals-of-the-signature
idea in Abi Jaber & Gerard (2025, arXiv:2508.02759): posit that the
number of shares held over [t_i, t_i+1) is a linear functional of the
signature of the price path observed so far,

theta_i = < ell , Sig(X)_{[0,t_i]} >

for a single, time-independent coefficient vector ell shared across all
times (the time-dependence of the hedge gets absorbed into the signature
itself, since the path is augmented with a time channel, see below). The
discretized self-financing hedging P&L of a portfolio that starts with V0
cash and rebalances at every t_i is then

P&L_j  = V0 + sum_i theta_{i,j} * (S_{t_i+1,j} - S_{t_i,j})
        = V0 + ell . [ sum_i Sig(X)_{[0,t_i]} * (S_{t_i+1,j} - S_{t_i,j}) ]
        = V0 + ell . R_j

where R_j in R^{sig_dim} is a single feature vector per Monte Carlo path
(the signature-weighted sum of price increments). Because R_j is linear
in the (already nonlinear-in-price) signature features, minimizing the
expected squared hedging error

    E[ (payoff_j - V0 - ell.R_j)^2 ]

over (V0, ell) is an ordinary least squares problem: no dynamic
programming, no neural network, just a single regression, yet the
resulting hedge is a genuinely nonlinear, path-dependent function of the
realized price path, because R_j itself is built from a nonlinear feature
map of the path. That's what makes the signature approach a decent
research baseline: it's "deep hedging lite", convex, with a closed-form
optimum.

Benchmark: classical Black-Scholes delta hedge with a single constant
volatility, i.e. the best the classical toolbox can do once you refuse to
assume a fully specified stochastic-vol model. We measure how much of the
residual variance left behind by the (misspecified, constant-vol) BS
delta can be recovered by the signature regression, purely from observing
the realized price path, without ever specifying or estimating the
rough-vol model itself.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rough_bergomi import simulate_rough_bergomi
from signature import running_signature_features


# ----------------------------------------------------------------------
# 1. Market / model parameters
# ----------------------------------------------------------------------
N_PATHS   = 30000
N_STEPS   = 50
T         = 0.25          # 3-month option
S0        = 100.0
K         = 100.0         # at-the-money call
XI0       = 0.04          # forward variance level -> sigma ~ 20%
H         = 0.10          # Hurst exponent (rough regime, Gatheral et al. 2018)
ETA       = 1.5           # vol-of-vol
RHO       = -0.7          # leverage correlation
SIG_LEVEL = 4              # signature truncation level
RIDGE_LAMBDA = 1e-2        # L2 regularization on signature regression (applied to standardized features)
TRAIN_FRAC = 0.7
SEED = 42


def bs_call_price(S, K, tau, sigma, r=0.0):
    tau = np.maximum(tau, 1e-12)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def bs_call_delta(S, K, tau, sigma, r=0.0):
    tau = np.maximum(tau, 1e-12)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def main():
    print(f"Simulating {N_PATHS} rough Bergomi paths, {N_STEPS} steps, T={T} ...")
    t, S, V = simulate_rough_bergomi(
        n_paths=N_PATHS, n_steps=N_STEPS, T=T, S0=S0, xi0=XI0, H=H, eta=ETA, rho=RHO, seed=SEED
    )
    payoff = np.maximum(S[:, -1] - K, 0.0)
    dS = np.diff(S, axis=1)  # (N, n_steps), real price increments used for P&L

    # ------------------------------------------------------------------
    # 2. Signature features of the (time, log-return) path.
    #    Using log(S/S0) rather than raw S keeps signature terms at a
    #    well-scaled O(1) order of magnitude, matching how the
    #    signature-hedging literature augments paths in practice.
    # ------------------------------------------------------------------
    logret = np.log(S / S0)
    aug_path = np.stack([np.broadcast_to(t, (N_PATHS, N_STEPS + 1)), logret], axis=-1)
    print(f"Computing running signature (level {SIG_LEVEL}) ...")
    feats = running_signature_features(aug_path, level=SIG_LEVEL)  # (N, n+1, sig_dim)
    sig_dim = feats.shape[-1]
    print(f"Signature feature dimension: {sig_dim}")

    # R_j = sum_i Sig(t_i)_j * dS_{i,j}   (adapted: feature at t_i times increment i->i+1)
    feats_pretime = feats[:, :-1, :]          # signature at t_0 .. t_{n-1}
    R = np.einsum('nik,ni->nk', feats_pretime, dS)  # (N, sig_dim)

    # ------------------------------------------------------------------
    # 3. Train / test split, ridge regression: payoff ~ V0 + ell . R
    #    Signature levels sit on very different natural scales (level-1
    #    terms are O(price move), level-4 terms are O(price move^4 / 4!)),
    #    so each regressor column is standardized on the training set
    #    before ridge, otherwise the L2 penalty ends up over-penalizing
    #    the high-order, small-scale terms and the fit is dominated by
    #    level 1.
    # ------------------------------------------------------------------
    n_train = int(TRAIN_FRAC * N_PATHS)
    idx = np.random.default_rng(0).permutation(N_PATHS)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    mu, sd = R[train_idx].mean(axis=0), R[train_idx].std(axis=0)
    sd[sd == 0] = 1.0
    Rs = (R - mu) / sd

    X_train = np.concatenate([np.ones((n_train, 1)), Rs[train_idx]], axis=1)
    y_train = payoff[train_idx]
    reg = RIDGE_LAMBDA * np.eye(X_train.shape[1])
    reg[0, 0] = 0.0  # don't penalize the V0 intercept
    beta = np.linalg.solve(X_train.T @ X_train + reg, X_train.T @ y_train)
    V0_sig = beta[0]
    print(f"Signature hedge: fitted V0 = {V0_sig:.4f}")

    X_test = np.concatenate([np.ones((len(test_idx), 1)), Rs[test_idx]], axis=1)
    err_sig = payoff[test_idx] - X_test @ beta

    # ------------------------------------------------------------------
    # 4. Black-Scholes delta-hedge benchmark, constant vol = sqrt(xi0),
    #    i.e. the natural "flat implied vol" a modeler would use if they
    #    don't know / don't model the roughness of the true variance
    #    process.
    # ------------------------------------------------------------------
    sigma_bs = np.sqrt(XI0)
    V0_bs = bs_call_price(S0, K, T, sigma_bs)
    tau = T - t[:-1]
    delta_bs = bs_call_delta(S[:, :-1], K, tau[None, :], sigma_bs)  # (N, n_steps)
    pnl_bs = V0_bs + np.sum(delta_bs * dS, axis=1)
    err_bs = payoff - pnl_bs
    err_bs_test = err_bs[test_idx]

    # ------------------------------------------------------------------
    # 5. Unhedged benchmark (naked short option position, cash = fair BS premium)
    # ------------------------------------------------------------------
    err_unhedged_test = payoff[test_idx] - V0_bs

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
    def stats(x):
        return x.mean(), x.std()

    m_u, s_u = stats(err_unhedged_test)
    m_b, s_b = stats(err_bs_test)
    m_s, s_s = stats(err_sig)

    print("\n=== Out-of-sample hedging error (test set, n=%d) ===" % len(test_idx))
    print(f"{'strategy':<22}{'mean':>10}{'std':>12}{'var reduction vs unhedged':>28}")
    print(f"{'unhedged':<22}{m_u:>10.4f}{s_u:>12.4f}{'--':>28}")
    print(f"{'BS delta hedge':<22}{m_b:>10.4f}{s_b:>12.4f}{100*(1-(s_b/s_u)**2):>27.1f}%")
    print(f"{'signature hedge':<22}{m_s:>10.4f}{s_s:>12.4f}{100*(1-(s_s/s_u)**2):>27.1f}%")
    print(f"\nsignature hedge variance vs BS delta hedge variance: "
          f"{100*(1-(s_s/s_b)**2):.1f}% additional reduction")

    # ------------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(
        min(err_unhedged_test.min(), err_bs_test.min(), err_sig.min()),
        max(err_unhedged_test.max(), err_bs_test.max(), err_sig.max()),
        60,
    )
    ax.hist(err_unhedged_test, bins=bins, alpha=0.45, label=f"Unhedged (std={s_u:.3f})", color="#999999")
    ax.hist(err_bs_test, bins=bins, alpha=0.55, label=f"BS delta hedge (std={s_b:.3f})", color="#4C72B0")
    ax.hist(err_sig, bins=bins, alpha=0.55, label=f"Signature hedge (std={s_s:.3f})", color="#C44E52")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Terminal hedging P&L error")
    ax.set_ylabel("count (test set)")
    ax.set_title("Hedging error distributions under rough Bergomi (out-of-sample)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("hedging_error_distributions.png", dpi=150)
    print("\nSaved plot to hedging_error_distributions.png")

    return dict(m_u=m_u, s_u=s_u, m_b=m_b, s_b=s_b, m_s=m_s, s_s=s_s, sig_dim=sig_dim,
                n_train=n_train, n_test=len(test_idx))


if __name__ == "__main__":
    main()
