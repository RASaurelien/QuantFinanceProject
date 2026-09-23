"""
experiment.py
===============

Trains the deep hedging network (deep_hedging.py)
on simulated rough Bergomi price paths
(rough_bergomi.py, same validated simulator as project 1),
WITH proportional transaction costs.

It then compares the network out-of-sample
against the classical Black-Scholes delta hedge,
subjected to the SAME costs.

This is the setting where a fixed closed-form Greek is provably suboptimal:
transaction costs make the truly optimal hedge depend on the trading history,
something no single Greek captures.
It is also where a learned policy has clear room to add value.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rough_bergomi import simulate_rough_bergomi
from deep_hedging import DeepHedge

# ----------------------------------------------------------------------
N_PATHS = 40000
N_STEPS = 30
T = 0.25
S0 = 100.0
K = 100.0
XI0 = 0.04
H = 0.15
ETA = 1.3
RHO = -0.7
COST_RATE = 0.010          # 100 bp of notional per unit of turnover, per rebalance
LAM = 8.0                  # entropic risk aversion parameter
N_HIDDEN = 16
N_EPOCHS = 60
BATCH_SIZE = 2000
LR = 5e-3
TRAIN_FRAC = 0.75
SEED = 0


def bs_call_delta(S, K, tau, sigma):
    tau = np.maximum(tau, 1e-12)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def bs_delta_hedge_pnl(t, S, K, cost_rate, sigma):
    N, n = S.shape[0], S.shape[1] - 1
    T_ = t[-1]
    theta_prev = np.zeros(N)
    wealth = np.zeros(N)
    for i in range(n):
        tau = T_ - t[i]
        theta_i = bs_call_delta(S[:, i], K, tau, sigma)
        dS = S[:, i + 1] - S[:, i]
        cost_i = cost_rate * np.abs(theta_i - theta_prev) * S[:, i]
        wealth += theta_i * dS - cost_i
        theta_prev = theta_i
    return wealth


def entropic_risk(err, lam):
    w = np.exp(lam * (err - err.max()))
    return err.max() + np.log(w.mean()) / lam


def main():
    print(f"Simulating {N_PATHS} rough Bergomi paths, "
          f"{N_STEPS} steps, T={T} ...")
    t, S, V = simulate_rough_bergomi(
        n_paths=N_PATHS, n_steps=N_STEPS, T=T, S0=S0, xi0=XI0,
        H=H, eta=ETA, rho=RHO, seed=SEED
    )
    payoff = np.maximum(S[:, -1] - K, 0.0)

    n_train = int(TRAIN_FRAC * N_PATHS)
    rng = np.random.default_rng(0)
    idx = rng.permutation(N_PATHS)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    # Fair premium: Monte Carlo mean payoff
    # under the (already risk-neutral, driftless-S) simulated dynamics,
    # fixed once on the FULL sample.
    p0 = payoff.mean()
    print(f"Fair premium p0 (MC mean payoff) = {p0:.4f}")

    # ------------------------------------------------------------------
    # Train the deep hedge
    # ------------------------------------------------------------------
    model = DeepHedge(n_steps=N_STEPS, n_hidden=N_HIDDEN, seed=1)
    print(f"\nTraining deep hedge: {N_EPOCHS} epochs, batch={BATCH_SIZE}, "
          f"lr={LR}, cost_rate={COST_RATE}, lambda={LAM}")
    loss_history = model.train(
        t, S[train_idx], K, COST_RATE, payoff[train_idx], p0, LAM,
        n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LR, verbose_every=10
    )

    # ------------------------------------------------------------------
    # Out-of-sample evaluation
    # ------------------------------------------------------------------
    _, wealth_dh, _ = model.forward(t, S[test_idx], K, COST_RATE)
    err_dh = payoff[test_idx] - (p0 + wealth_dh)

    sigma_bs = np.sqrt(XI0)  # flat vol a BS hedger would plausibly assume
    wealth_bs = bs_delta_hedge_pnl(t, S[test_idx], K, COST_RATE, sigma_bs)
    err_bs = payoff[test_idx] - (p0 + wealth_bs)

    err_unhedged = payoff[test_idx] - p0

    def report(name, err):
        return dict(name=name, mean=err.mean(), std=err.std(),
                    entropic=entropic_risk(err, LAM),
                    cvar95=err[err >= np.quantile(err, 0.95)].mean())

    rows = [report("Unhedged", err_unhedged),
            report("BS delta (with costs)", err_bs),
            report("Deep hedge (with costs)", err_dh)]

    print(f"\n=== Out-of-sample hedging error e = payoff - (p0 + P&L), "
          f"n={len(test_idx)} ===")
    print(f"{'strategy':<24}{'mean':>9}{'std':>9}"
          f"{'entropic risk':>16}{'CVaR 95%':>12}")
    for r in rows:
        print(f"{r['name']:<24}{r['mean']:>9.4f}{r['std']:>9.4f}"
              f"{r['entropic']:>16.4f}{r['cvar95']:>12.4f}")

    # ------------------------------------------------------------------
    # Also compare WITHOUT transaction costs.
    # This is a sanity check that the network is at least competitive
    # with the BS delta when frictions are absent
    # (the regime where BS delta is close to optimal, see project 1).
    # ------------------------------------------------------------------
    print("\n--- Sanity check: same comparison but cost_rate=0 ---")
    model_nc = DeepHedge(n_steps=N_STEPS, n_hidden=N_HIDDEN, seed=2)
    model_nc.train(t, S[train_idx], K, 0.0, payoff[train_idx], p0, LAM,
                   n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                   verbose_every=0)
    _, wealth_dh_nc, _ = model_nc.forward(t, S[test_idx], K, 0.0)
    err_dh_nc = payoff[test_idx] - (p0 + wealth_dh_nc)
    wealth_bs_nc = bs_delta_hedge_pnl(t, S[test_idx], K, 0.0, sigma_bs)
    err_bs_nc = payoff[test_idx] - (p0 + wealth_bs_nc)
    print(f"  BS delta   (no costs): std={err_bs_nc.std():.4f}  "
          f"entropic={entropic_risk(err_bs_nc, LAM):.4f}")
    print(f"  Deep hedge (no costs): std={err_dh_nc.std():.4f}  "
          f"entropic={entropic_risk(err_dh_nc, LAM):.4f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(loss_history)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("entropic-risk training loss")
    axes[0].set_title("Deep hedge training curve")

    bins = np.linspace(
        min(err_unhedged.min(), err_bs.min(), err_dh.min()),
        max(err_unhedged.max(), err_bs.max(), err_dh.max()), 60
    )
    axes[1].hist(err_unhedged, bins=bins, alpha=0.4, color="#999999",
                 label=f"Unhedged (std={err_unhedged.std():.2f})")
    axes[1].hist(err_bs, bins=bins, alpha=0.55, color="#4C72B0",
                 label=f"BS delta + costs (std={err_bs.std():.2f})")
    axes[1].hist(err_dh, bins=bins, alpha=0.55, color="#C44E52",
                 label=f"Deep hedge + costs (std={err_dh.std():.2f})")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("hedging error e (test set)")
    axes[1].set_title("Out-of-sample hedging error, with transaction costs")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("deep_hedging_results.png", dpi=150)
    print("\nSaved plot to deep_hedging_results.png")

    return rows


if __name__ == "__main__":
    main()
