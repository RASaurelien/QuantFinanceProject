"""
experiment.py
===============

Four experiments built on execution.py's validated QP solver.

1. Trajectory comparison: Almgren-Chriss closed form vs transient-impact
   optimal trajectories for a range of decay rates rho.
2. The Obizhaeva-Wang "block trade" phenomenon. For slowly-decaying
   (persistent) impact, the optimal discretized trajectory places large
   trades at the start and end of the execution window, with much
   smaller, near-uniform trading in between -- a textbook qualitative
   prediction of the OW continuous-time theory, which we did not
   hard-code anywhere. It falls out of the generic QP solver on its own.
3. Cost of model misspecification. How much more does it cost, under
   the true transient kernel, to follow the Almgren-Chriss-optimal
   trajectory (which ignores transience) instead of the trajectory
   optimized for the correct kernel?
4. A concrete numerical check of Gatheral's no-dynamic-arbitrage
   condition for the exponential kernel: round-trip trades must not
   generate negative expected cost.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execution import (
    diagonal_kernel, exponential_kernel, solve_optimal_trajectory,
    holdings_from_trades, almgren_chriss_closed_form, cumulative_matrix,
)

X, T, N = 100_000.0, 1.0, 20
ETA, GAMMA, SIGMA, LAM = 0.05, 0.001, 0.3, 2e-6
TAU = T / N
KAPPA0 = ETA / TAU
T_MID = (np.arange(N) + 0.5) * TAU


def impact_cost(n: np.ndarray, K: np.ndarray) -> float:
    return 0.5 * n @ K @ n


# x_1..x_N, i.e. post-trade holdings each period.
def variance_cost(n: np.ndarray, X: float, sigma: float, tau: float) -> float:
    x = holdings_from_trades(n, X)[1:]
    return sigma ** 2 * tau * np.sum(x ** 2)


def main():
    # ------------------------------------------------------------------
    # 1 & 2. Trajectories and the block-trade phenomenon
    # ------------------------------------------------------------------
    rhos = [0.5, 2.0, 10.0, 1000.0]
    K_diag = diagonal_kernel(N, ETA, TAU)
    n_ac = solve_optimal_trajectory(K_diag, X, LAM, SIGMA, TAU)
    x_ac = holdings_from_trades(n_ac, X)

    trajectories = {"Almgren-Chriss (closed form)": almgren_chriss_closed_form(X, T, N, ETA, GAMMA, SIGMA, LAM)}
    trade_profiles = {}
    for rho in rhos:
        K = exponential_kernel(T_MID, KAPPA0, rho)
        n = solve_optimal_trajectory(K, X, LAM, SIGMA, TAU)
        trajectories[f"Transient, rho={rho}"] = holdings_from_trades(n, X)
        trade_profiles[rho] = n

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    t_grid = np.linspace(0, T, N + 1)
    for label, x in trajectories.items():
        axes[0].plot(t_grid, x, marker="o", markersize=3, label=label)
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("remaining shares to sell")
    axes[0].set_title("Optimal liquidation trajectories")
    axes[0].legend(fontsize=8)

    width = TAU * 0.8 / len(rhos)
    for i, rho in enumerate(rhos):
        offset = (i - (len(rhos) - 1) / 2) * width
        axes[1].bar(T_MID + offset, trade_profiles[rho], width=width, label=f"rho={rho}")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("shares traded in interval")
    axes[1].set_title("Trade-size profile: the OW 'block trade' effect\n(slow decay -> big trades at both ends)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("trajectories.png", dpi=150)
    print("Saved trajectories.png")

    print("=== Block-trade phenomenon: first-interval trade as % of total X ===")
    for rho in rhos:
        pct = 100 * trade_profiles[rho][0] / X
        print(f"  rho={rho:>8.1f}: first trade = {pct:5.1f}% of X"
              f"  (uniform/TWAP baseline = {100/N:.1f}%)")

    # ------------------------------------------------------------------
    # 3. Cost of ignoring transience. Apply the AC-optimal
    # (diagonal-kernel) trajectory under the true transient kernel,
    # and compare against the trajectory that was actually optimized
    # for that kernel.
    # ------------------------------------------------------------------
    print("\n=== Cost of model misspecification (impact cost only, in $) ===")
    print(f"{'rho':>8}{'cost of correctly-optimized n':>32}{'cost of AC-optimal n under true K':>36}{'extra cost':>14}")
    for rho in rhos:
        K_true = exponential_kernel(T_MID, KAPPA0, rho)
        n_correct = solve_optimal_trajectory(K_true, X, LAM, SIGMA, TAU)
        cost_correct = impact_cost(n_correct, K_true)
        cost_misspecified = impact_cost(n_ac, K_true)  # AC's trajectory, priced under the true kernel
        extra = cost_misspecified - cost_correct
        print(f"{rho:>8.1f}{cost_correct:>32,.0f}{cost_misspecified:>36,.0f}{extra:>14,.0f}"
              f"   (+{100*extra/cost_correct:.1f}%)")

    # ------------------------------------------------------------------
    # 4. No-dynamic-arbitrage check. A round trip (buy y then sell y
    # after a lag Delta) must have non-negative expected impact cost
    # under a well-specified kernel -- Gatheral 2010's no-dynamic-
    # arbitrage condition, violated by some naive kernel choices, e.g.
    # kernels that increase over some range. We check it directly for
    # our exponential kernel across a range of lags.
    # ------------------------------------------------------------------
    print("\n=== No-dynamic-arbitrage check (exponential kernel) ===")
    print("Round trip: buy y at t=0, sell y at t=Delta. Expected cost should be >= 0 for all Delta > 0.")
    y = 1000.0
    rho_check = 5.0
    deltas = np.linspace(0.01, 2.0, 50)
    min_cost = np.inf
    for d in deltas:
        # cost = 0.5*[G(0)*y^2 + G(0)*y^2 + 2*G(d)*y*(-y)] = y^2*(G(0) - G(d))
        G0 = KAPPA0  # G(0) = kappa0 * exp(0)
        Gd = KAPPA0 * np.exp(-rho_check * d)
        cost = y ** 2 * (G0 - Gd)
        min_cost = min(min_cost, cost)
    print(f"  minimum round-trip cost over Delta in [0.01, 2.0]: {min_cost:,.2f}  "
          f"({'>= 0, OK -- no arbitrage detected' if min_cost >= -1e-6 else 'NEGATIVE -- ARBITRAGE!'})")
    print("  (holds because G(t)=kappa0*exp(-rho*t) is positive, decreasing and convex,")
    print("   the standard sufficient condition from Gatheral (2010) / Alfonsi-Schied for")
    print("   exponentially-decaying kernels to rule out round-trip price manipulation.)")

    # ------------------------------------------------------------------
    # Efficient frontier: expected impact cost vs variance, AC diagonal
    # kernel, sweeping the risk-aversion parameter lambda.
    # ------------------------------------------------------------------
    lambdas = np.logspace(-6, 1.3, 30)
    costs, variances = [], []
    for lam in lambdas:
        n = solve_optimal_trajectory(K_diag, X, lam, SIGMA, TAU)
        costs.append(impact_cost(n, K_diag))
        variances.append(variance_cost(n, X, SIGMA, TAU))
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.plot(np.sqrt(variances), np.array(costs) / 1e6, "-o", markersize=3, color="#C44E52")
    ax2.set_xlabel("std. dev. of implementation shortfall ($)")
    ax2.set_ylabel("expected impact cost ($ millions)")
    ax2.set_title("Almgren-Chriss efficient frontier")
    fig2.tight_layout()
    fig2.savefig("efficient_frontier.png", dpi=150)
    print("\nSaved efficient_frontier.png")


if __name__ == "__main__":
    main()
