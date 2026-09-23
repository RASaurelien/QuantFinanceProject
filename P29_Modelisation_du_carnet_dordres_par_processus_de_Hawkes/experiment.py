"""
experiment.py
===============

Applies the validated Hawkes machinery (hawkes.py) to a stylized
order-flow model: two event types, BUY and SELL market-order arrivals,
with self-excitation on each side (order-flow clustering / herding, a
well documented stylized fact of high-frequency markets, see
Bacry-Muzy 2015) and cross-excitation (a buy can trigger further sells
and vice versa, e.g. through market-making / mean-reversion flow
reacting to the trade).

Three things are demonstrated:
1. MLE recovers the true (mu, alpha, beta) from simulated order flow.
2. The Hawkes process reproduces empirical clustering of trade
   arrivals (over-dispersion relative to a Poisson process of the same
   mean rate), measured via the Fano factor / index of dispersion, a
   standard microstructure diagnostic.
3. As the branching ratio (spectral radius of alpha) approaches the
   stability boundary of 1, clustering and the Fano factor blow up --
   the near-critical regime that empirical fits to real order flow are
   often found close to (Bacry-Muzy report branching ratios around
   0.5-0.9 for real futures/equity order flow).
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes import MultivariateHawkes, fit_mle

MU_TRUE = np.array([0.4, 0.4])
ALPHA_TRUE = np.array([[0.35, 0.15],
                        [0.15, 0.35]])
BETA_TRUE = 3.0
T_SIM = 15000.0


# Index of dispersion Var(N(window)) / E[N(window)], over non-overlapping
# windows.
def fano_factor(times: np.ndarray, T: float, window: float) -> float:
    edges = np.arange(0, T, window)
    counts, _ = np.histogram(times, bins=edges)
    if counts.mean() == 0:
        return np.nan
    return counts.var() / counts.mean()


def simulate_poisson(rate: float, T: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_expected = int(rate * T * 1.2) + 100
    dt = rng.exponential(1.0 / rate, size=n_expected)
    t = np.cumsum(dt)
    return t[t < T]


def main():
    model = MultivariateHawkes(MU_TRUE, ALPHA_TRUE, BETA_TRUE)
    print("Branching ratio (spectral radius of alpha):", f"{model.branching_ratio_spectral_radius():.3f}")
    print("Theoretical stationary mean intensity:", model.theoretical_mean_intensity())

    times, dims = model.simulate(T_SIM, seed=7)
    n_buy, n_sell = np.sum(dims == 0), np.sum(dims == 1)
    print(f"Simulated {len(times)} events over T={T_SIM} (buy={n_buy}, sell={n_sell})")
    print("Empirical mean rates:", [n_buy / T_SIM, n_sell / T_SIM])

    # ------------------------------------------------------------------
    # 1. MLE fit
    # ------------------------------------------------------------------
    print("\n=== MLE fit ===")
    mu_hat, alpha_hat, beta_hat, res = fit_mle(times, dims, T_SIM, M=2, seed=1)
    print(f"{'':>12}{'true':>10}{'recovered':>12}")
    print(f"{'mu_buy':>12}{MU_TRUE[0]:>10.4f}{mu_hat[0]:>12.4f}")
    print(f"{'mu_sell':>12}{MU_TRUE[1]:>10.4f}{mu_hat[1]:>12.4f}")
    print(f"{'alpha_bb':>12}{ALPHA_TRUE[0,0]:>10.4f}{alpha_hat[0,0]:>12.4f}")
    print(f"{'alpha_bs':>12}{ALPHA_TRUE[0,1]:>10.4f}{alpha_hat[0,1]:>12.4f}")
    print(f"{'alpha_sb':>12}{ALPHA_TRUE[1,0]:>10.4f}{alpha_hat[1,0]:>12.4f}")
    print(f"{'alpha_ss':>12}{ALPHA_TRUE[1,1]:>10.4f}{alpha_hat[1,1]:>12.4f}")
    print(f"{'beta':>12}{BETA_TRUE:>10.4f}{beta_hat:>12.4f}")

    # ------------------------------------------------------------------
    # 2. Intensity path snapshot (short window), to visualize clustering
    # ------------------------------------------------------------------
    window_start, window_end = 100.0, 130.0
    grid = np.linspace(window_start, window_end, 3000)
    lam_buy, lam_sell = np.zeros_like(grid), np.zeros_like(grid)
    mask = times < window_end
    t_hist, d_hist = times[mask], dims[mask]
    for i, tg in enumerate(grid):
        past = t_hist[t_hist < tg]
        dpast = d_hist[t_hist < tg]
        R = np.array([
            np.sum(np.exp(-BETA_TRUE * (tg - past[dpast == 0]))),
            np.sum(np.exp(-BETA_TRUE * (tg - past[dpast == 1]))),
        ])
        lam = MU_TRUE + BETA_TRUE * (ALPHA_TRUE @ R)
        lam_buy[i], lam_sell[i] = lam[0], lam[1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid, lam_buy, color="#4C72B0", label="lambda_buy(t)")
    ax.plot(grid, lam_sell, color="#C44E52", label="lambda_sell(t)")
    buy_t = times[(dims == 0) & (times >= window_start) & (times < window_end)]
    sell_t = times[(dims == 1) & (times >= window_start) & (times < window_end)]
    ax.scatter(buy_t, np.full_like(buy_t, -0.1), marker="|", color="#4C72B0", s=80, label="buy events")
    ax.scatter(sell_t, np.full_like(sell_t, -0.25), marker="|", color="#C44E52", s=80, label="sell events")
    ax.set_xlabel("time")
    ax.set_ylabel("intensity")
    ax.set_title("Hawkes order-flow intensity: self- and cross-excitation bursts")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig("intensity_path.png", dpi=150)
    print("\nSaved intensity_path.png")

    # ------------------------------------------------------------------
    # 3. Fano factor: Hawkes (clustered) vs Poisson (same mean rate)
    # ------------------------------------------------------------------
    total_rate = len(times) / T_SIM
    poisson_times = simulate_poisson(total_rate, T_SIM, seed=99)

    windows = np.array([0.5, 1, 2, 5, 10, 20, 50, 100])
    fano_hawkes = [fano_factor(times, T_SIM, w) for w in windows]
    fano_poisson = [fano_factor(poisson_times, T_SIM, w) for w in windows]

    print("\n=== Fano factor (index of dispersion): Hawkes (all events) vs Poisson, same mean rate ===")
    print(f"{'window':>10}{'Hawkes':>12}{'Poisson':>12}")
    for w, fh, fp in zip(windows, fano_hawkes, fano_poisson):
        print(f"{w:>10.1f}{fh:>12.3f}{fp:>12.3f}")

    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    ax2.plot(windows, fano_hawkes, "-o", color="#C44E52", label="Hawkes (order flow)")
    ax2.plot(windows, fano_poisson, "-o", color="#999999", label="Poisson (same mean rate)")
    ax2.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xscale("log")
    ax2.set_xlabel("window size")
    ax2.set_ylabel("Fano factor  Var(N)/E[N]")
    ax2.set_title("Clustering of order flow: Hawkes vs Poisson")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("fano_factor.png", dpi=150)
    print("Saved fano_factor.png")

    # ------------------------------------------------------------------
    # 4. Approaching criticality: scale alpha towards spectral radius -> 1
    # ------------------------------------------------------------------
    print("\n=== Approaching the critical branching ratio: effect on clustering ===")
    print(f"{'spectral radius':>18}{'Fano(window=5)':>18}{'mean rate':>14}")
    base_alpha = ALPHA_TRUE / model.branching_ratio_spectral_radius()  # normalize to radius 1
    for scale in [0.5, 0.7, 0.85, 0.95, 0.99]:
        a = base_alpha * scale
        m = MultivariateHawkes(MU_TRUE, a, BETA_TRUE)
        tt, dd = m.simulate(T_SIM, seed=5)
        f5 = fano_factor(tt, T_SIM, 5.0)
        rate = len(tt) / T_SIM
        print(f"{scale:>18.2f}{f5:>18.3f}{rate:>14.3f}")


if __name__ == "__main__":
    main()
