"""
main.py
=======
Scanne l'univers, sélectionne les meilleures paires, 
backteste CHAQUE paire avec ET sans gestion du risque 
(stop-loss + sizing par vol), 
et compare, pour démontrer que la gestion du risque a un effet mesurable,
pas juste "c'est plus prudent en théorie".
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from universe import simulate_universe, COINTEGRATED_PAIRS
from scanner import scan_pairs
from strategy import run_pair_strategy


def sharpe(pnl: np.ndarray) -> float:
    return 0.0 if pnl.std() < 1e-12 else pnl.mean() / pnl.std() * np.sqrt(252)


def max_drawdown(cum_pnl: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum_pnl)
    return float(np.max(peak - cum_pnl))


def main():
    import os
    os.makedirs("../P22_Pairs_trading_mean_reversion_(scan_univers_et_gestion du risque)/outputs", exist_ok=True)

    print("=" * 60)
    print(" Pairs Trading Mean-Reversion, scan d'univers + gestion du risque")
    print("=" * 60 + "\n")

    prices = simulate_universe()
    print(f"Univers simulé : {prices.shape[1]} actifs, {prices.shape[0]} jours "
          f"({len(COINTEGRATED_PAIRS)} vraies paires cointégrées noyées dans l'univers)\n")

    scan = scan_pairs(prices, top_n=5)
    print("--- Scan de cointégration (top 5 paires, classées par t-stat ADF) ---")
    print(f"{'Paire':<14}{'beta':>8}{'t-stat ADF':>12}  Vraie paire ?")
    for _, row in scan.iterrows():
        i, j = int(row["asset_i"]), int(row["asset_j"])
        is_true = (i, j) in COINTEGRATED_PAIRS or (j, i) in COINTEGRATED_PAIRS
        print(f"asset_{i}-asset_{j:<7}{row['beta']:>8.3f}{row['adf_tstat']:>12.2f}  "
              f"{'OUI' if is_true else 'non'}")
    n_true_found = sum((int(r.asset_i), int(r.asset_j)) in COINTEGRATED_PAIRS for _, r in scan.iterrows())
    print(f"\n-> {n_true_found}/{len(COINTEGRATED_PAIRS)} vraies paires retrouvées dans le top 5 "
          f"({scan.shape[0]} paires testées sur {prices.shape[1]*(prices.shape[1]-1)//2} au total)\n")

    print("--- Backtest par paire : avec vs sans gestion du risque ---\n")
    print(f"{'Paire':<14}{'Sharpe (sans)':>15}{'MaxDD (sans)':>14}{'Sharpe (avec)':>15}{'MaxDD (avec)':>14}{'Stops':>8}")

    total_pnl_managed = np.zeros(prices.shape[0])
    total_pnl_naive = np.zeros(prices.shape[0])

    for _, row in scan.iterrows():
        i, j = int(row["asset_i"]), int(row["asset_j"])
        p1, p2 = prices.iloc[:, i].to_numpy(), prices.iloc[:, j].to_numpy()

        pnl_naive, _ = run_pair_strategy(p1, p2, row["beta"], use_risk_mgmt=False)
        pnl_managed, n_stops = run_pair_strategy(p1, p2, row["beta"], use_risk_mgmt=True)

        cum_naive = np.cumsum(pnl_naive)
        cum_managed = np.cumsum(pnl_managed)

        print(f"asset_{i}-asset_{j:<7}{sharpe(pnl_naive):>15.3f}{max_drawdown(cum_naive):>14.4f}"
              f"{sharpe(pnl_managed):>15.3f}{max_drawdown(cum_managed):>14.4f}{n_stops:>8}")

        total_pnl_naive += pnl_naive
        total_pnl_managed += pnl_managed

    cum_total_naive = np.cumsum(total_pnl_naive)
    cum_total_managed = np.cumsum(total_pnl_managed)

    print(f"\n--- Portefeuille agrégé (toutes les paires sélectionnées) ---")
    print(f"Sans gestion du risque : Sharpe={sharpe(total_pnl_naive):.3f}  "
          f"MaxDD={max_drawdown(cum_total_naive):.4f}")
    print(f"Avec gestion du risque  : Sharpe={sharpe(total_pnl_managed):.3f}  "
          f"MaxDD={max_drawdown(cum_total_managed):.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cum_total_naive, color="firebrick", lw=1.3, label="Sans gestion du risque (sizing fixe, pas de stop)")
    ax.plot(cum_total_managed, color="navy", lw=1.3, label="Avec gestion du risque (sizing par vol + stop-loss)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Jour")
    ax.set_ylabel("P&L cumulé (budget de risque)")
    ax.set_title("Portefeuille pairs trading : impact de la gestion du risque")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("../P22_Pairs_trading_mean_reversion_(scan_univers_et_gestion du risque)/outputs/pairs_trading_pnl.png", dpi=140)
    plt.close(fig)
    print("\nGraphique exporté -> ../P22_Pairs_trading_mean_reversion_(scan_univers_et_gestion du risque)/outputs/pairs_trading_pnl.png")


if __name__ == "__main__":
    main()
