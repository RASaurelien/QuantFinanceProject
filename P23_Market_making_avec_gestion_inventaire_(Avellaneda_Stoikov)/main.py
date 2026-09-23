"""
main.py
=======
Compare, sur EXACTEMENT les mêmes chemins de prix et flux d'ordres 
(Monte Carlo apparié), 
la stratégie Avellaneda-Stoikov (cotation ajustée à l'inventaire) 
contre une cotation naïve à spread FIXE 
(même largeur de spread au départ, mais jamais ajustée à l'inventaire)
- pour isoler l'effet spécifique de la gestion d'inventaire, 
pas juste "un spread plus large gagne plus".
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from avellaneda_stoikov import as_quotes, naive_quotes, optimal_spread
from simulate import run_session

GAMMA, SIGMA, K, T = 0.1, 2.0, 1.5, 1.0


def main():
    import os
    os.makedirs("../P23_Market_making_avec_gestion_inventaire_(Avellaneda_Stoikov)/outputs", exist_ok=True)

    print("=" * 60)
    print(" Market Making, Avellaneda-Stoikov vs cotation naïve")
    print("=" * 60 + "\n")

    naive_half_spread = optimal_spread(GAMMA, SIGMA, T, K) / 2
    print(f"Spread naïf (fixe, calibré sur AS à t=0, q=0) : {naive_half_spread*2:.4f}\n")

    def as_fn(s, q, time_left):
        return as_quotes(s, q, GAMMA, SIGMA, time_left, K)

    def naive_fn(s, q, time_left):
        return naive_quotes(s, naive_half_spread)

    N_SESSIONS = 300
    pnl_as, pnl_naive = [], []
    inv_std_as, inv_std_naive = [], []
    example_as = example_naive = None

    for i in range(N_SESSIONS):
        mid, inv_as, pnl_path_as = run_session(as_fn, seed=i)
        _, inv_naive, pnl_path_naive = run_session(naive_fn, seed=i)   # MÊME seed -> même chemin de prix/ordres

        pnl_as.append(pnl_path_as[-1])
        pnl_naive.append(pnl_path_naive[-1])
        inv_std_as.append(inv_as.std())
        inv_std_naive.append(inv_naive.std())

        if i == 0:
            example_as = (mid, inv_as, pnl_path_as)
            example_naive = (mid, inv_naive, pnl_path_naive)

    pnl_as, pnl_naive = np.array(pnl_as), np.array(pnl_naive)
    inv_std_as, inv_std_naive = np.array(inv_std_as), np.array(inv_std_naive)

    print(f"--- Résultat sur {N_SESSIONS} sessions (Monte Carlo apparié, même flux d'ordres) ---\n")
    print(f"{'':<28}{'P&L moyen':>12}{'Écart-type P&L':>16}{'Sharpe':>9}{'Écart-type inventaire':>24}")
    print(f"{'Avellaneda-Stoikov':<28}{pnl_as.mean():>12.2f}{pnl_as.std():>16.2f}"
          f"{pnl_as.mean()/pnl_as.std():>9.3f}{inv_std_as.mean():>24.2f}")
    print(f"{'Naïf (spread fixe)':<28}{pnl_naive.mean():>12.2f}{pnl_naive.std():>16.2f}"
          f"{pnl_naive.mean()/pnl_naive.std():>9.3f}{inv_std_naive.mean():>24.2f}")

    reduction = (1 - inv_std_as.mean() / inv_std_naive.mean()) * 100
    print(f"\n-> Réduction de l'écart-type d'inventaire grâce à Avellaneda-Stoikov : {reduction:.1f}%")
    print("   (c'est l'effet recherché : le MM accepte un P&L légèrement différent en échange")
    print("   d'un risque d'inventaire structurellement mieux contrôlé, pas une stratégie qui")
    print("   'gagne plus', une stratégie qui gère mieux le risque pour un spread donné.)\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    mid, inv_as, pnl_as_path = example_as
    _, inv_naive, pnl_naive_path = example_naive

    axes[0, 0].plot(inv_as, color="navy", lw=1, label="Avellaneda-Stoikov")
    axes[0, 0].plot(inv_naive, color="firebrick", lw=1, alpha=0.7, label="Naïf")
    axes[0, 0].axhline(0, color="black", lw=0.6)
    axes[0, 0].set_title("Inventaire dans le temps (une session)")
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(pnl_as_path, color="navy", lw=1, label="Avellaneda-Stoikov")
    axes[0, 1].plot(pnl_naive_path, color="firebrick", lw=1, alpha=0.7, label="Naïf")
    axes[0, 1].set_title("P&L marked-to-market (une session)")
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].hist(pnl_as, bins=30, alpha=0.6, color="navy", label="Avellaneda-Stoikov", density=True)
    axes[1, 0].hist(pnl_naive, bins=30, alpha=0.6, color="firebrick", label="Naïf", density=True)
    axes[1, 0].set_title(f"Distribution du P&L terminal ({N_SESSIONS} sessions)")
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].hist(inv_std_as, bins=25, alpha=0.6, color="navy", label="Avellaneda-Stoikov", density=True)
    axes[1, 1].hist(inv_std_naive, bins=25, alpha=0.6, color="firebrick", label="Naïf", density=True)
    axes[1, 1].set_title("Distribution de l'écart-type d'inventaire par session")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

    fig.suptitle("Market Making : Avellaneda-Stoikov vs cotation naïve à spread fixe", fontsize=13)
    fig.tight_layout()
    fig.savefig("../P23_Market_making_avec_gestion_inventaire_(Avellaneda_Stoikov)/outputs/market_making.png", dpi=140)
    plt.close(fig)
    print("Graphique exporté -> ../P23_Market_making_avec_gestion_inventaire_(Avellaneda_Stoikov)/outputs/market_making.png")


if __name__ == "__main__":
    main()
