"""
main.py
=======
Simule le marche multi-agents de Cont-Bouchaud et verifie qu'il reproduit, 
PAR CONSTRUCTION STRUCTURELLE (pas par hypothese comportementale),
les deux faits stylises les plus robustes de la finance empirique :
    1. Queues epaisses des rendements (kurtosis >> 3, loi normale rejetee)
    2. Clustering de volatilite (ACF des rendements au carre significative)

Et, cote physique, verifie le mecanisme sous-jacent : 
la distribution de la taille des clusters, 
suit une loi de puissance pres du seuil de percolation.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_model import simulate_market, find_clusters

"""Verifie que pile au seuil de percolation (<k>=1), la distribution des tailles de clusters suit une loi de puissance."""
def analyze_cluster_distribution(n_agents: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    p_link = 1.0 / (n_agents - 1)
    sizes = find_clusters(n_agents, p_link, rng)
    unique_sizes, counts = np.unique(sizes, return_counts=True)
    n_clusters = counts / unique_sizes   # nombre de clusters de chaque taille (pas nombre d'agents)
    return unique_sizes, n_clusters


def main():
    import os
    os.makedirs("../outputs", exist_ok=True)

    print("=" * 60)
    print(" Simulateur de marché multi-agents (Cont-Bouchaud, percolation)")
    print("=" * 60 + "\n")

    N_AGENTS, N_STEPS = 2000, 2000
    print(f"Simulation : {N_AGENTS} agents, {N_STEPS} pas de temps...")
    returns, avg_degree = simulate_market(n_agents=N_AGENTS, n_steps=N_STEPS)
    print("Terminé.\n")

    # --- Fait stylisé 1 : queues épaisses ---
    kurt = stats.kurtosis(returns, fisher=True)   # excès de kurtosis (0 = gaussien)
    _, p_normal = stats.jarque_bera(returns)
    print(f"[Fait stylisé 1] Excès de kurtosis des rendements : {kurt:.2f} "
          f"(0 = gaussien ; positif = queues épaisses)")
    print(f"  Test de Jarque-Bera (normalité) : p-value = {p_normal:.2e} "
          f"({'rejet de la normalité' if p_normal < 0.01 else 'normalité non rejetée'})\n")

    # --- Fait stylisé 2 : clustering de volatilité ---
    abs_returns = np.abs(returns)
    acf_returns = [np.corrcoef(returns[:-k], returns[k:])[0, 1] for k in range(1, 21)]
    acf_abs = [np.corrcoef(abs_returns[:-k], abs_returns[k:])[0, 1] for k in range(1, 21)]
    print(f"[Fait stylisé 2] ACF moyenne des rendements (retards 1-20)     : {np.mean(acf_returns):+.4f}")
    print(f"                  ACF moyenne des rendements ABSOLUS (retards 1-20) : {np.mean(acf_abs):+.4f}")
    print("                  -> le clustering de volatilité émerge des PHASES de proximité")
    print("                     au seuil de percolation, pas d'une hypothèse GARCH imposée.\n")

    # --- Mécanisme physique : loi de puissance des tailles de clusters ---
    sizes, n_clusters = analyze_cluster_distribution(n_agents=5000)
    mask = (sizes > 1) & (n_clusters > 0)
    log_s, log_n = np.log(sizes[mask]), np.log(n_clusters[mask])
    slope, intercept, r_value, _, _ = stats.linregress(log_s, log_n)
    print(f"[Mécanisme] Distribution de la taille des clusters (au seuil, <k>=1) :")
    print(f"  Exposant de la loi de puissance : {slope:.2f}  (théorie champ moyen : -2.5)")
    print(f"  R² de l'ajustement log-log : {r_value**2:.3f}\n")

    # --- Graphiques ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(returns, color="steelblue", lw=0.6)
    axes[0, 0].set_title("Rendements simulés (Cont-Bouchaud)")
    axes[0, 0].set_xlabel("t"); axes[0, 0].grid(alpha=0.3)

    x = np.linspace(returns.min(), returns.max(), 200)
    axes[0, 1].hist(returns, bins=60, density=True, color="steelblue", alpha=0.7, label="Simulé")
    axes[0, 1].plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), color="firebrick", lw=2,
                      label="Normale (même moyenne/vol)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title(f"Distribution des rendements (queues épaisses, kurtosis={kurt:.1f})")
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].bar(range(1, 21), acf_returns, color="gray", alpha=0.7, label="Rendements")
    axes[1, 0].bar(range(1, 21), acf_abs, color="darkorange", alpha=0.6, label="Rendements absolus")
    axes[1, 0].axhline(0, color="black", lw=0.6)
    axes[1, 0].set_title("Clustering de volatilité (ACF)")
    axes[1, 0].set_xlabel("Retard"); axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].loglog(sizes[mask], n_clusters[mask], "o", color="steelblue", ms=4, label="Simulé")
    axes[1, 1].loglog(sizes[mask], np.exp(intercept) * sizes[mask] ** slope, color="firebrick",
                        lw=2, label=f"Loi de puissance (pente={slope:.2f})")
    axes[1, 1].set_title("Distribution de la taille des clusters (au seuil critique)")
    axes[1, 1].set_xlabel("Taille du cluster"); axes[1, 1].set_ylabel("Nombre de clusters")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3, which="both")

    fig.suptitle("Marché multi-agents : faits stylisés émergents (Cont-Bouchaud)", fontsize=13)
    fig.tight_layout()
    fig.savefig("../P13_Marche_multi_agents_(Cont_Bouchaud_percolation)/outputs/agent_market_diagnostics.png", dpi=140)
    plt.close(fig)
    print("Graphique exporté -> ../P13_Marche_multi_agents_(Cont_Bouchaud_percolation)/outputs/agent_market_diagnostics.png")


if __name__ == "__main__":
    main()
