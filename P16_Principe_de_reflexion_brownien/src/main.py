"""
main.py
=======
Verifie numeriquement le PRINCIPE DE REFLEXION du mouvement brownien
et ses deux consequences directes :

  1. La loi du maximum courant M_t = max_{s<=t} W_s :
       P(M_t >= a) = 2 * P(W_t >= a)   pour a >= 0
     et sa densite (loi demi-normale) :
       f_{M_t}(m) = sqrt(2/(pi t)) * exp(-m^2/(2t)),   m >= 0

  2. Le temps de premier passage tau_a = inf{t : W_t = a} :
       P(tau_a <= t) = 2 * P(W_t >= a) = 2*(1 - Phi(a/sqrt(t)))
     et sa densite (loi de Levy) :
       f_{tau_a}(t) = (a / sqrt(2*pi*t^3)) * exp(-a^2/(2t))

Argument de reflexion (Desire Andre, 1887 / Levy) : toute trajectoire
qui touche le niveau a avant t peut etre "reflechie" apres l'instant
de premier passage (W devient -W a partir de tau_a) pour produire une
AUTRE trajectoire brownienne tout aussi probable -- d'ou le facteur 2.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate_paths(n_paths: int, n_steps: int, T: float, seed: int = 42):
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    increments = rng.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    times = np.linspace(0, T, n_steps + 1)
    return times, W


def first_passage_times(times: np.ndarray, W: np.ndarray, a: float) -> np.ndarray:
    """Pour chaque trajectoire, le premier instant ou W >= a (NaN si jamais atteint sur l'horizon simule)."""
    hit = W >= a
    first_idx = np.argmax(hit, axis=1)               # index du premier True (0 si aucun True)
    never_hit = ~hit.any(axis=1)
    tau = times[first_idx]
    tau[never_hit] = np.nan
    return tau


def main():
    import os
    os.makedirs("../outputs", exist_ok=True)

    print("=" * 60)
    print(" Principe de réflexion du mouvement brownien -- validation numérique")
    print("=" * 60 + "\n")

    T, n_steps, n_paths = 1.0, 3000, 20_000
    a = 0.5   # niveau de reflexion / barriere

    times, W = simulate_paths(n_paths, n_steps, T)
    M = np.max(W, axis=1)   # maximum courant sur [0,T] pour chaque trajectoire

    # --- Vérification 1 : P(M_T >= a) = 2 * P(W_T >= a) ---
    p_max_empirical = np.mean(M >= a)
    p_wT_empirical = np.mean(W[:, -1] >= a)
    p_max_theory = 2 * (1 - stats.norm.cdf(a / np.sqrt(T)))

    print(f"[Vérification 1] Niveau a={a}, horizon T={T}")
    print(f"  P(M_T >= a)  empirique             : {p_max_empirical:.5f}")
    print(f"  2 * P(W_T >= a)  empirique          : {2*p_wT_empirical:.5f}")
    print(f"  P(M_T >= a)  théorique (réflexion)  : {p_max_theory:.5f}")
    print(f"  Écart empirique vs théorie          : {abs(p_max_empirical-p_max_theory):.5f}\n")

    # --- Vérification 2 : densité du maximum courant (loi demi-normale) ---
    m_grid = np.linspace(0.001, M.max(), 300)
    density_theory = np.sqrt(2 / (np.pi * T)) * np.exp(-m_grid**2 / (2 * T))

    # --- Vérification 3 : temps de premier passage (loi de Lévy) ---
    tau = first_passage_times(times, W, a)
    frac_hit = np.mean(~np.isnan(tau))
    tau_hit = tau[~np.isnan(tau)]
    print(f"[Vérification 2] Temps de premier passage au niveau a={a}")
    print(f"  Fraction de trajectoires ayant atteint a avant T={T} : {frac_hit:.4f}")
    print(f"  (théorie : P(tau_a <= T) = {p_max_theory:.4f}, cohérent avec Vérification 1 "
          f"puisque {{M_T >= a}} = {{tau_a <= T}})")
    print(f"  Temps de passage moyen (trajectoires ayant touché a) : {tau_hit.mean():.4f}\n")

    t_grid = np.linspace(0.005, T, 300)
    levy_density = (a / np.sqrt(2 * np.pi * t_grid**3)) * np.exp(-a**2 / (2 * t_grid))
    # NB : l'histogramme empirique ci-dessous (panneau bas-droit) ne porte que sur
    # les trajectoires ayant effectivement touche a avant T (tau_hit), et
    # np.histogram(..., density=True) le normalise pour integrer a 1 sur CE
    # sous-ensemble. La densite de Levy brute ci-dessus est la densite
    # INCONDITIONNELLE (integrant a 1 sur [0, +inf)) : la comparer telle
    # quelle a l'histogramme sous-estime systematiquement la courbe d'un
    # facteur 1/P(tau_a<=T). On trace donc la densite CONDITIONNELLE
    # f_{tau_a}(t) / P(tau_a<=T), seule quantite comparable a l'histogramme.
    levy_density_conditional = levy_density / frac_hit

    # --- Graphiques ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Quelques trajectoires + niveau a
    n_show = 15
    for i in range(n_show):
        axes[0, 0].plot(times, W[i], lw=0.8, alpha=0.7)
    axes[0, 0].axhline(a, color="firebrick", ls="--", lw=1.5, label=f"Niveau a={a}")
    axes[0, 0].set_title(f"{n_show} trajectoires browniennes")
    axes[0, 0].set_xlabel("t"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    # (b) Illustration de la réflexion sur UNE trajectoire qui touche a
    idx_hit = np.where(~np.isnan(tau))[0]
    if len(idx_hit) > 0:
        j = idx_hit[0]
        t_hit = tau[j]
        idx_t_hit = np.searchsorted(times, t_hit)
        path = W[j].copy()
        reflected = path.copy()
        reflected[idx_t_hit:] = 2 * a - path[idx_t_hit:]   # réflexion : W -> 2a - W après tau_a
        axes[0, 1].plot(times, path, color="navy", lw=1.3, label="Trajectoire originale")
        axes[0, 1].plot(times, reflected, color="firebrick", lw=1.3, ls="--", label="Trajectoire réfléchie après τ_a")
        axes[0, 1].axhline(a, color="gray", ls=":", lw=1.2)
        axes[0, 1].axvline(t_hit, color="gray", ls=":", lw=1.2)
        axes[0, 1].set_title("Illustration du principe de réflexion")
        axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

    # (c) Distribution du maximum courant vs loi demi-normale théorique
    axes[1, 0].hist(M, bins=80, density=True, color="steelblue", alpha=0.7, label="Empirique")
    axes[1, 0].plot(m_grid, density_theory, color="firebrick", lw=2, label="Loi demi-normale (théorie)")
    axes[1, 0].set_title("Distribution de M_T = max sur [0,T] de W_s")
    axes[1, 0].set_xlabel("m"); axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

    # (d) Distribution du temps de premier passage vs loi de Lévy théorique
    axes[1, 1].hist(tau_hit, bins=80, density=True, color="seagreen", alpha=0.7, label="Empirique (conditionnel à τ_a≤T)")
    axes[1, 1].plot(t_grid, levy_density_conditional, color="firebrick", lw=2,
                     label="Loi de Lévy (théorie, conditionnelle à τ_a≤T)")
    axes[1, 1].set_title(f"Distribution du temps de premier passage τ_a (a={a})")
    axes[1, 1].set_xlabel("t"); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

    fig.suptitle("Principe de réflexion du mouvement brownien : validation numérique", fontsize=13)
    fig.tight_layout()
    fig.savefig("../P16_Principe_de_reflexion_brownien/outputs/reflection_principle.png", dpi=140)
    plt.close(fig)
    print("Graphique exporté -> ../P16_Principe_de_reflexion_brownien/outputs/reflection_principle.png")


if __name__ == "__main__":
    main()
