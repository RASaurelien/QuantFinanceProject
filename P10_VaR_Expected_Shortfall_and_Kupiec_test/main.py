"""
main.py
=======
Pipeline complet :

1. Simule une longue série de rendements sous GARCH(1,1) 
    à innovations de Student 
    (queues épaisses + clustering de volatilité réalistes).
2. Backtest glissant des trois méthodes de VaR à 99% 
    (gaussienne, historique, Monte Carlo Student-t), 
    fenêtre de 500 jours.
3. Test de Kupiec (POF) sur chaque méthode : 
    le taux de dépassement observé est-il statistiquement cohérent, 
    avec le taux attendu (1%) ?
4. Graphiques : rendements + dépassements de VaR dans le temps,
    comparaison des taux de dépassement observés vs attendus.

Résultat attendu (et raison d'être du projet) : 
la VaR gaussienne doit sous-estimer le risque de queue sur des données à queues épaisses,
trop de dépassements, rejet du test de Kupiec, 
alors que les méthodes historique et Monte Carlo (Student-t), 
doivent mieux passer le test.
"""

from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate import simulate_garch_t
from backtest import rolling_backtest, kupiec_pof_test


ALPHA = 0.99
WINDOW = 500
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def plot_var_breaches(returns: np.ndarray, results: dict, outpath: str) -> None:
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3.2 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        ax.plot(returns, color="steelblue", lw=0.6, alpha=0.8, label="Rendement")
        ax.plot(-res["var_series"], color="black", lw=1.1, label=f"-VaR 99% ({name})")
        breach_idx = np.where(res["breaches"])[0]
        ax.scatter(breach_idx, returns[breach_idx], color="firebrick", s=22, zorder=5,
                    label=f"Dépassements ({res['breaches'].sum()})")
        ax.set_title(name)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("t (jours)")
    fig.suptitle("Backtest glissant : rendements, VaR 99% et dépassements", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[main] Graphique des dépassements exporté -> {outpath}")


def plot_kupiec_summary(kupiec_results: dict, outpath: str) -> None:
    names = list(kupiec_results.keys())
    observed = [kupiec_results[n]["observed_rate"] * 100 for n in names]
    expected = kupiec_results[names[0]]["expected_rate"] * 100

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["firebrick" if kupiec_results[n]["reject_5pct"] else "seagreen" for n in names]
    bars = ax.bar(names, observed, color=colors)
    ax.axhline(expected, color="black", ls="--", lw=1.3, label=f"Taux attendu ({expected:.1f}%)")

    for bar, name in zip(bars, names):
        pval = kupiec_results[name]["p_value"]
        verdict = "REJETÉ" if kupiec_results[name]["reject_5pct"] else "non rejeté"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"p={pval:.3f}\n({verdict})", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Taux de dépassement observé (%)")
    ax.set_title("Test de Kupiec (POF) : taux de dépassement observé vs attendu, par méthode")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[main] Résumé du test de Kupiec exporté -> {outpath}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"[main] Dossier de sortie : {OUTPUT_DIR}")

    print("=" * 60)
    print(" VaR / Expected Shortfall, backtest et test de Kupiec")
    print("=" * 60 + "\n")

    returns = simulate_garch_t(n_days=2200, nu=5.0)
    print(f"[main] {len(returns)} jours simulés (GARCH(1,1), innovations Student nu=5 : queues épaisses).")
    print(f"[main] Fenêtre de backtest glissante : {WINDOW} jours - "
          f"{len(returns) - WINDOW} jours testés hors échantillon.\n")

    t0 = time.time()
    results = rolling_backtest(returns, window=WINDOW, alpha=ALPHA)
    print(f"[main] Backtest glissant terminé en {time.time() - t0:.1f} s.\n")

    n_test = len(returns) - WINDOW
    kupiec_results = {}
    print(f"--- Test de Kupiec (POF), alpha={ALPHA}, taux attendu={1-ALPHA:.1%} ---\n")
    for name, res in results.items():
        n_breaches = int(res["breaches"][WINDOW:].sum())
        kupiec = kupiec_pof_test(n_test, n_breaches, alpha=ALPHA)
        kupiec_results[name] = kupiec

        verdict = "REJETÉ (mal calibré)" if kupiec["reject_5pct"] else "non rejeté (bien calibré)"
        print(f"{name} :")
        print(f"  Dépassements observés : {kupiec['n_breaches']} / {kupiec['n_obs']} "
              f"({kupiec['observed_rate']*100:.2f}%)   [attendu : {kupiec['expected_breaches']:.1f} "
              f"soit {kupiec['expected_rate']*100:.1f}%]")
        print(f"  Statistique LR = {kupiec['lr_stat']:.2f}   p-value = {kupiec['p_value']:.4f}   -> {verdict}\n")

    # --- Graphiques ---
    # Pour la lisibilité du graphique de dépassements, on limite l'affichage
    # à la portion hors-warmup (les WINDOW premiers points n'ont pas de VaR).
    plot_results = {name: {"var_series": res["var_series"][WINDOW:], "breaches": res["breaches"][WINDOW:]}
                     for name, res in results.items()}
    plot_var_breaches(returns[WINDOW:], plot_results, OUTPUT_DIR / "var_breaches.png")
    plot_kupiec_summary(kupiec_results, OUTPUT_DIR / "kupiec_summary.png")


if __name__ == "__main__":
    main()
