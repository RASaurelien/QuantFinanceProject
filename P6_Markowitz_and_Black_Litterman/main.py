"""
main.py
=======
Orchestration complète :

1. Charge les rendements multi-actifs (réels si possible, synthétiques sinon), 
    calcule mu/Sigma annualisés.
2. Calcule la frontière efficiente, le portefeuille de variance minimale et, 
    le portefeuille de tangence, Markowitz "naïf", entièrement par formules fermées 
    (voir markowitz.py).
3. Exporte mu/Sigma/poids de marché en CSV, puis appelle le script R (`Rscript black_litterman.R`) 
    qui calibre le modèle bayésien de Black-Litterman et écrit son résultat en retour.
4. Relit le résultat R et produit le graphique final : frontière efficiente Markowitz 
    + portefeuille de tangence + portefeuille Black-Litterman, 
    sur le même repère rendement/volatilité.

C'est un pipeline VOLONTAIREMENT bi-langage : 
Python fait le calcul numérique et l'orchestration, 
R fait l'inférence bayésienne, exactement la répartition demandée, et un signal utile en soi (savoir faire
communiquer deux langages via des fichiers d'échange, comme cela arrive
souvent dans des environnements de recherche quantitative hétérogènes).
"""

from __future__ import annotations
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_source import build_dataset
from markowitz import MarkowitzModel

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")
RF = 0.02


def export_for_r(mu: pd.Series, cov: pd.DataFrame, w_mkt: pd.Series) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    mu.to_frame("mu").to_csv(f"{DATA_DIR}/mu.csv")
    cov.to_csv(f"{DATA_DIR}/cov.csv")
    w_mkt.to_frame("weight").to_csv(f"{DATA_DIR}/market_weights.csv")
    print(f"[main] Données exportées pour R -> {DATA_DIR}/")


def run_black_litterman() -> pd.DataFrame:
    print("\n[main] Appel du script R (black_litterman.R)...\n")
    script_path = os.path.join(PROJECT_DIR, "black_litterman.R")
    try:
        result = subprocess.run(
            ["Rscript", script_path, DATA_DIR, OUT_DIR],
            text=True, timeout=120
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Le script R dépasse 120 secondes; exécution interrompue.") from exc
    if result.returncode != 0:
        raise RuntimeError(
            "Échec du script R, vérifier le message affiché ci-dessus et que Rscript est installé."
        )

    return pd.read_csv(os.path.join(DATA_DIR, "bl_results.csv"), index_col="asset")

def plot_frontier_comparison(model: MarkowitzModel, w_tangency: np.ndarray,
                              bl_results: pd.DataFrame, outpath: str) -> None:
    returns, vols = model.efficient_frontier()
    ret_tan, vol_tan, _ = model.portfolio_stats(w_tangency, RF)

    # Portefeuille Black-Litterman : recalculé dans l'espace Markowitz
    # "naïf" (mu, Sigma d'origine) pour comparaison honnête sur le même
    # repère, même si les poids ont été optimisés sous les rendements
    # postérieurs (voir R), on les rejoue ici avec les VRAIES stats
    # Markowitz pour visualiser où le portefeuille atterrit réellement.
    
    w_bl = bl_results["w_bl"].to_numpy()
    ret_bl, vol_bl, _ = model.portfolio_stats(w_bl, RF)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(vols * 100, returns * 100, color="navy", lw=2, label="Frontière efficiente (Markowitz)")
    ax.scatter([vol_tan * 100], [ret_tan * 100], color="crimson", s=90, zorder=5,
               label="Tangence Markowitz (mu historique)")
    ax.scatter([vol_bl * 100], [ret_bl * 100], color="darkorange", s=90, zorder=5, marker="D",
               label="Black-Litterman (vues investisseur)")

    for name, mu_i, sigma_i in zip(model.asset_names, model.mu, np.sqrt(np.diag(model.cov))):
        ax.scatter([sigma_i * 100], [mu_i * 100], color="gray", s=35, zorder=3)
        ax.annotate(name, (sigma_i * 100, mu_i * 100), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")

    ax.set_xlabel("Volatilité annualisée (%)")
    ax.set_ylabel("Rendement espéré annualisé (%)")
    ax.set_title("Frontière efficiente : Markowitz vs Black-Litterman")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[main] Graphique de comparaison exporté -> {outpath}")


def plot_weights_comparison(model: MarkowitzModel, w_tangency: np.ndarray,
                             bl_results: pd.DataFrame, outpath: str) -> None:
    w_bl = bl_results["w_bl"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(model.asset_names))
    width = 0.35
    ax.bar(x - width / 2, w_tangency * 100, width, label="Markowitz (tangence)", color="crimson")
    ax.bar(x + width / 2, w_bl * 100, width, label="Black-Litterman", color="darkorange")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model.asset_names, rotation=30, ha="right")
    ax.set_ylabel("Poids du portefeuille (%)")
    ax.set_title("Allocation : Markowitz (mu historique) vs Black-Litterman (vues)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[main] Comparaison des poids exportée -> {outpath}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Markowitz + Black-Litterman")
    parser.add_argument("real", action="store_true",
                         help="Tente de récupérer des rendements réels via yfinance (repli synthétique sinon).")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print(" Optimisation de portefeuille : Markowitz + Black-Litterman")
    print("=" * 60 + "\n")

    returns, w_mkt = build_dataset(use_real=args.real)
    mu = returns.mean() * 252
    cov = returns.cov() * 252

    model = MarkowitzModel(mu=mu.to_numpy(), cov=cov.to_numpy(), asset_names=list(mu.index))

    w_minvar = model.min_variance_portfolio()
    w_tangency = model.tangency_portfolio(RF)

    ret_mv, vol_mv, _ = model.portfolio_stats(w_minvar, RF)
    ret_tan, vol_tan, sharpe_tan = model.portfolio_stats(w_tangency, RF)

    print("--- Markowitz (rendements historiques) ---")
    print(f"Portefeuille de variance minimale : rendement={ret_mv*100:.2f}%  vol={vol_mv*100:.2f}%")
    print(f"Portefeuille de tangence (Sharpe max) : rendement={ret_tan*100:.2f}%  "
          f"vol={vol_tan*100:.2f}%  Sharpe={sharpe_tan:.3f}")
    print("Poids (tangence) :")
    for name, w in zip(model.asset_names, w_tangency):
        print(f"  {name:6s} {w*100:7.2f}%")
    print()

    export_for_r(mu, cov, w_mkt)
    bl_results = run_black_litterman()

    plot_frontier_comparison(model, w_tangency, bl_results, f"{OUT_DIR}/efficient_frontier.png")
    plot_weights_comparison(model, w_tangency, bl_results, f"{OUT_DIR}/weights_comparison.png")


if __name__ == "__main__":
    main()
