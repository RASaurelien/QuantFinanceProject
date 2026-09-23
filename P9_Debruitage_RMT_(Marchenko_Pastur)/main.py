"""
main.py
=======
Pipeline complet :

1. Simule un univers d'actifs sous un modèle à facteurs 
    (covariance VRAIE connue, voir simulate.py).
2. Calcule la matrice de corrélation empirique 
    (bruitée, puisque N=200 proche de T=500, ratio q=N/T=0.4, loin d'être négligeable).
3. Superpose le spectre de valeurs propres empirique, 
    à la loi de Marchenko-Pastur théorique : 
    montre visuellement combien de valeurs propres sont de PUR BRUIT.
4. Débruite la matrice de corrélation (filtre RMT), 
    reconstruit une covariance débruitée.
5. VALIDATION PRATIQUE : 
    compare le portefeuille de variance minimale construit sur: 
        (a) la covariance brute, 
        (b) la covariance débruitée, 
        (c) la vraie covariance 
            (oracle, inaccessible en pratique), 
            et mesure la variance RÉELLE de chacun. 
C'est la démonstration qui compte : pas "le spectre a l'air plus propre",
mais "ça réduit vraiment le risque de portefeuille mal estimé".
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate import simulate_factor_model
from rmt import marchenko_pastur_bounds, marchenko_pastur_density, denoise_correlation_matrix, correlation_to_covariance
from portfolio import min_variance_weights, realized_variance


def plot_eigenvalue_spectrum(eigvals: np.ndarray, q: float, outpath: str) -> None:
    lambda_minus, lambda_plus = marchenko_pastur_bounds(q)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panneau 1 : histogramme complet, avec la loi MP superposee ---
    ax = axes[0]
    ax.hist(eigvals, bins=60, density=True, color="steelblue", alpha=0.7, label="Spectre empirique")
    x_mp = np.linspace(max(lambda_minus, 1e-6), lambda_plus, 500)
    ax.plot(x_mp, marchenko_pastur_density(x_mp, q), color="black", lw=2, label="Loi de Marchenko-Pastur (bruit pur)")
    ax.axvline(lambda_plus, color="firebrick", ls="--", lw=1.5, label=r"$\lambda_+$ (borne de bruit)")
    ax.set_xlabel("Valeur propre")
    ax.set_ylabel("Densité")
    ax.set_title("Spectre complet : le \"bulk\" de bruit vs les valeurs propres signal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panneau 2 : zoom sur le bulk de bruit uniquement ---
    ax = axes[1]
    bulk = eigvals[eigvals < lambda_plus * 1.5]
    ax.hist(bulk, bins=50, density=True, color="steelblue", alpha=0.7, label="Spectre empirique (zoom)")
    ax.plot(x_mp, marchenko_pastur_density(x_mp, q), color="black", lw=2, label="Loi de Marchenko-Pastur")
    ax.axvline(lambda_plus, color="firebrick", ls="--", lw=1.5, label=r"$\lambda_+$")
    ax.set_xlabel("Valeur propre")
    ax.set_title("Zoom sur le bulk : ajustement à la théorie")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[main] Spectre de valeurs propres exporté -> {outpath}")


def plot_correlation_matrices(corr_raw: np.ndarray, corr_denoised: np.ndarray, outpath: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    vmax = max(np.abs(corr_raw - np.eye(len(corr_raw))).max(),
                np.abs(corr_denoised - np.eye(len(corr_denoised))).max())

    im0 = axes[0].imshow(corr_raw, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Corrélation brute (bruitée)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(corr_denoised, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Corrélation débruitée (RMT)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle("Matrice de corrélation : avant / après débruitage RMT")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[main] Matrices de corrélation exportées -> {outpath}")


def plot_portfolio_comparison(variances: dict, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    labels = list(variances.keys())
    values = [variances[k] * 1e4 for k in labels]   # en points de variance (x1e4 pour lisibilité)
    colors = ["firebrick", "darkorange", "seagreen"]

    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel(r"Variance réelle du portefeuille ($\times 10^{-4}$)")
    ax.set_title("Portefeuille de variance minimale : covariance brute vs débruitée vs oracle")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[main] Comparaison des portefeuilles exportée -> {outpath}")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" Débruitage de matrice de corrélation par RMT (Marchenko-Pastur)")
    print("=" * 60 + "\n")

    N, T = 200, 500
    returns, Sigma_true = simulate_factor_model(n_assets=N, n_periods=T)
    q = N / T
    print(f"Univers simulé : N={N} actifs, T={T} observations -> q = N/T = {q:.2f}")
    print("(q proche de 1 = régime où l'estimation de covariance est la plus difficile "
          "et où le débruitage RMT apporte le plus)\n")

    # --- Corrélation empirique (brute) ---
    vols_sample = returns.std(axis=0)
    corr_raw = np.corrcoef(returns.T)
    eigvals_raw = np.linalg.eigvalsh(corr_raw)

    lambda_minus, lambda_plus = marchenko_pastur_bounds(q)
    n_above_mp = int((eigvals_raw > lambda_plus).sum())
    print(f"Borne de bruit Marchenko-Pastur : lambda_+ = {lambda_plus:.3f}")
    print(f"Valeurs propres au-dessus de lambda_+ (signal détecté) : {n_above_mp} / {N}")
    print(f"-> cohérent avec les {5} facteurs vrais utilisés pour générer les données "
          f"(le facteur marché domine largement, donc 1 valeur propre très grande + "
          f"quelques autres significatives)\n")

    # --- Débruitage RMT ---
    corr_denoised, info = denoise_correlation_matrix(corr_raw, q)
    print(f"Valeurs propres signal conservées : {info['n_signal_eigenvalues']}")
    print(f"Valeurs propres bruit remplacées par leur moyenne ({info['noise_mean']:.4f}) : "
          f"{info['n_noise_eigenvalues']}\n")

    Sigma_raw = correlation_to_covariance(corr_raw, vols_sample)
    Sigma_denoised = correlation_to_covariance(corr_denoised, vols_sample)

    # --- Validation pratique : portefeuille de variance minimale ---
    w_raw = min_variance_weights(Sigma_raw)
    w_denoised = min_variance_weights(Sigma_denoised)
    w_oracle = min_variance_weights(Sigma_true)

    var_raw = realized_variance(w_raw, Sigma_true)
    var_denoised = realized_variance(w_denoised, Sigma_true)
    var_oracle = realized_variance(w_oracle, Sigma_true)

    print("--- Portefeuille de variance minimale : variance RÉELLE (évaluée avec Sigma vraie) ---")
    print(f"Covariance brute (échantillon)   : {var_raw*1e4:.4f}  (x1e-4)")
    print(f"Covariance débruitée (RMT)         : {var_denoised*1e4:.4f}  (x1e-4)")
    print(f"Covariance oracle (vraie, inconnue en pratique) : {var_oracle*1e4:.4f}  (x1e-4)")
    print(f"\nRéduction de la variance réelle grâce au débruitage : "
          f"{100*(var_raw - var_denoised)/var_raw:.1f} %")
    print(f"Écart résiduel à l'oracle après débruitage : "
          f"{100*(var_denoised - var_oracle)/var_oracle:.1f} % "
          f"(contre {100*(var_raw - var_oracle)/var_oracle:.1f} % avant débruitage)\n")

    # --- Graphiques ---
    plot_eigenvalue_spectrum(eigvals_raw, q, str(output_dir / "eigenvalue_spectrum.png"))
    plot_correlation_matrices(corr_raw, corr_denoised, str(output_dir / "correlation_matrices.png"))
    plot_portfolio_comparison(
        {"Brute": var_raw, "Débruitée (RMT)": var_denoised, "Oracle": var_oracle},
        str(output_dir / "portfolio_comparison.png")
    )


if __name__ == "__main__":
    main()
