"""
plot_results.py
================
Lit les CSV produits par le moteur Rust (`signature_plot.csv`,
`bias_rmse.csv`) et génère les deux graphiques diagnostiques standard 
de la littérature sur le bruit de microstructure :

1. Le "signature plot" : RV naïve vs TSRV en fonction de la fréquence d'échantillonnage. 
    Le graphique de référence pour *montrer*, pas seulement expliquer, pourquoi échantillonner trop finement est
     dangereux sans correction de bruit.
2. Biais et RMSE des deux estimateurs en fonction de l'intensité du bruit, 
    avec la loi d'échelle théorique du biais naïf (~2n*sigma^2) superposée pour validation visuelle.

Usage :
python plot_results.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def reset_output_dir() -> None:
    for existing in OUTPUT_DIR.glob("*.png"):
        existing.unlink()


def resolve_csv(path_csv: str | Path) -> Path:
    candidates = [
        Path(path_csv),
        ROOT / path_csv,
        ROOT / "python" / path_csv,
        Path.cwd() / path_csv,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"CSV introuvable : {path_csv}. Le binaire Rust doit d'abord créer "
        f"signature_plot.csv et bias_rmse.csv dans le dossier du projet. "
        f"Recherche effectuée : {searched}"
    )


def resolve_output_path(outpath: str | Path) -> Path:
    if isinstance(outpath, Path):
        target = outpath
    else:
        target = Path(outpath)

    if target.is_absolute():
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    target_path = OUTPUT_DIR / target.name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path


def plot_signature(path_csv: str | Path, outpath: str | Path) -> None:
    csv_path = resolve_csv(path_csv)
    df = pd.read_csv(csv_path)
    output_path = resolve_output_path(outpath)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(df["n_ticks"], df["avg_naive_rv"] * 1e4, "o-", color="crimson",
             label="RV naïve (tout tick)", lw=1.8, markersize=5)
    ax.plot(df["n_ticks"], df["avg_tsrv"] * 1e4, "o-", color="navy",
             label="TSRV (Zhang-Mykland-Aït-Sahalia)", lw=1.8, markersize=5)
    ax.axhline(df["true_iv"].iloc[0] * 1e4, color="black", ls="--", lw=1.2,
                label="Variance intégrée vraie")

    ax.set_xscale("log")
    ax.set_xlabel("Nombre de ticks (fréquence d'échantillonnage croissante →)")
    ax.set_ylabel(r"Variance réalisée moyenne ($\times 10^{-4}$)")
    ax.set_title("Signature plot : la RV naïve diverge, le TSRV reste stable")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"[plot] Signature plot exporté -> {output_path}")


def plot_bias_rmse(path_csv: str | Path, outpath: str | Path, n_ticks: int = 23_400) -> None:
    csv_path = resolve_csv(path_csv)
    df = pd.read_csv(csv_path)
    output_path = resolve_output_path(outpath)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Biais ---
    ax = axes[0]
    ax.plot(df["noise_std"], df["bias_naive"], "o-", color="crimson", label="RV naïve")
    ax.plot(df["noise_std"], df["bias_tsrv"], "o-", color="navy", label="TSRV")
    # Loi d'échelle théorique du biais naïf : E[RV_naive] - IV ~= 2n*sigma_noise^2
    theory = 2 * n_ticks * df["noise_std"] ** 2
    ax.plot(df["noise_std"], theory, "--", color="gray", lw=1.2,
             label=r"Théorie : $2n\,\sigma_{bruit}^2$")
    ax.set_xlabel(r"Écart-type du bruit de microstructure $\sigma_{bruit}$")
    ax.set_ylabel("Biais de l'estimateur")
    ax.set_title("Biais vs intensité du bruit")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- RMSE (échelle log, car la RV naïve explose) ---
    ax = axes[1]
    ax.plot(df["noise_std"], df["rmse_naive"], "o-", color="crimson", label="RV naïve")
    ax.plot(df["noise_std"], df["rmse_tsrv"], "o-", color="navy", label="TSRV")
    ax.set_yscale("log")
    ax.set_xlabel(r"Écart-type du bruit de microstructure $\sigma_{bruit}$")
    ax.set_ylabel("RMSE (échelle log)")
    ax.set_title("RMSE vs intensité du bruit")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"Robustesse au bruit de microstructure (n_ticks={n_ticks})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"[plot] Graphique biais/RMSE exporté -> {output_path}")


if __name__ == "__main__":
    reset_output_dir()
    plot_signature("signature_plot.csv", "signature_plot.png")
    plot_bias_rmse("bias_rmse.csv", "bias_rmse.png")
