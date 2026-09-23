"""
plot_results.py
================
Lit equity_curves.csv (produit par le moteur Rust) 
et trace les courbes d'équité des deux stratégies, 
vs le benchmark Buy & Hold, 
ainsi que le prix sous-jacent pour visualiser les régimes de marché traversés.

Usage : après ./target/release/tier2_backtester, dans le même dossier
    python plot_results.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_equity_curves(df: pd.DataFrame, outpath: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                               gridspec_kw={"height_ratios": [1, 2]})

    axes[0].plot(df["day"], df["price"], color="gray", lw=0.9)
    axes[0].set_ylabel("Prix simulé")
    axes[0].set_title("Série de prix (changement de régime : tendance / range)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["day"], df["momentum"], color="firebrick", lw=1.3, label="Momentum (MA Crossover)")
    axes[1].plot(df["day"], df["mean_reversion"], color="navy", lw=1.3, label="Mean-Reversion (Z-score)")
    axes[1].plot(df["day"], df["buy_hold"], color="gray", lw=1.1, ls="-", label="Buy & Hold")
    axes[1].axhline(df["momentum"].iloc[0], color="black", lw=0.6, ls=":")
    axes[1].set_xlabel("Jour")
    axes[1].set_ylabel("Valeur du portefeuille")
    axes[1].set_title("Courbes d'équité : Momentum vs Mean-Reversion vs Buy & Hold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[plot] Courbes d'équité exportées -> {outpath}")


def plot_drawdowns(df: pd.DataFrame, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))

    for col, color, label in [("momentum", "firebrick", "Momentum"),
                                ("mean_reversion", "navy", "Mean-Reversion"),
                                ("buy_hold", "gray", "Buy & Hold")]:
        equity = df[col].to_numpy()
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        ax.plot(df["day"], drawdown, color=color, lw=1.1, label=label)

    ax.set_xlabel("Jour")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown glissant par stratégie")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[plot] Drawdowns exportés -> {outpath}")


if __name__ == "__main__":
    from pathlib import Path

    project_dir = Path(__file__).resolve().parent
    csv_path = project_dir / "equity_curves.csv"
    equity_path = project_dir / "equity_curves.png"
    drawdowns_path = project_dir / "drawdowns.png"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run the Rust backtester first to create it."
        )

    df = pd.read_csv(csv_path)
    plot_equity_curves(df, str(equity_path))
    plot_drawdowns(df, str(drawdowns_path))
