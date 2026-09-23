"""
main.py
=======
Pipeline complet de calibration de surface de volatilité :

1. Charge une chaîne d'options (réelle via yfinance si --ticker est fourni et le réseau est disponible, 
    sinon un jeu synthétique réaliste généré localement).
2. Inverse Black-Scholes pour obtenir la volatilité implicite de marché à partir des prix.
3. Calibre un SVI "raw" indépendamment sur chaque tranche de maturité.
4. Vérifie l'absence d'arbitrage de calendrier entre tranches.
5. Exporte : graphiques de smile par maturité, surface 3D, table des paramètres calibrés (CSV).

Usage :
    python main.py                     # mode synthétique (par défaut)
    python main.py --ticker AAPL       # tente des données réelles
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # pas d'affichage interactif : on exporte directement en PNG
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (nécessaire pour projection='3d')

from data_source import fetch_real_chain, generate_synthetic_chain, compute_implied_vols
from svi import calibrate_svi_slice, svi_implied_vol, min_total_variance, check_calendar_arbitrage

"""Supprime le fichier existant puis sauvegarde la figure pour forcer l'écrasement."""
def save_figure(fig, outpath: str) -> None:
    path = Path(outpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build_chain(args: argparse.Namespace) -> pd.DataFrame:
    if args.ticker:
        chain = fetch_real_chain(args.ticker, r=args.r)
        if chain is not None:
            print(f"[main] Données réelles récupérées pour {args.ticker} ({len(chain)} cotations).")
            return chain
        print("[main] Bascule sur le jeu de données synthétique.")

    chain = generate_synthetic_chain(S0=args.S0, r=args.r, q=args.q)
    print(f"[main] Jeu de données synthétique généré ({len(chain)} cotations, "
          f"{chain['T'].nunique()} maturités).")
    return chain

"""Calibre un SVI par maturité et retourne {T: SVIParams}."""
def calibrate_all_maturities(chain_iv: pd.DataFrame) -> dict:
    params_by_T = {}
    for T, group in chain_iv.groupby("T"):
        forward = group["S"].iloc[0] * np.exp((group["r"].iloc[0] - group["q"].iloc[0]) * T)
        k = np.log(group["K"].to_numpy() / forward)
        market_iv = group["iv"].to_numpy()

        params = calibrate_svi_slice(k, T, market_iv)
        params_by_T[T] = params

        fitted_iv = svi_implied_vol(k, T, params)
        rmse = np.sqrt(np.nanmean((fitted_iv - market_iv) ** 2))
        arb_ok = min_total_variance(params) >= -1e-8

        print(f"  T={T:5.3f}  a={params.a:+.4f} b={params.b:.4f} "
              f"rho={params.rho:+.3f} m={params.m:+.3f} sigma={params.sigma:.4f}  "
              f"| RMSE(vol)={rmse*100:.3f}%  | butterfly OK: {arb_ok}")

    return params_by_T


def plot_smiles(chain_iv: pd.DataFrame, params_by_T: dict, outpath: str) -> None:
    maturities = sorted(params_by_T.keys())
    n = len(maturities)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, T in enumerate(maturities):
        ax = axes[idx // ncols][idx % ncols]
        group = chain_iv[chain_iv["T"] == T]
        forward = group["S"].iloc[0] * np.exp((group["r"].iloc[0] - group["q"].iloc[0]) * T)
        k_market = np.log(group["K"].to_numpy() / forward)

        k_grid = np.linspace(k_market.min(), k_market.max(), 200)
        fitted_iv = svi_implied_vol(k_grid, T, params_by_T[T])

        ax.scatter(k_market, group["iv"] * 100, s=18, color="crimson", label="Marché (implied vol)", zorder=3)
        ax.plot(k_grid, fitted_iv * 100, color="navy", lw=1.8, label="SVI calibré")
        ax.set_title(f"T = {T:.3f} an(s)")
        ax.set_xlabel("log-moneyness  k = log(K/F)")
        ax.set_ylabel("Vol. implicite (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Calibration SVI par tranche de maturité", fontsize=14)
    fig.tight_layout()
    save_figure(fig, outpath)
    print(f"[main] Graphique des smiles exporté -> {outpath}")


def plot_surface(params_by_T: dict, outpath: str) -> None:
    maturities = np.array(sorted(params_by_T.keys()))
    k_grid = np.linspace(-0.4, 0.4, 80)

    K_mesh, T_mesh = np.meshgrid(k_grid, maturities)
    IV_mesh = np.array([svi_implied_vol(k_grid, T, params_by_T[T]) for T in maturities])

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K_mesh, T_mesh, IV_mesh * 100, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("Maturité T (années)")
    ax.set_zlabel("Vol. implicite (%)")
    ax.set_title("Surface de volatilité calibrée (SVI)")
    fig.colorbar(surf, shrink=0.6, aspect=12, label="Vol. implicite (%)")
    fig.tight_layout()
    save_figure(fig, outpath)
    print(f"[main] Surface 3D exportée -> {outpath}")


def export_params(params_by_T: dict, outpath: str) -> None:
    rows = [{"T": T, "a": p.a, "b": p.b, "rho": p.rho, "m": p.m, "sigma": p.sigma,
             "min_total_variance": min_total_variance(p)}
            for T, p in sorted(params_by_T.items())]
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"[main] Paramètres calibrés exportés -> {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration de surface de volatilité (SVI)")
    parser.add_argument("--ticker", type=str, default=None,
                         help="Ticker Yahoo Finance (ex: AAPL). Si absent ou si le réseau "
                              "est indisponible, bascule automatiquement sur des données synthétiques.")
    parser.add_argument("--S0", type=float, default=100.0, help="Spot (mode synthétique)")
    parser.add_argument("--r", type=float, default=0.03, help="Taux sans risque")
    parser.add_argument("--q", type=float, default=0.01, help="Taux de dividende (mode synthétique)")
    parser.add_argument("--outdir", type=str, default="outputs", help="Dossier d'export des résultats")
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print(" Calibration de surface de volatilité implicite (SVI)")
    print("=" * 60)

    chain = build_chain(args)
    chain_iv = compute_implied_vols(chain)
    print(f"[main] Volatilités implicites recalculées pour {len(chain_iv)} cotations "
          f"({len(chain) - len(chain_iv)} cotations exclues : hors bornes d'arbitrage / illiquides).\n")

    print("[main] Calibration SVI par maturité :")
    params_by_T = calibrate_all_maturities(chain_iv)

    k_grid = np.linspace(-0.4, 0.4, 200)
    no_calendar_arb = check_calendar_arbitrage(params_by_T, k_grid)
    print(f"\n[main] Absence d'arbitrage de calendrier (variance croissante en T) : {no_calendar_arb}")

    plot_smiles(chain_iv, params_by_T, f"{args.outdir}/smile_by_maturity.png")
    plot_surface(params_by_T, f"{args.outdir}/vol_surface_3d.png")
    export_params(params_by_T, f"{args.outdir}/svi_params.csv")


if __name__ == "__main__":
    main()
