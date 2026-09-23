"""
Lead-Lag Matrix pour données tick asynchrones (HFT)

Implémente:
  1. Simulation de données tick-by-tick asynchrones avec une structure 
  lead-lag connue (pour validation).
  2. Estimateur de Hayashi-Yoshida (covariance/corrélation sans resynchronisation).
  3. Construction de la matrice lead-lag (lag optimal + score de leadership)
  sur une grille de lags candidats.
  4. Visualisation (heatmap).

Référence: Hayashi, T. & Yoshida, N. (2005), "On covariance estimation of
non-synchronously observed diffusion processes"; Huth, N. & Abergel, F. (2014),
"High frequency lead/lag relationships Empirical facts".
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Simulation de données tick asynchrones avec structure lead-lag connue
# ---------------------------------------------------------------------------

"""
Simule un marché où l'actif 0 est le 'leader' : ses rendements se
propagent aux autres actifs avec un délai `lag_ms` et un bruit ajouté.

Retourne une liste de DataFrames (un par actif), chacun avec colonnes
['time', 'price'], timestamps en secondes, non synchronisés entre actifs
(c'est le point clé : les temps de trade sont aléatoires et différents
pour chaque actif, comme dans un vrai carnet d'ordres).
"""

def simulate_asynchronous_ticks(
    n_assets: int = 4,
    duration_seconds: float = 3600.0,
    mean_intertrade_ms: float = 200.0,
    lag_ms: float = 50.0,
    noise_std: float = 0.3,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    # Rendements "vrais" du leader sur une grille fine (1ms), qu'on va ensuite
    # répercuter avec délai sur les autres actifs.
    dt_fine = 0.001  # 1 ms
    n_steps = int(duration_seconds / dt_fine)
    leader_returns = rng.normal(0, 0.0005, n_steps)
    leader_price_path = 100 * np.exp(np.cumsum(leader_returns))
    time_grid = np.arange(n_steps) * dt_fine

    lag_steps = int((lag_ms / 1000.0) / dt_fine)

    asset_paths = [leader_price_path]
    for k in range(1, n_assets):
        shifted = np.roll(leader_price_path, lag_steps * k)
        shifted[: lag_steps * k] = leader_price_path[0]
        # bruit idiosyncratique additionnel (chaque actif ne suit pas parfaitement)
        idio_noise = np.cumsum(rng.normal(0, noise_std * 0.0005, n_steps))
        asset_paths.append(shifted * np.exp(idio_noise))

    # Échantillonnage asynchrone : pour chaque actif, on tire des temps de
    # trade selon un processus de Poisson (intertrade ~ exponentielle).
    ticks = []
    for path in asset_paths:
        t = 0.0
        times = []
        while t < duration_seconds:
            t += rng.exponential(mean_intertrade_ms / 1000.0)
            times.append(t)
        times = np.array(times)
        idx = np.clip((times / dt_fine).astype(int), 0, n_steps - 1)
        prices = path[idx]
        ticks.append(pd.DataFrame({"time": times, "price": prices}))

    return ticks


# ---------------------------------------------------------------------------
# 2. Estimateur de Hayashi-Yoshida (covariance réalisée non-synchrone)
# ---------------------------------------------------------------------------

"""Convertit une série de prix tick en intervalles (t_start, t_end, log-return)."""
def _tick_returns(df: pd.DataFrame):
    t = df["time"].to_numpy()
    p = np.log(df["price"].to_numpy())
    t_start = t[:-1]
    t_end = t[1:]
    ret = p[1:] - p[:-1]
    return t_start, t_end, ret

"""
Covariance de Hayashi-Yoshida entre deux séries tick asynchrones.
`lag` décale la série j de `lag` secondes (lag > 0 => on teste si i mène j).

HY-cov = somme_{k,l} r_i^k * r_j^l  sur tous les couples d'intervalles
qui se chevauchent (overlap), ce qui évite le biais d'un resampling
artificiel (effet Epps).
"""

def hayashi_yoshida_cov(df_i: pd.DataFrame, df_j: pd.DataFrame, lag: float = 0.0) -> float:
    ti0, ti1, ri = _tick_returns(df_i)
    tj0, tj1, rj = _tick_returns(df_j)

    # on décale les timestamps de j : lag > 0 teste "i mène j" en ramenant
    # le futur de j (t+lag) au présent de i (t), pour que leurs intervalles
    # se recouvrent quand r_j(t+lag) est effectivement corrélé à r_i(t).
    tj0 = tj0 - lag
    tj1 = tj1 - lag

    # Algorithme à deux pointeurs (les temps sont triés) : O(n_i + n_j)
    cov = 0.0
    j_start = 0
    n_j = len(rj)

    for k in range(len(ri)):
        a0, a1 = ti0[k], ti1[k]
        # avance le pointeur j tant que l'intervalle j est totalement avant i
        while j_start < n_j and tj1[j_start] < a0:
            j_start += 1
        l = j_start
        while l < n_j and tj0[l] <= a1:
            b0, b1 = tj0[l], tj1[l]
            # overlap si max(a0,b0) <= min(a1,b1)
            if max(a0, b0) <= min(a1, b1):
                cov += ri[k] * rj[l]
            l += 1

    return cov

"""Corrélation HY = cov_HY(i,j,lag) / sqrt(var_HY(i) * var_HY(j))."""
def hayashi_yoshida_corr(df_i: pd.DataFrame, df_j: pd.DataFrame, lag: float = 0.0) -> float:
    cov = hayashi_yoshida_cov(df_i, df_j, lag)
    var_i = hayashi_yoshida_cov(df_i, df_i, 0.0)
    var_j = hayashi_yoshida_cov(df_j, df_j, 0.0)
    denom = np.sqrt(var_i * var_j)
    return cov / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# 3. Construction de la matrice lead-lag
# ---------------------------------------------------------------------------

@dataclass
class LeadLagResult:
    optimal_lag: np.ndarray      # matrice N x N, lag optimal en secondes (i mène j si >0)
    leadership_score: np.ndarray  # matrice N x N, asymétrie corr(lag>0) - corr(lag<0)
    corr_curves: dict            # (i,j) -> array de corrélations pour chaque lag testé

"""
Pour chaque paire (i, j), calcule la corrélation HY sur une grille de lags
et en déduit:
- le lag qui maximise |corr| (lag optimal, signé)
- un score de leadership = intégrale(corr, lag>0) - intégrale(corr, lag<0) 
(positif => i mène j)
"""

def build_lead_lag_matrix(
    ticks: list,
    lags_ms=None,
) -> LeadLagResult:
    if lags_ms is None:
        lags_ms = np.arange(-200, 201, 10)  # -200ms à +200ms par pas de 10ms
    lags_s = lags_ms / 1000.0

    n = len(ticks)
    optimal_lag = np.zeros((n, n))
    leadership_score = np.zeros((n, n))
    corr_curves = {}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            corrs = np.array([hayashi_yoshida_corr(ticks[i], ticks[j], lag) for lag in lags_s])
            corr_curves[(i, j)] = corrs

            best_idx = np.argmax(np.abs(corrs))
            optimal_lag[i, j] = lags_s[best_idx]

            pos_mask = lags_s > 0
            neg_mask = lags_s < 0
            leadership_score[i, j] = corrs[pos_mask].sum() - corrs[neg_mask].sum()

    return LeadLagResult(optimal_lag, leadership_score, corr_curves)


# ---------------------------------------------------------------------------
# 4. Visualisation
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "leadlag_heatmap.png"


def plot_leadership_heatmap(result: LeadLagResult, labels=None, out_path=None):
    n = result.leadership_score.shape[0]
    if labels is None:
        labels = [f"Actif {i}" for i in range(n)]

    if out_path is None:
        output_path = DEFAULT_OUTPUT_PATH
    else:
        output_path = Path(out_path)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(result.optimal_lag * 1000, cmap="RdBu_r", vmin=-150, vmax=150)
    axes[0].set_title("Lag optimal (ms)\n(positif: ligne i mène colonne j)")
    axes[0].set_xticks(range(n)); axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_yticks(range(n)); axes[0].set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            axes[0].text(j, i, f"{result.optimal_lag[i,j]*1000:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(result.leadership_score, cmap="RdBu_r")
    axes[1].set_title("Score de leadership\n(positif: ligne i mène colonne j)")
    axes[1].set_xticks(range(n)); axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_yticks(range(n)); axes[1].set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            axes[1].text(j, i, f"{result.leadership_score[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nHeatmap sauvegardée: {output_path}")


# ---------------------------------------------------------------------------
# 5. Démo
# ---------------------------------------------------------------------------

def main():
    print("Simulation de 4 actifs (actif 0 = leader, lag=50ms)...")
    ticks = simulate_asynchronous_ticks(n_assets=4, duration_seconds=600.0, lag_ms=50.0)
    for k, df in enumerate(ticks):
        print(f"  actif {k}: {len(df)} ticks")

    print("\nCalcul de la matrice lead-lag (Hayashi-Yoshida)...")
    result = build_lead_lag_matrix(ticks, lags_ms=np.arange(-150, 151, 10))

    print("\nMatrice des lags optimaux (en ms), ligne i / colonne j:")
    print("  (positif => l'actif i mène l'actif j)")
    df_lag = pd.DataFrame(result.optimal_lag * 1000).round(0)
    print(df_lag)

    print("\nMatrice des scores de leadership (i mène j si positif):")
    df_score = pd.DataFrame(result.leadership_score).round(3)
    print(df_score)

    print("\nScore de leadership net par actif (somme des colonnes = 'à quel point l'actif mène les autres'):")
    net_leadership = result.leadership_score.sum(axis=1)
    for k, score in enumerate(net_leadership):
        print(f"  actif {k}: {score:+.3f}")

    plot_leadership_heatmap(result, out_path=DEFAULT_OUTPUT_PATH)


if __name__ == "__main__":
    main()
