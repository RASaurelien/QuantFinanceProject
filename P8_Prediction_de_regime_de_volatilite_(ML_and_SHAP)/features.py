"""
features.py
===========
Construit des features de marché à partir des rendements PASSÉS
uniquement (fenêtres glissantes strictement rétrospectives), pour
prédire le régime de volatilité du jour SUIVANT, une vraie tâche de
nowcasting/prévision, pas une classification a posteriori qui
tricherait en utilisant de l'information future.

Toutes les features sont des quantités qu'un desk de trading peut
calculer EN TEMPS RÉEL avec les données déjà disponibles à la clôture
du jour t.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def build_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    r = df["return"]

    features = pd.DataFrame(index=df.index)

    # --- Volatilité réalisée, plusieurs horizons (annualisée) ---
    for window in [5, 10, 20, 60]:
        features[f"vol_{window}d"] = r.rolling(window).std() * np.sqrt(252)

    # --- Ratio vol courte / vol longue : détecte un changement de regime en cours ---
    features["vol_ratio_5_60"] = features["vol_5d"] / features["vol_60d"]
    features["vol_ratio_10_60"] = features["vol_10d"] / features["vol_60d"]

    # --- Rendement moyen glissant (le stress s'accompagne souvent d'un drift négatif) ---
    features["mean_return_5d"] = r.rolling(5).mean()
    features["mean_return_20d"] = r.rolling(20).mean()

    # --- Asymétrie et aplatissement glissants (les queues de distribution s'épaississent en régime de stress) ---
    features["skew_20d"] = r.rolling(20).skew()
    features["kurtosis_20d"] = r.rolling(20).kurt()

    # --- Autocorrélation glissante (proxy simple, sans fuite temporelle) ---
    features["autocorr_20d"] = r.rolling(21).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) == 21 else np.nan, raw=True
    )

    # --- Choc le plus récent et compteur de jours extrêmes ---
    features["abs_return_lag1"] = r.shift(1).abs()
    rolling_std_20 = r.rolling(20).std()
    features["n_extreme_days_20d"] = (r.abs() > 2 * rolling_std_20).rolling(20).sum()

    # --- Drawdown maximal glissant sur 20 jours (proxy de stress cumulatif) ---
    cum_returns = (1 + r).cumprod()
    rolling_max = cum_returns.rolling(20).max()
    features["max_drawdown_20d"] = (cum_returns / rolling_max - 1).rolling(20).min()

    # --- Cible : régime du jour SUIVANT (t+1), connu seulement a posteriori ---
    target = df["regime"].shift(-1)
    target.name = "next_regime"

    # On retire les lignes avec NaN (warmup des fenêtres glissantes en début
    # de série, et la toute dernière ligne dont la cible future est inconnue).
    valid = features.notna().all(axis=1) & target.notna()
    return features[valid], target[valid].astype(int)
