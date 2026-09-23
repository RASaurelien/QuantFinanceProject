"""
universe.py
===========
Simule un univers de 20 actifs : 
3 paires VRAIMENT cointégrées 
(tendance commune + bruit idiosyncratique stationnaire AR(1)) 
noyées parmi 14 actifs indépendants, 
pour que le scanner de cointégration ait un vrai travail de sélection à faire 
(comme un trader qui scanne un univers,
pas juste 2 actifs choisis à l'avance).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

COINTEGRATED_PAIRS = [(0, 1), (2, 3), (4, 5)]   # indices des 3 vraies paires


def simulate_universe(n_days: int = 1500, n_independent: int = 14, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_pairs_assets = len(COINTEGRATED_PAIRS) * 2
    n_assets = n_pairs_assets + n_independent

    log_prices = np.zeros((n_days, n_assets))
    common_trends = np.cumsum(rng.normal(0, 0.4, size=(n_days, len(COINTEGRATED_PAIRS))), axis=0)
    betas = rng.uniform(0.8, 1.3, size=len(COINTEGRATED_PAIRS))

    for k, (i, j) in enumerate(COINTEGRATED_PAIRS):
        # Bruit idiosyncratique STATIONNAIRE (AR(1) avec |phi|<1) : chaque actif
        # suit la tendance commune, mais l'écart individuel à cette tendance
        # ne diverge jamais -- c'est exactement la définition de la cointégration.
        stat_noise_i = np.zeros(n_days)
        stat_noise_j = np.zeros(n_days)
        for t in range(1, n_days):
            stat_noise_i[t] = 0.9 * stat_noise_i[t - 1] + rng.normal(0, 0.3)
            stat_noise_j[t] = 0.9 * stat_noise_j[t - 1] + rng.normal(0, 0.3)
        log_prices[:, i] = common_trends[:, k] + stat_noise_i
        log_prices[:, j] = betas[k] * common_trends[:, k] + stat_noise_j

    for a in range(n_pairs_assets, n_assets):
        log_prices[:, a] = np.cumsum(rng.normal(0.0002, 0.02, size=n_days))   # marche aléatoire indépendante

    prices = 100 * np.exp(log_prices - log_prices[0])
    cols = [f"asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, columns=cols)
