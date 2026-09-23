"""
scanner.py
==========
Scanne TOUTES les paires d'un univers d'actifs 
(pas une paire choisie à l'avance) pour détecter la cointégration, 
via Engle-Granger 
(2 étapes : 
régression de long terme + test ADF sur le résidu). 
Classe les paires par force du signal 
(t-stat ADF le plus négatif = spread le plus clairement stationnaire)
"""

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

"""T-stat ADF simplifié (constante, sans tendance)."""
def adf_tstat(spread: np.ndarray, lags: int = 1) -> float:
    dy = np.diff(spread)
    y_lag = spread[:-1]
    n = len(dy)

    X_cols = [np.ones(n - lags), y_lag[lags:]]
    for i in range(1, lags + 1):
        X_cols.append(dy[lags - i:n - i])
    X = np.column_stack(X_cols)
    y = dy[lags:]

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = n - lags - X.shape[1]
    sigma2 = np.sum(resid ** 2) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(sigma2 * XtX_inv[1, 1])
    return coef[1] / se_beta


def scan_pairs(prices: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    log_prices = np.log(prices)
    results = []

    for i, j in itertools.combinations(range(prices.shape[1]), 2):
        p1, p2 = log_prices.iloc[:, i].to_numpy(), log_prices.iloc[:, j].to_numpy()
        beta = np.polyfit(p2, p1, 1)[0]
        spread = p1 - beta * p2
        t_stat = adf_tstat(spread)
        results.append({"asset_i": i, "asset_j": j, "beta": beta, "adf_tstat": t_stat})

    df = pd.DataFrame(results).sort_values("adf_tstat")   # plus négatif = plus stationnaire
    return df.head(top_n).reset_index(drop=True)
