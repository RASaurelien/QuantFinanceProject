"""
fama_macbeth.py
===============
Procedure de Fama & MacBeth (1973), en deux etapes :

Etape 1 (cross-sectionnelle, repetee a chaque periode t) :
    r_{i,t} = lambda_{0,t} + sum_k lambda_{k,t} * exposure_{i,k} + e_{i,t}
    Une regression OLS PAR PERIODE sur la coupe transversale des actifs
-> donne une serie temporelle d'estimateurs lambda_{k,t} 
(une prime de facteur par mois).

Etape 2 (serie temporelle) :
    lambda_k = moyenne_t( lambda_{k,t} )
La prime de facteur estimee est la moyenne de la serie mensuelle.
L'erreur-type NE PEUT PAS supposer l'independance temporelle 
(les lambda_{k,t} peuvent etre autocorreles) 
-- d'ou la correction de Newey-West 
(HAC : Heteroskedasticity and Autocorrelation Consistent).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

"""Une regression OLS par periode t : r_t (n_assets,) ~ [1, exposures]. Retourne la serie des lambda_t."""
def cross_sectional_regressions(returns: pd.DataFrame, exposures: pd.DataFrame) -> pd.DataFrame:
    X = np.column_stack([np.ones(len(exposures)), exposures.to_numpy()])
    XtX_inv = np.linalg.inv(X.T @ X)   # identique a chaque periode (expositions fixes) -> calcule une seule fois

    lambdas = np.zeros((len(returns), X.shape[1]))
    for t in range(len(returns)):
        y = returns.iloc[t].to_numpy()
        lambdas[t] = XtX_inv @ X.T @ y

    cols = ["Intercept"] + list(exposures.columns)
    return pd.DataFrame(lambdas, columns=cols)

"""
Erreur-type de la moyenne d'une série temporelle, 
corrigée Newey-West (1987) pour l'autocorrélation jusqu'à `max_lags` retards
(règle de Newey-West 1994 si non spécifié : floor(4*(T/100)^(2/9))).
"""
def newey_west_se(series: np.ndarray, max_lags: int = None) -> float:
    T = len(series)
    if max_lags is None:
        max_lags = int(np.floor(4 * (T / 100) ** (2 / 9)))

    demeaned = series - series.mean()
    gamma0 = np.sum(demeaned ** 2) / T
    variance = gamma0

    for lag in range(1, max_lags + 1):
        weight = 1 - lag / (max_lags + 1)   # noyau de Bartlett
        gamma_lag = np.sum(demeaned[lag:] * demeaned[:-lag]) / T
        variance += 2 * weight * gamma_lag

    return np.sqrt(variance / T)

"""Moyenne, erreur-type Newey-West, t-stat et p-value (asymptotique normale) pour chaque facteur."""
def fama_macbeth_summary(lambdas: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats

    rows = []
    for col in lambdas.columns:
        series = lambdas[col].to_numpy()
        mean_premium = series.mean()
        se = newey_west_se(series)
        t_stat = mean_premium / se
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        rows.append({"factor": col, "premium_monthly": mean_premium, "se_newey_west": se,
                     "t_stat": t_stat, "p_value": p_value})

    return pd.DataFrame(rows).set_index("factor")
