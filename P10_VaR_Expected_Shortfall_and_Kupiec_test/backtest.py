"""
backtest.py
===========
Backtest glissant ("rolling window") de chaque méthode de VaR : 
à chaque date t, la VaR est estimée UNIQUEMENT à partir des rendements passés, 
[t-W, t-1], puis comparée au rendement RÉALISÉ au jour t. 
C'est la seule façon honnête de tester une méthode de VaR, 
l'utiliser sur la même fenêtre que celle qui a servi à l'estimer serait, 
un backtest en échantillon, 
qui ne dit rien de la performance réelle en production.

Test de Kupiec (1995), "Proportion of Failures" (POF) : 
sous H0, le nombre de dépassements de VaR observés suit une loi binomiale, 
de paramètre p = 1-alpha (le taux de dépassement attendu). 
Le test du rapport de vraisemblance compare le taux de dépassement OBSERVÉ, 
à ce taux théorique.
"""

from __future__ import annotations
import numpy as np
from scipy import stats

from var_methods import parametric_var_es, historical_var_es, monte_carlo_var_es


"""
Calcule la VaR glissante pour chaque méthode et compte les
dépassements (jours où la perte réalisée dépasse la VaR prédite
la veille, avec seulement l'information disponible ce jour-là).

Retourne un dict {method_name: {"var_series": ..., "breaches": ...}}
"""
def rolling_backtest(returns: np.ndarray, window: int, alpha: float = 0.99,
                       methods: dict = None) -> dict:
    if methods is None:
        methods = {
            "Gaussienne": parametric_var_es,
            "Historique": historical_var_es,
            "Monte Carlo (Student-t)": monte_carlo_var_es,
        }

    n = len(returns)
    results = {name: {"var_series": np.full(n, np.nan), "breaches": np.zeros(n, dtype=bool)}
               for name in methods}

    for t in range(window, n):
        window_returns = returns[t - window:t]
        realized = returns[t]

        for name, func in methods.items():
            var_t, _ = func(window_returns, alpha=alpha)
            results[name]["var_series"][t] = var_t
            results[name]["breaches"][t] = (-realized) > var_t   # perte réalisée > VaR prédite

    return results

"""
Test du rapport de vraisemblance de Kupiec (1995) :
    H0 : le taux de dépassement observé = 1 - alpha (le modèle est bien calibré)
    LR = -2 * ln[ (1-p)^(n-x) p^x / (1-x/n)^(n-x) (x/n)^x ]  ~ chi2(1) sous H0
"""
def kupiec_pof_test(n_obs: int, n_breaches: int, alpha: float = 0.99) -> dict:
    p = 1 - alpha
    x = n_breaches
    n = n_obs
    p_hat = x / n

    if x == 0:
        log_num = (n - x) * np.log(1 - p)
        log_den = (n - x) * np.log(1 - p_hat) if p_hat < 1 else 0.0
    else:
        log_num = (n - x) * np.log(1 - p) + x * np.log(p)
        log_den = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)

    lr_stat = -2 * (log_num - log_den)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return {
        "n_obs": n, "n_breaches": x, "expected_breaches": n * p,
        "observed_rate": p_hat, "expected_rate": p,
        "lr_stat": lr_stat, "p_value": p_value,
        "reject_5pct": p_value < 0.05,   # rejet de H0 = modèle mal calibré
    }
