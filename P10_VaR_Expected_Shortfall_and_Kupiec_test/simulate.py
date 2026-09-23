"""
simulate.py
===========
Simule une série de rendements sous un processus GARCH(1,1) 
à innovations de Student, pas gaussiennes, pour générer des données,
avec de vraies queues épaisses et du clustering de volatilité, 
comme un marché réel. 
C'est volontaire : le but du projet est de montrer que la VaR gaussienne, 
SOUS-ESTIME le risque de queue sur ce type de données, 
ce qui n'aurait aucun sens à démontrer sur des données déjà gaussiennes par construction.
"""

from __future__ import annotations
import numpy as np
from scipy import stats

"""
GARCH(1,1) avec innovations de Student standardisées (variance
unitaire), nu = degrés de liberté (nu petit = queues plus épaisses ;
nu=5 est une valeur empiriquement plausible pour des rendements
actions quotidiens).
"""
def simulate_garch_t(n_days: int = 3000, omega: float = 0.02, alpha: float = 0.08,
                       beta: float = 0.90, nu: float = 5.0, burn_in: int = 500,
                       seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_total = n_days + burn_in

    # Standardisation : une Student(nu) a une variance de nu/(nu-2), pas 1.
    # On la renormalise pour que sigma_t soit bien l'écart-type conditionnel.
    t_scale = np.sqrt((nu - 2) / nu)

    sigma2 = np.zeros(n_total)
    r = np.zeros(n_total)
    sigma2[0] = omega / (1 - alpha - beta)
    r[0] = np.sqrt(sigma2[0]) * stats.t.rvs(nu, random_state=rng) * t_scale

    for t in range(1, n_total):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        z = stats.t.rvs(nu, random_state=rng) * t_scale
        r[t] = np.sqrt(sigma2[t]) * z

    return r[burn_in:]
