"""
var_methods.py
==============
Trois méthodes standard de calcul de la VaR (Value-at-Risk), 
et de l'ES (Expected Shortfall / CVaR), 
appliquées à une fenêtre glissante de rendements passés :

1. Paramétrique (gaussienne) : suppose des rendements normaux,rapide, 
   mais ignore les queues épaisses réelles des marchés.
2. Historique : quantile empirique des rendements passés, 
   aucune hypothèse de distribution, 
   mais suppose que le passé récent est représentatif du futur.
3. Monte Carlo (Student-t ajustée) : ajuste une loi de Student sur la fenêtre (MLE), 
   puis simule un grand nombre de tirages pour estimer le quantile, 
   capture les queues épaisses tout en lissant le bruit d'échantillonnage historique.

Convention : VaR et ES sont retournées en PERTE positive 
(une VaR de 0.03 signifie "3% de perte potentielle"), 
convention la plus lisible pour un rapport de risque.
"""

from __future__ import annotations
import numpy as np
from scipy import stats

"""VaR/ES gaussienne en forme fermée (pas de simulation nécessaire)."""
def parametric_var_es(returns: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(alpha)
    var = -(mu - z * sigma)
    # ES gaussienne : formule fermée classique, E[X | X < -VaR] sous normalité
    es = -(mu - sigma * stats.norm.pdf(z) / (1 - alpha))
    return var, es

"""VaR/ES par simulation historique : quantile empirique, aucune hypothèse de distribution."""
def historical_var_es(returns: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    losses = -returns
    var = np.quantile(losses, alpha)
    es = losses[losses >= var].mean()
    return var, es

"""
VaR/ES Monte Carlo : ajuste une loi de Student (MLE) sur la fenêtre, 
simule n_sim tirages, puis calcule VaR/ES comme pour la méthode historique, 
mais sur les tirages SIMULÉS plutôt que sur les données observées, 
capture les queues épaisses (contrairement à la méthode gaussienne), 
tout en lissant le bruit d'échantillonnage
(contrairement à la méthode historique, 
limitée par la taille de la fenêtre).
"""
def monte_carlo_var_es(returns: np.ndarray, alpha: float = 0.99,
                         n_sim: int = 20_000, seed: int = 0) -> tuple[float, float]:
    nu, loc, scale = stats.t.fit(returns)
    nu = max(nu, 2.5)   # garde-fou : en dessous de nu=2, la variance n'existe même plus

    rng = np.random.default_rng(seed)
    simulated = stats.t.rvs(nu, loc=loc, scale=scale, size=n_sim, random_state=rng)

    losses = -simulated
    var = np.quantile(losses, alpha)
    es = losses[losses >= var].mean()
    return var, es
