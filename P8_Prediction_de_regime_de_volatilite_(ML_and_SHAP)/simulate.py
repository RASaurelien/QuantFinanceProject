"""
simulate.py
===========
Simule une série de rendements sous un modèle GBM à changement de régime de 
volatilité (Markov-switching), 
avec le régime VRAI connu à chaque date, 
ce qui permet de valider objectivement le classifieur ML plus loin 
(contrairement au marché réel, où le "régime" n'est jamais directement observable).

Deux régimes, persistants (matrice de transition à forte probabilité
de rester dans le même état, comme les vrais régimes de marché) :
  - régime 0 ("calme")  : vol annualisée basse
  - régime 1 ("stress")  : vol annualisée haute
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def simulate_regime_switching_returns(
    n_days: int = 2500,
    sigma_low: float = 0.10,
    sigma_high: float = 0.35,
    mu_low: float = 0.08,
    mu_high: float = -0.05,   # les phases de stress s'accompagnent souvent d'un drift négatif (asymétrie empirique)
    p_stay_low: float = 0.985,
    p_stay_high: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Retourne un DataFrame indexé par date, avec les colonnes :
      - 'return'  : rendement quotidien simulé
      - 'regime'  : régime VRAI (0=calme, 1=stress), connu par construction
    """
    rng = np.random.default_rng(seed)

    regimes = np.zeros(n_days, dtype=int)
    returns = np.zeros(n_days)

    state = 0   # on démarre en régime calme
    for t in range(n_days):
        # Transition markovienne : le régime du jour dépend seulement du régime de la veille
        if state == 0:
            state = 0 if rng.random() < p_stay_low else 1
        else:
            state = 1 if rng.random() < p_stay_high else 0

        regimes[t] = state
        sigma_annual = sigma_low if state == 0 else sigma_high
        mu_annual = mu_low if state == 0 else mu_high

        daily_sigma = sigma_annual / np.sqrt(252)
        daily_mu = mu_annual / 252
        returns[t] = daily_mu + daily_sigma * rng.standard_normal()

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days + 5)[-n_days:]
    return pd.DataFrame({"return": returns, "regime": regimes}, index=dates)
