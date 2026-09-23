"""
simulate.py
===========
Simule un panel (actifs x temps) de rendements mensuels sous un modele a 4 facteurs de style
(Value, Momentum, Quality, Low-Vol) + un 5eme facteur PLACEBO 
(bruit pur, prime vraie = 0), pour valider objectivement la methodologie de Fama-MacBeth : 
si elle retrouve les vraies primes ET rejette correctement le placebo, 
la methodologie est correctement implementee.

Chaque actif a une exposition FIXE 
(standardisee, comme une caracteristique z-scoree) a chaque facteur 
-- simplification deliberee par rapport a la pratique reelle 
(ou les caracteristiques sont recalculees chaque mois a partir des fondamentaux), 
qui garde le test de la methodologie Fama-MacBeth elle-meme propre et isolable.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

FACTOR_NAMES = ["Value", "Momentum", "Quality", "LowVol", "Placebo"]

# Primes mensuelles VRAIES (les 4 premieres sont des ordres de grandeur
# typiquement cites dans la litterature US ; la 5eme -- Placebo -- est
# un bruit pur, prime vraie = 0).
TRUE_PREMIA = np.array([0.0030, 0.0050, 0.0020, 0.0015, 0.0000])


def simulate_panel(n_assets: int = 500, n_periods: int = 240, seed: int = 42):
    """
    Retourne :
      returns   : DataFrame (n_periods, n_assets) de rendements mensuels
      exposures : DataFrame (n_assets, 5) expositions factorielles (standardisees, fixes dans le temps)
    """
    rng = np.random.default_rng(seed)

    exposures = rng.standard_normal((n_assets, len(FACTOR_NAMES)))
    exposures = (exposures - exposures.mean(axis=0)) / exposures.std(axis=0)

    idio_vol = rng.uniform(0.04, 0.10, size=n_assets)

    returns = np.zeros((n_periods, n_assets))
    for t in range(n_periods):
        factor_shock = rng.standard_normal(len(FACTOR_NAMES)) * 0.02
        period_premia = TRUE_PREMIA + factor_shock
        systematic = exposures @ period_premia
        idio = rng.standard_normal(n_assets) * idio_vol
        returns[t] = systematic + idio

    returns_df = pd.DataFrame(returns, columns=[f"asset_{i}" for i in range(n_assets)])
    exposures_df = pd.DataFrame(exposures, columns=FACTOR_NAMES,
                                  index=[f"asset_{i}" for i in range(n_assets)])
    return returns_df, exposures_df
