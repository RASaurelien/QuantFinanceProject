"""
strategy.py
===========
Pairs trading, pas juste un signal z-score, 
mais une vraie gestion du risque par position :

- Entrée quand |z-score du spread| > z_entry
- Sortie quand |z-score| < z_exit (retour à la moyenne réalisé)
- STOP-LOSS quand |z-score| > z_stop (le spread diverge encore plus
    au lieu de revenir, 
signe possible que la relation de cointégration s'est cassée, pas juste du bruit normal)
- Sizing INVERSEMENT PROPORTIONNEL à la volatilité du spread
(contribution au risque égale entre paires, pas une quantité fixe)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

"""
Retourne le P&L quotidien de la stratégie sur cette paire 
(en unités de "budget de risque"), 
et le nombre de stop-loss déclenchés.
"""
def run_pair_strategy(p1: np.ndarray, p2: np.ndarray, beta: float,
                        z_entry: float = 2.0, z_exit: float = 0.5, z_stop: float = 3.5,
                        window: int = 60, risk_budget: float = 1.0, use_risk_mgmt: bool = True):
    log_p1, log_p2 = np.log(p1), np.log(p2)
    spread = log_p1 - beta * log_p2

    roll_mean = pd.Series(spread).rolling(window).mean().to_numpy()
    roll_std = pd.Series(spread).rolling(window).std().to_numpy()
    z = (spread - roll_mean) / roll_std

    # Vol de référence (médiane sur la période) : le sizing inverse-vol
    # est exprimé RELATIVEMENT à cette référence, avec un plafond de
    # levier -- sans ce garde-fou, une vol de spread qui s'effondre
    # temporairement (roll_std -> 0) ferait exploser la taille de
    # position, ce qui est le contraire de l'effet recherché.
    target_vol = np.nanmedian(roll_std)

    position = 0   # +1 : long spread (long actif1, short actif2) ; -1 : inverse
    pnl = np.zeros(len(spread))
    n_stops = 0

    spread_ret = np.diff(spread, prepend=spread[0])   # approx. rendement du spread (échelle log)

    for t in range(window, len(spread)):
        if np.isnan(z[t]):
            continue

        if use_risk_mgmt:
            leverage_ratio = np.clip(target_vol / max(roll_std[t], 1e-6), 0.3, 3.0)
            size = risk_budget * leverage_ratio
        else:
            size = risk_budget

        if position == 0:
            if z[t] > z_entry:
                position = -1
            elif z[t] < -z_entry:
                position = 1
        else:
            hit_stop = use_risk_mgmt and abs(z[t]) > z_stop
            hit_exit = abs(z[t]) < z_exit
            if hit_stop:
                n_stops += 1
                position = 0
            elif hit_exit:
                position = 0

        pnl[t] = position * size * spread_ret[t]

    return pnl, n_stops
