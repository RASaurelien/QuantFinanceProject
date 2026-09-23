"""
simulate.py
===========
Simule des evenements d'annonce de resultats trimestriels, avec :
- une fenetre d'ESTIMATION 
  (200 jours avant l'evenement, SANS aucune derive) 
  utilisee pour estimer le modele de marche de chaque firme (alpha, beta), 
  exactement comme en pratique.
- une fenetre d'EVENEMENT 
  (-20 a +70 jours autour de l'annonce, jour 0 = annonce) 
  ou une VRAIE derive post-annonce (PEAD) est injectee,
  proportionnelle a la surprise de resultat et amortie dans le temps
  (l'information s'incorpore progressivement dans le prix, 
  le phenomene empirique que Ball & Brown (1968), 
  puis Bernard & Thomas (1989) ont documente et qui reste, 
  encore aujourd'hui, 
  l'une des anomalies les plus robustes de la finance empirique).

Un groupe PLACEBO (surprise tiree aleatoirement mais AUCUNE derive
vraie injectee) sert de controle negatif pour la methodologie.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


EST_WINDOW = 200
PRE_EVENT = 20
POST_EVENT = 70
TOTAL_CAL_DAYS = EST_WINDOW + PRE_EVENT + POST_EVENT + 5

"""
Profil temporel CUMULÉ de la dérive : 
nul avant l'annonce (t<0), 
puis incorporation progressive de l'information (t>=0), 
amortie exponentiellement.
"""
def pead_decay(t: np.ndarray, half_life: float = 25.0) -> np.ndarray:
    return np.where(t >= 0, 1 - np.exp(-t / half_life), 0.0)

"""
Retourne un DataFrame avec, pour chaque evenement :
  surprise, is_placebo, est_returns (200,), market_est (200,),
  event_returns (91,), market_event (91,)
"""
def simulate_events(n_events: int = 2000, drift_scale: float = 0.02,
                     placebo_fraction: float = 0.3, seed: int = 42):
    rng = np.random.default_rng(seed)
    market_returns = rng.normal(0.0003, 0.01, size=TOTAL_CAL_DAYS)

    events = []
    t_event = np.arange(-PRE_EVENT, POST_EVENT + 1)

    for i in range(n_events):
        start = rng.integers(0, TOTAL_CAL_DAYS - EST_WINDOW - PRE_EVENT - POST_EVENT - 1)
        beta_i = rng.normal(1.0, 0.3)
        alpha_i = rng.normal(0.0, 0.0002)
        idio_vol = rng.uniform(0.01, 0.025)

        est_mkt = market_returns[start:start + EST_WINDOW]
        est_ret = alpha_i + beta_i * est_mkt + rng.normal(0, idio_vol, size=EST_WINDOW)

        ev_start = start + EST_WINDOW
        ev_mkt = market_returns[ev_start:ev_start + len(t_event)]

        surprise = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        is_placebo = rng.random() < placebo_fraction

        true_drift_cum = np.zeros(len(t_event)) if is_placebo else drift_scale * surprise * pead_decay(t_event)
        daily_drift = np.diff(np.concatenate([[0.0], true_drift_cum]))

        ev_ret = alpha_i + beta_i * ev_mkt + daily_drift + rng.normal(0, idio_vol, size=len(t_event))

        events.append({
            "event_id": i, "surprise": surprise, "is_placebo": is_placebo,
            "est_returns": est_ret, "market_est": est_mkt,
            "event_returns": ev_ret, "market_event": ev_mkt,
        })

    return pd.DataFrame(events), t_event
