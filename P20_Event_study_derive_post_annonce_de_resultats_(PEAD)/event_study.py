"""
event_study.py
==============
Méthodologie standard d'event study (MacKinlay, 1997, "Event Studies
in Economics and Finance") :

1. Modèle de marché estimé sur la fenêtre d'ESTIMATION 
   (avant l'événement, jamais sur la fenêtre d'événement, 
   sinon l'estimation elle-même absorberait la dérive qu'on cherche à mesurer) : 
   r_{i,t} = alpha_i + beta_i * r_{m,t} + eps_{i,t}
2. Rendement anormal (AR) dans la fenêtre d'événement :
   AR_{i,t} = r_{i,t} - (alpha_i_hat + beta_i_hat * r_{m,t})
3. Rendement anormal cumulé (CAR) : 
   somme des AR depuis le début de la fenêtre d'événement.
4. Test de significativité (Brown & Warner, 1985) : 
   t-test cross-sectionnel sur le CAR moyen à un horizon donné, 
   écart-type estimé à partir de la DISPERSION cross-sectionnellen, 
   des CAR individuels (pas de la fenêtre d'estimation).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

"""Estime (alpha, beta) sur la fenêtre d'estimation, retourne les AR sur la fenêtre d'événement."""
def market_model_ar(row: pd.Series) -> np.ndarray:
    X = np.column_stack([np.ones(len(row["market_est"])), row["market_est"]])
    y = row["est_returns"]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_hat, beta_hat = coef

    expected = alpha_hat + beta_hat * row["market_event"]
    return row["event_returns"] - expected

"""AR par événement (lignes) x par jour d'événement (colonnes), puis cumulé -> CAR."""
def compute_car_matrix(events: pd.DataFrame) -> np.ndarray:
    ar_matrix = np.vstack([market_model_ar(row) for _, row in events.iterrows()])
    car_matrix = np.cumsum(ar_matrix, axis=1)
    return car_matrix

"""Test de Brown & Warner (1985) : t-test cross-sectionnel du CAR moyen à un jour donné."""
def car_ttest(car_matrix: np.ndarray, day_idx: int) -> dict:
    car_at_day = car_matrix[:, day_idx]
    mean_car = car_at_day.mean()
    se = car_at_day.std(ddof=1) / np.sqrt(len(car_at_day))
    t_stat = mean_car / se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    return {"mean_car": mean_car, "se": se, "t_stat": t_stat, "p_value": p_value, "n": len(car_at_day)}
