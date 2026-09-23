"""
xva.py
======
Calcul des profils d'exposition (EE, ENE, PFE) à partir de la matrice
de MtM simulée, puis de la CVA et de la DVA par intégration discrète
sur la courbe de survie de la contrepartie / de la banque elle-même.

Convention actuarielle standard (Gregory, *The xVA Challenge*) :

CVA = (1 - R_c) * sum_k  EE(t_k) * P(0,t_k) * PD_c(t_{k-1}, t_k)
DVA = (1 - R_b) * sum_k ENE(t_k) * P(0,t_k) * PD_b(t_{k-1}, t_k)

où PD(t_{k-1}, t_k) est la probabilité de défaut marginale sur l'intervalle, 
obtenue à partir d'une courbe de survie exponentielle à intensité de défaut 
(hasard) constante, 
elle-même calibrée sur un spread de CDS, 
via l'approximation standard  lambda ~= spread / (1-R).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

"""Approximation standard (credit triangle) : lambda ~= spread / (1-R)."""
def hazard_rate_from_cds(spread_bps: float, recovery: float) -> float:
    spread = spread_bps * 1e-4
    return spread / (1.0 - recovery)

"""Courbe de survie exponentielle à intensité de défaut constante."""
def survival_probability(hazard_rate: float, t: np.ndarray) -> np.ndarray:
    return np.exp(-hazard_rate * t)


@dataclass
class ExposureProfile:
    times: np.ndarray
    ee: np.ndarray       # Expected Exposure : E[max(MtM,0)]
    ene: np.ndarray       # Expected Negative Exposure : E[min(MtM,0)]
    pfe_95: np.ndarray     # Potential Future Exposure, quantile 95%
    epe: float             # Expected Positive Exposure moyenne dans le temps (résumé scalaire)

"""mtm_matrix : (n_paths, n_times). Colonne 0 = t=0 (doit valoir ~0 pour un swap "par")."""
def compute_exposure_profile(times: np.ndarray, mtm_matrix: np.ndarray) -> ExposureProfile:
    ee = np.mean(np.maximum(mtm_matrix, 0.0), axis=0)
    ene = np.mean(np.minimum(mtm_matrix, 0.0), axis=0)
    pfe_95 = np.quantile(np.maximum(mtm_matrix, 0.0), 0.95, axis=0)
    epe = float(np.mean(ee))
    return ExposureProfile(times=times, ee=ee, ene=ene, pfe_95=pfe_95, epe=epe)

"""CVA unilatérale, actualisée à la courbe déterministe P(0,t) = exp(-r*t)."""
def compute_cva(profile: ExposureProfile, discount_rate: float,
                 hazard_rate: float, recovery: float) -> float:
    t = profile.times
    discount = np.exp(-discount_rate * t)
    survival = survival_probability(hazard_rate, t)

    survival_prev = np.concatenate([[1.0], survival[:-1]])
    marginal_pd = survival_prev - survival   # PD sur chaque intervalle [t_{k-1}, t_k]

    return float((1.0 - recovery) * np.sum(profile.ee * discount * marginal_pd))


"""
DVA : symétrique de la CVA, mais sur l'exposition NÉGATIVE (ce que
la banque doit à la contrepartie) et le risque de défaut PROPRE de
la banque, c'est un gain comptable (la banque "profite" de son
propre risque de crédit), d'où le signe négatif conventionnel de
ene dans profile qu'on neutralise ici en prenant la valeur absolue.
"""
def compute_dva(profile: ExposureProfile, discount_rate: float,
                 hazard_rate_own: float, recovery_own: float) -> float:
    t = profile.times
    discount = np.exp(-discount_rate * t)
    survival = survival_probability(hazard_rate_own, t)

    survival_prev = np.concatenate([[1.0], survival[:-1]])
    marginal_pd = survival_prev - survival

    return float((1.0 - recovery_own) * np.sum(np.abs(profile.ene) * discount * marginal_pd))
