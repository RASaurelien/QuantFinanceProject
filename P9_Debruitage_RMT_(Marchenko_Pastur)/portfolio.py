"""
portfolio.py
============
Portefeuille de variance minimale globale 
(formule fermée, même théorème des deux fonds que le projet Markowitz), 
utilisé ici comme BANC D'ESSAI pour juger objectivement, 
l'intérêt du débruitage RMT : moins de bruit dans la covariance estimée,
doit se traduire par un portefeuille dont la variance RÉELLE 
(mesurée avec la vraie covariance, 
inconnue en pratique mais connue ici par construction) 
est plus proche de l'optimum théorique.
"""

from __future__ import annotations
import numpy as np

"""w = Sigma^-1 1 / (1' Sigma^-1 1), portefeuille de variance minimale globale, sans contrainte."""
def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    ones = np.ones(n)
    cov_inv = np.linalg.inv(cov)
    raw = cov_inv @ ones
    return raw / (ones @ raw)

"""Variance RÉELLE d'un portefeuille, évaluée avec la vraie covariance (inconnue en pratique)."""
def realized_variance(weights: np.ndarray, true_cov: np.ndarray) -> float:
    return float(weights @ true_cov @ weights)
