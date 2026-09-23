"""
markowitz.py
============
Optimisation de portefeuille moyenne-variance, par formules FERMÉES
(théorème des deux fonds), plutôt que par un solveur QP numérique.

Pourquoi les formules fermées plutôt qu'un optimiseur ? 
Sans contrainte de vente à découvert, 
le problème moyenne-variance a une solution analytique exacte (algèbre matricielle simple),
utiliser un solveur itératif dans ce cas serait plus lent ET moins précis qu'une formule fermée. 
C'est un choix d'optimisation déliberé, 
dans le même esprit que le choix de Brennan-Schwartz plutôt que PSOR dans le projet EDP. 
La contrepartie, l'absence de contrainte long-only, est documentée dans le README comme limite assumée.

Notations (Merton, 1972) :
A = 1' Σ^-1 1
B = 1' Σ^-1 mu
C = mu' Σ^-1 mu
D = A*C - B^2
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class MarkowitzModel:
    mu: np.ndarray        # rendements espérés annualisés, shape (n,)
    cov: np.ndarray        # matrice de covariance annualisée, shape (n,n)
    asset_names: list

    def __post_init__(self):
        self.n = len(self.mu)
        self.cov_inv = np.linalg.inv(self.cov)
        ones = np.ones(self.n)
        self.A = ones @ self.cov_inv @ ones
        self.B = ones @ self.cov_inv @ self.mu
        self.C = self.mu @ self.cov_inv @ self.mu
        self.D = self.A * self.C - self.B ** 2

    """Portefeuille de variance minimale globale : w = Σ^-1 1 / A."""
    def min_variance_portfolio(self) -> np.ndarray:
        ones = np.ones(self.n)
        return (self.cov_inv @ ones) / self.A

    """
    Portefeuille efficient pour un rendement cible donné (formule
    fermée du théorème des deux fonds) :
        w = Σ^-1 [ (C - B*mu_p)/D * 1 + (A*mu_p - B)/D * mu ]
    """
    def efficient_portfolio(self, target_return: float) -> np.ndarray:

        ones = np.ones(self.n)
        lam = (self.C - self.B * target_return) / self.D
        gam = (self.A * target_return - self.B) / self.D
        return self.cov_inv @ (lam * ones + gam * self.mu)

    """
    Portefeuille de marché (Sharpe maximal), avec actif sans risque : w proportionnel à Σ^-1(mu - rf*1).
    """
    def tangency_portfolio(self, risk_free_rate: float) -> np.ndarray:
        ones = np.ones(self.n)
        excess = self.mu - risk_free_rate * ones
        raw = self.cov_inv @ excess
        return raw / np.sum(raw)   # normalisation : les poids somment à 1

    def portfolio_stats(self, w: np.ndarray, risk_free_rate: float = 0.0) -> tuple[float, float, float]:
        ret = float(w @ self.mu)
        vol = float(np.sqrt(w @ self.cov @ w))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else np.nan
        return ret, vol, sharpe

    """Variance minimale pour chaque niveau de rendement cible, sur toute la frontière."""
    def efficient_frontier(self, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
        mu_min_var = self.B / self.A
        target_returns = np.linspace(mu_min_var, np.max(self.mu) * 1.15, n_points)
        vols = np.array([
            np.sqrt((self.A * mp ** 2 - 2 * self.B * mp + self.C) / self.D)
            for mp in target_returns
        ])
        return target_returns, vols
