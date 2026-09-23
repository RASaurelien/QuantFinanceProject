"""
black_scholes.py
=================
Pricing Black-Scholes + inversion numérique de la volatilité implicite
(prix marché -> sigma).

Ce module est le point de passage obligé du pipeline de calibration :
le marché donne des PRIX (ou des cotations bid/ask), pas des
volatilités. Il faut d'abord inverser Black-Scholes pour obtenir la
volatilité implicite avant de pouvoir calibrer un modèle de smile
(SVI, SABR, ...) dessus.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

"""Prix Black-Scholes-Merton fermé (avec dividende continu q)."""
def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0, option_type: str = "call") -> float:
    if T <= 0 or sigma <= 0:
        # Cas dégénéré : valeur intrinsèque actualisée
        if option_type == "call":
            return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

"""
Inverse Black-Scholes : retrouve sigma tel que bs_price(...) == price.

Brent (bissection garantie convergente) est préféré à un Newton pur :
Newton peut diverger quand le prix marché est proche d'une borne d'arbitrage 
(option très OTM/ITM, vega quasi nulle). 
Brent est légèrement plus lent mais toujours robuste, 
critère retenu pour un pipeline qui doit tourner sans supervision, 
sur des centaines de cotations.
"""
def implied_vol(price: float, S: float, K: float, T: float, r: float,
                 q: float = 0.0, option_type: str = "call",
                 lo: float = 1e-4, hi: float = 5.0) -> float:
    intrinsic = (max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0) if option_type == "call"
                 else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0))

    if price <= intrinsic + 1e-12:
        return np.nan  # prix sous la borne d'arbitrage : cotation aberrante, on l'exclut

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, q, option_type) - price

    try:
        return brentq(objective, lo, hi, xtol=1e-8, maxiter=200)
    except ValueError:
        return np.nan  # pas de racine dans [lo, hi] : cotation illiquide/aberrante

"""Applique implied_vol() sur un ensemble de cotations d'une même maturité."""
def implied_vol_batch(prices: np.ndarray, S: float, strikes: np.ndarray, T: float,
                       r: float, q: float, option_types: np.ndarray) -> np.ndarray:
    out = np.empty(len(prices), dtype=float)
    for i in range(len(prices)):
        out[i] = implied_vol(prices[i], S, strikes[i], T, r, q, option_types[i])
    return out
