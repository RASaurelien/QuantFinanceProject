"""
avellaneda_stoikov.py
=====================
Formules fermées d'Avellaneda & Stoikov (2008), 
"High-frequency trading in a limit order book" 
- le modèle de référence du market making avec gestion d'inventaire.

Le teneur de marché ne cote PAS symétriquement autour du prix milieu :
il décale ses cotations en fonction de son inventaire courant pour,
inciter le marché à le ramener vers une position neutre (q=0).

Prix de réservation (le "vrai" prix perçu par le MM, ajusté du risque d'inventaire) :
    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

Spread optimal total (bid-ask) :
    delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

Cotations : bid = r - delta/2 , ask = r + delta/2

s     : prix milieu courant
q     : inventaire courant (positif = long, négatif = short)
gamma : aversion au risque du MM (plus grand = skew plus agressif sur l'inventaire)
sigma : volatilité du prix milieu
k     : paramètre de décroissance de l'intensité d'arrivée des ordres,
        (lambda(delta) = A * exp(-k * delta), plus k est grand,
        plus s'éloigner du prix milieu fait chuter vite la probabilité d'exécution)
T-t   : temps restant avant la fin de la session
"""

from __future__ import annotations
import numpy as np


def reservation_price(s: float, q: int, gamma: float, sigma: float, time_left: float) -> float:
    return s - q * gamma * sigma ** 2 * time_left


def optimal_spread(gamma: float, sigma: float, time_left: float, k: float) -> float:
    inventory_term = gamma * sigma ** 2 * time_left
    liquidity_term = (2.0 / gamma) * np.log(1 + gamma / k)
    return inventory_term + liquidity_term

"""Retourne (bid, ask) selon Avellaneda-Stoikov."""
def as_quotes(s: float, q: int, gamma: float, sigma: float, time_left: float, k: float) -> tuple[float, float]:
    r = reservation_price(s, q, gamma, sigma, time_left)
    delta = optimal_spread(gamma, sigma, time_left, k)
    return r - delta / 2, r + delta / 2

"""Cotation symétrique naïve, ne dépendant PAS de l'inventaire, le comparatif du projet."""
def naive_quotes(s: float, half_spread: float) -> tuple[float, float]:
    return s - half_spread, s + half_spread
