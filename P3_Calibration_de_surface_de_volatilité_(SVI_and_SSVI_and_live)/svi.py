"""
svi.py
======
Paramétrisation SVI "raw" de Gatheral (2004) pour la variance totale implicite, 
et calibration par maturité sur un jeu de volatilités implicites de marché.

Pourquoi SVI plutôt qu'un simple spline sur le smile ? 
Un spline colle parfaitement aux points observés mais peut introduire de l'arbitrage
(convexité locale négative -> densité de probabilité implicite négative). 
SVI est une forme paramétrique restreinte à 5 paramètres, 
qui produit un smile lisse et pour laquelle il existe des conditions explicites de non-arbitrage, 
c'est la référence industrie pour interpoler/extrapoler un smile de façon utilisable en pricing/risque.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.optimize import least_squares


@dataclass
class SVIParams:
    a: float       # niveau global de variance totale
    b: float       # pente globale du smile (b >= 0)
    rho: float     # asymétrie / skew (-1 < rho < 1)
    m: float       # décalage horizontal du smile
    sigma: float   # convexité au voisinage de m (sigma > 0)

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

"""
Formule SVI "raw" :
    w(k) = a + b * ( rho*(k-m) + sqrt((k-m)^2 + sigma^2) )
où k = log(K/F) est la log-moneyness (F = forward) et w = sigma_impl^2 * T
est la variance totale implicite (pas la vol : on divise par T puis
on prend la racine carrée pour revenir à une vol annualisée).
"""
def svi_total_variance(k: np.ndarray, params: SVIParams) -> np.ndarray:
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_implied_vol(k: np.ndarray, T: float, params: SVIParams) -> np.ndarray:
    """Convertit la variance totale SVI en volatilité implicite annualisée."""
    w = svi_total_variance(k, params)
    return np.sqrt(np.maximum(w, 1e-12) / T)


# Bornes des paramètres : garantissent des smiles économiquement
# plausibles (b positif = smile qui s'ouvre vers le haut, |rho|<1,
# sigma>0 = convexité non dégénérée) sans imposer de valeur précise.
_BOUNDS_LOWER = np.array([-1.0, 1e-6, -0.999, -2.0, 1e-4])
_BOUNDS_UPPER = np.array([1.0, 5.0, 0.999, 2.0, 2.0])

"""
Calibre les 5 paramètres SVI sur UNE tranche de maturité T, par
moindres carrés pondérés sur la variance totale (plus stable
numériquement que de calibrer directement sur la vol, qui explose
en 1/sqrt(T) pour les très courtes maturités).

weights permet de sur-pondérer les strikes liquides (proches de la
monnaie) si l'on dispose de volumes ou de spreads bid/ask.
"""
def calibrate_svi_slice(k: np.ndarray, T: float, market_iv: np.ndarray,
                         weights: np.ndarray = None) -> SVIParams:
    mask = ~np.isnan(market_iv)
    k_fit = k[mask]
    w_market = (market_iv[mask] ** 2) * T   # variance totale observée

    w_fit = np.ones_like(w_market) if weights is None else weights[mask]

    # Point de départ raisonnable : niveau proche du minimum observé,
    # skew négatif modéré (typique actions), pas de décalage.
    a0 = float(np.min(w_market)) * 0.9
    x0 = np.clip(np.array([a0, 0.15, -0.3, 0.0, 0.15]), _BOUNDS_LOWER, _BOUNDS_UPPER)

    def residuals(x: np.ndarray) -> np.ndarray:
        p = SVIParams(*x)
        w_model = svi_total_variance(k_fit, p)
        return w_fit * (w_model - w_market)

    result = least_squares(residuals, x0, bounds=(_BOUNDS_LOWER, _BOUNDS_UPPER),
                            method="trf", xtol=1e-12, ftol=1e-12, max_nfev=5000)

    return SVIParams(*result.x)

"""
Variance totale minimale atteinte par la paramétrisation SVI (au
point k* = m - rho*sigma/sqrt(1-rho^2)). Doit être >= 0 : sinon la
surface implique une variance négative quelque part, donc un
arbitrage trivial.
"""
def min_total_variance(params: SVIParams) -> float:
    return params.a + params.b * params.sigma * np.sqrt(1.0 - params.rho ** 2)

"""
Vérifie l'absence d'arbitrage de calendrier au sens simple : la
variance totale w(k, T) doit être croissante en T pour un k donné
(deux tranches de maturité ne doivent jamais se croiser). Condition
nécessaire (pas suffisante) de non-arbitrage, mais garde-fou très
utile puisque chaque tranche est calibrée indépendamment.
"""
def check_calendar_arbitrage(params_by_maturity: dict, k_grid: np.ndarray) -> bool:
    maturities = sorted(params_by_maturity.keys())
    surfaces = np.array([svi_total_variance(k_grid, params_by_maturity[T]) for T in maturities])
    return bool(np.all(np.diff(surfaces, axis=0) >= -1e-8))
