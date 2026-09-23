"""
data_source.py
===============
Fournit un jeu de rendements multi-actifs, réel si possible (yfinance),
synthétique sinon, même logique de repli que le projet de calibration de surface de volatilité.

Univers par défaut : 6 classes d'actifs diversifiées (actions US,
actions internationales, obligations, or, émergents, immobilier coté),
pour que l'optimisation de portefeuille ait un intérêt réel (des
classes faiblement corrélées, pas 6 fois la même chose).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TICKERS = {
    "SPY": "Actions US",
    "EFA": "Actions internationales",
    "AGG": "Obligations US",
    "GLD": "Or",
    "EEM": "Actions émergentes",
    "VNQ": "Immobilier coté (REIT)",
}

"""Tente de récupérer des rendements quotidiens réels via yfinance."""
def fetch_real_returns(period: str = "3y") -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        print("[data_source] yfinance non installé -> pip install yfinance")
        return None

    try:
        data = yf.download(list(TICKERS.keys()), period=period, progress=False)["Close"]
        if data.empty:
            raise RuntimeError("Téléchargement vide.")
        returns = data.pct_change().dropna()
        if len(returns) < 100:
            raise RuntimeError("Historique trop court.")
        return returns

    except Exception as exc:
        print(f"[data_source] Récupération réelle impossible ({exc}) -> bascule synthétique.")
        return None

"""
Génère des rendements quotidiens synthétiques mais plausibles pour
les 6 classes d'actifs ci-dessus : volatilités annualisées et
corrélations réalistes (actions domestiques/internationales/
émergentes fortement corrélées entre elles, obligations légèrement
anti-corrélées aux actions, or quasi-décorrélé).
"""
def generate_synthetic_returns(n_days: int = 750, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assets = list(TICKERS.keys())

    annual_vol = np.array([0.18, 0.19, 0.05, 0.15, 0.23, 0.20])   # SPY, EFA, AGG, GLD, EEM, VNQ
    annual_mu = np.array([0.09, 0.07, 0.03, 0.05, 0.08, 0.07])     # rendements espérés annualisés

    # Matrice de corrélation plausible (symétrique, diagonale = 1)
    corr = np.array([
        #   SPY   EFA   AGG   GLD   EEM   VNQ
        [1.00, 0.85, -0.15, 0.05, 0.70, 0.65],
        [0.85, 1.00, -0.10, 0.10, 0.75, 0.60],
        [-0.15, -0.10, 1.00, 0.20, -0.10, 0.05],
        [0.05, 0.10, 0.20, 1.00, 0.15, 0.10],
        [0.70, 0.75, -0.10, 0.15, 1.00, 0.55],
        [0.65, 0.60, 0.05, 0.10, 0.55, 1.00],
    ])

    daily_vol = annual_vol / np.sqrt(252)
    daily_mu = annual_mu / 252
    cov_daily = np.outer(daily_vol, daily_vol) * corr

    daily_returns = rng.multivariate_normal(daily_mu, cov_daily, size=n_days)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days + 5)[-n_days:]

    return pd.DataFrame(daily_returns, index=dates, columns=assets)

"""
Poids de marché approximatifs (pour l'équilibre de Black-Litterman),
à défaut de vraies capitalisations flottantes. Ordre de grandeur
réaliste pour un portefeuille multi-actifs global (dominance
actions US, part significative d'obligations).
"""
def approximate_market_weights() -> pd.Series:
    weights = {"SPY": 0.35, "EFA": 0.15, "AGG": 0.30, "GLD": 0.05, "EEM": 0.08, "VNQ": 0.07}
    s = pd.Series(weights)
    return s / s.sum()


def build_dataset(use_real: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    returns = fetch_real_returns() if use_real else None
    if returns is None:
        returns = generate_synthetic_returns()
        print(f"[data_source] Rendements synthétiques générés ({len(returns)} jours, "
              f"{returns.shape[1]} actifs).")
    else:
        print(f"[data_source] Rendements réels récupérés ({len(returns)} jours, "
              f"{returns.shape[1]} actifs).")
    return returns, approximate_market_weights()
