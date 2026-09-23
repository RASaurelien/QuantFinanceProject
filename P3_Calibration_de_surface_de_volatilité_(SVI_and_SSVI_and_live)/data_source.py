"""
data_source.py
===============
Fournit les cotations d'options à calibrer, de deux façons :

1. fetch_real_chain() : tente de récupérer une vraie chaîne d'options
   via yfinance (Yahoo Finance). 
   Nécessite un accès réseau sortant et l'installation de yfinance (pip install yfinance).

2. generate_synthetic_chain() : 
    construit un jeu de données synthétique MAIS réaliste, par un aller-retour complet :  
        vol "vraie" (SVI) -> prix BS + bruit de cotation -> implied vol
   C'est volontairement le même chemin qu'un pipeline de marché réel
   (le marché ne donne jamais directement une vol, seulement des prix),
   ce qui permet de tester honnêtement le calibrateur, 
   y compris son étape d'inversion de volatilité implicite.

Le fallback synthétique tourne partout, sans dépendance réseau, 
ce qui permet de valider et démontrer le pipeline même hors connexion.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from black_scholes import bs_price, implied_vol_batch
from svi import SVIParams, svi_implied_vol


""" 
Tente de récupérer la chaîne d'options réelle d'un ticker via,
yfinance. Retourne None (plutôt que de lever une exception),
si le réseau est indisponible, pour laisser l'appelant basculer proprement, 
sur les données synthétiques.
"""

def fetch_real_chain(ticker: str, r: float = 0.04) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        print("[data_source] yfinance non installé -> pip install yfinance")
        return None

    try:
        tk = yf.Ticker(ticker)
        spot = tk.history(period="1d")["Close"].iloc[-1]
        expiries = tk.options
        if not expiries:
            raise RuntimeError("Aucune échéance d'option disponible pour ce ticker.")

        rows = []
        today = pd.Timestamp.today().normalize()
        for expiry in expiries:
            T = (pd.Timestamp(expiry) - today).days / 365.0
            if T <= 0:
                continue
            chain = tk.option_chain(expiry)
            for df, opt_type in [(chain.calls, "call"), (chain.puts, "put")]:
                for _, row in df.iterrows():
                    mid = (row["bid"] + row["ask"]) / 2.0 if row["bid"] > 0 and row["ask"] > 0 else row["lastPrice"]
                    if mid <= 0:
                        continue
                    rows.append({"T": T, "K": row["strike"], "price": mid,
                                 "option_type": opt_type, "S": spot, "r": r, "q": 0.0})
        if not rows:
            raise RuntimeError("Chaîne d'options vide.")
        return pd.DataFrame(rows)

    except Exception as exc:  # réseau indisponible, ticker invalide, etc.
        print(f"[data_source] Récupération réelle impossible ({exc}) -> bascule synthétique.")
        return None

"""
Génère une chaîne d'options synthétique par aller-retour complet
vol -> prix bruité -> implied vol recalculée, avec une surface SVI
"vraie" plausible pour un sous-jacent actions (skew négatif,
niveau et convexité croissants avec la maturité).
noise_bps : bruit de cotation appliqué aux PRIX (en points de base relatifs), 
pour simuler un spread bid/ask réaliste.
"""
def generate_synthetic_chain(S0: float = 100.0, r: float = 0.03, q: float = 0.01,
                              maturities: np.ndarray = None,
                              n_strikes: int = 15, noise_bps: float = 25.0,
                              seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if maturities is None:
        maturities = np.array([1 / 12, 3 / 12, 6 / 12, 1.0, 2.0])  # 1M, 3M, 6M, 1Y, 2Y

    def true_params(T: float) -> SVIParams:
        # Paramètres SVI "vrais" dépendant de la maturité : niveau et
        # convexité augmentent avec T (vol plus incertaine à long terme),
        # skew négatif constant (typique actions : krach > rally dans les prix des options).
        return SVIParams(a=0.015 + 0.02 * T, b=0.12 + 0.05 * T, rho=-0.45,
                          m=0.0, sigma=0.10 + 0.05 * np.sqrt(T))

    rows = []
    for T in maturities:
        forward = S0 * np.exp((r - q) * T)
        k = np.linspace(-0.4, 0.4, n_strikes)   # log-moneyness log(K/F)
        strikes = forward * np.exp(k)

        true_iv = svi_implied_vol(k, T, true_params(T))

        # Convention marché : on cote des options hors-la-monnaie
        # (puts pour K<F, calls pour K>F), plus liquides en pratique.
        option_types = np.where(strikes < forward, "put", "call")

        clean_prices = np.array([
            bs_price(S0, K, T, r, sigma, q, opt)
            for K, sigma, opt in zip(strikes, true_iv, option_types)
        ])

        # Bruit de cotation multiplicatif (spread bid/ask simulé)
        noise = 1.0 + rng.normal(0.0, noise_bps * 1e-4, size=len(clean_prices))
        noisy_prices = clean_prices * noise

        for K, price, opt in zip(strikes, noisy_prices, option_types):
            rows.append({"T": T, "K": K, "price": price, "option_type": opt,
                         "S": S0, "r": r, "q": q})

    return pd.DataFrame(rows)

"""Ajoute une colonne 'iv' au DataFrame en inversant Black-Scholes maturité par maturité."""
def compute_implied_vols(chain: pd.DataFrame) -> pd.DataFrame:
    chain = chain.copy()
    chain["iv"] = np.nan
    for T, group in chain.groupby("T"):
        ivs = implied_vol_batch(group["price"].to_numpy(), group["S"].iloc[0],
                                 group["K"].to_numpy(), T, group["r"].iloc[0],
                                 group["q"].iloc[0], group["option_type"].to_numpy())
        chain.loc[group.index, "iv"] = ivs
    return chain.dropna(subset=["iv"])
