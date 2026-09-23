"""
main.py
=======
Un trader vend une option, delta-hedge quotidiennement 
(position en sous-jacent ajustee chaque jour pour annuler le delta), 
et a la fin on decompose le P&L realise en contributions 
Delta, Gamma, Theta, Vega,
un trader option peut se demander : 
"pourquoi j'ai gagne/perdu de l'argent cette semaine, 
malgre un delta-hedge parfait ?"

Reutilise les formules Black-Scholes du pricing engine,
reecrites ici de facon compacte pour rester autonome.

Point cle : 
la vol UTILISEE pour le hedge (vol implicite au moment de la vente) 
peut differer de la vol REALISEE du sous-jacent, 
et c'est exactement cet ecart qui determine si le vendeur d'option gagne, 
ou perd de l'argent sur son gamma, 
independamment de la direction du marche 
(le hedge delta neutralise la direction, pas le niveau de volatilite).
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bs_price_greeks(S, K, T, r, sigma):
    if T <= 1e-8:
        price = max(S - K, 0.0)
        return dict(price=price, delta=float(S > K), gamma=0.0, theta=0.0, vega=0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return dict(price=price, delta=delta, gamma=gamma, theta=theta, vega=vega)

"""
Vend 1 call, delta-hedge quotidiennement. Retourne le P&L réel
(repricing complet) et sa décomposition Taylor (Delta/Gamma/Theta/Vega)
jour par jour.
"""
def simulate_hedged_position(S0, K, T_days, r, implied_vol, realized_vol, seed):
    rng = np.random.default_rng(seed)
    dt = 1 / 365
    n_days = T_days

    S = np.zeros(n_days + 1)
    S[0] = S0
    for t in range(n_days):
        S[t + 1] = S[t] * np.exp((r - 0.5 * realized_vol ** 2) * dt + realized_vol * np.sqrt(dt) * rng.standard_normal())

    T_remaining = T_days / 365

    real_pnl = np.zeros(n_days)
    taylor_components = np.zeros((n_days, 4))   # Delta(+hedge), Gamma, Theta, Vega

    g0 = bs_price_greeks(S[0], K, T_remaining, r, implied_vol)
    stock_position = g0["delta"]

    prev_greeks = g0
    for t in range(n_days):
        T_remaining_next = max((T_days - t - 1) / 365, 0)
        g_next = bs_price_greeks(S[t + 1], K, T_remaining_next, r, implied_vol)

        # --- P&L RÉEL du jour : variation MtM de (position actions + option courte) ---
        option_value_change = g_next["price"] - prev_greeks["price"]
        stock_pnl = stock_position * (S[t + 1] - S[t])
        real_pnl[t] = stock_pnl - option_value_change   # short l'option -> -variation de sa valeur

        # --- Décomposition Taylor, expliquée par les Greeks du DÉBUT du jour ---
        dS = S[t + 1] - S[t]
        delta_pnl = -prev_greeks["delta"] * dS           # côté position courte de l'option
        gamma_pnl = -0.5 * prev_greeks["gamma"] * dS ** 2
        theta_pnl = -prev_greeks["theta"]                 # le temps qui passe profite au VENDEUR
        vega_pnl = 0.0                                     # vol implicite constante dans ce scénario
        hedge_pnl = stock_position * dS                     # le hedge en actions compense le delta

        taylor_components[t] = [delta_pnl + hedge_pnl, gamma_pnl, theta_pnl, vega_pnl]

        stock_position = g_next["delta"]   # re-hedge : ajuste la position en actions
        prev_greeks = g_next

    return S, real_pnl, taylor_components


def main():
    import os
    os.makedirs("../P24_Attribution_P&L_market_maker_options/outputs", exist_ok=True)

    print("=" * 60)
    print(" Attribution P&L, vendeur d'option delta-hedgé")
    print("=" * 60 + "\n")

    S0, K, T_DAYS, R, IMPLIED_VOL = 100.0, 100.0, 30, 0.03, 0.20

    scenarios = {
        "Vol réalisée < vol implicite (le vendeur gagne)": 0.12,
        "Vol réalisée = vol implicite (référence)": 0.20,
        "Vol réalisée > vol implicite (le vendeur perd)": 0.32,
    }

    print(f"Option : call ATM, K={K}, maturité={T_DAYS}j, vol implicite (utilisée pour le hedge)={IMPLIED_VOL:.0%}\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (label, realized_vol) in enumerate(scenarios.items()):
        S, real_pnl, taylor = simulate_hedged_position(S0, K, T_DAYS, R, IMPLIED_VOL, realized_vol, seed=1)

        total_real = real_pnl.sum()
        total_taylor = taylor.sum(axis=0)
        reconstruction_error = total_real - total_taylor.sum()

        print(f"--- {label} (vol réalisée={realized_vol:.0%}) ---")
        print(f"  P&L réel total (repricing complet)      : {total_real:+.4f}")
        print(f"  Somme des contributions Taylor           : {total_taylor.sum():+.4f}")
        pct = abs(reconstruction_error) / abs(total_real) * 100 if total_real != 0 else 0
        print(f"  Erreur de reconstruction (résiduel)       : {reconstruction_error:+.4f} ({pct:.2f}% du P&L réel)")
        print(f"  Décomposition : Delta(+hedge résiduel)={total_taylor[0]:+.4f}  "
              f"Gamma={total_taylor[1]:+.4f}  Theta={total_taylor[2]:+.4f}  Vega={total_taylor[3]:+.4f}\n")

        cum_real = np.cumsum(real_pnl)
        cum_gamma = np.cumsum(taylor[:, 1])
        cum_theta = np.cumsum(taylor[:, 2])

        ax = axes[idx]
        ax.plot(cum_real, color="black", lw=2, label="P&L réel cumulé")
        ax.plot(cum_gamma, color="firebrick", lw=1.3, ls="--", label="Contribution Gamma cumulée")
        ax.plot(cum_theta, color="seagreen", lw=1.3, ls="--", label="Contribution Theta cumulée")
        ax.plot(cum_gamma + cum_theta, color="navy", lw=1, ls=":", label="Gamma+Theta")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(label.split(" (")[0], fontsize=9)
        ax.set_xlabel("Jour")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Attribution P&L d'un vendeur d'option delta-hedgé : Gamma vs Theta", fontsize=13)
    fig.tight_layout()
    fig.savefig("../P24_Attribution_P&L_market_maker_options/outputs/pnl_attribution.png", dpi=140)
    plt.close(fig)
    print("Graphique exporté -> ../P24_Attribution_P&L_market_maker_options/outputs/pnl_attribution.png")


if __name__ == "__main__":
    main()