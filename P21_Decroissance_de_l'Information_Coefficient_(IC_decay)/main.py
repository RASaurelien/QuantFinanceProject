"""
main.py
=======
Etude de decroissance de l'Information Coefficient (IC)  d'un signal predictif 
-- un question qu'un desk de recherche pose AVANT de deployer un signal : 
combien de temps l'edge dure-t-il, et se degrade-t-il progressivement ou disparait-il d'un coup ?

IC(h) = correlation de rang (Spearman) entre le signal au temps t, 
et le rendement futur sur l'horizon h : 
IC(h) = corr_rang(S_t, R_{t,t+h})

Prolongement du signal de regime de volatilite (Tier 2 #8) : 
la meme question, "le signal detecte-t-il l'appartenance a un regime, 
ou anticipe-t-il les BASCULES ?" 
- se reformule ici comme une courbe de decroissance de l'IC en fonction de l'horizon.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
Simule un signal S_t et une matrice de rendements futurs 
R_{t,h}(h=1..max_horizon) 
avec une correlation VRAIE qui decroit en exp(-h/half_life)*true_ic0 
- construit directement via un modele lineaire gaussien, 
pour que la correlation cible soit exacte par construction, 
ce qui permet de valider l'estimateur d'IC empirique contre une verite connue.
"""
def simulate_signal_and_returns(n_obs: int, true_ic0: float, half_life: float,
                                  max_horizon: int, seed: int):
    rng = np.random.default_rng(seed)
    S = rng.standard_normal(n_obs)

    R = np.zeros((n_obs, max_horizon))
    for h in range(1, max_horizon + 1):
        rho_h = true_ic0 * np.exp(-h / half_life)
        noise = rng.standard_normal(n_obs)
        R[:, h - 1] = rho_h * S + np.sqrt(max(1 - rho_h ** 2, 0)) * noise

    return S, R

"""IC(h) empirique (Spearman) pour chaque horizon h (colonne de R)."""
def empirical_ic(S: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.array([stats.spearmanr(S, R[:, h]).correlation for h in range(R.shape[1])])


def exp_decay(h, ic0, half_life):
    return ic0 * np.exp(-h / half_life)


def main():
    import os
    os.makedirs("../P21_Decroissance_de_l'Information_Coefficient_(IC_decay)/outputs", exist_ok=True)

    print("=" * 60)
    print(" Décroissance de l'Information Coefficient (IC decay)")
    print("=" * 60 + "\n")

    N_OBS, MAX_H = 3000, 60
    TRUE_IC0, TRUE_HALF_LIFE = 0.12, 12.0

    S, R = simulate_signal_and_returns(N_OBS, TRUE_IC0, TRUE_HALF_LIFE, MAX_H, seed=42)
    ic_signal = empirical_ic(S, R)

    S_noise, R_noise = simulate_signal_and_returns(N_OBS, 0.0, TRUE_HALF_LIFE, MAX_H, seed=7)
    ic_noise = empirical_ic(S_noise, R_noise)

    horizons = np.arange(1, MAX_H + 1)

    popt, _ = curve_fit(exp_decay, horizons, ic_signal, p0=[0.1, 10.0])
    ic0_hat, half_life_hat = popt

    print(f"Signal réel : IC vrai à h=1 = {TRUE_IC0:.3f}, demi-vie vraie = {TRUE_HALF_LIFE:.1f} jours")
    print(f"Ajustement empirique : IC0 estimé = {ic0_hat:.3f}, demi-vie estimée = {half_life_hat:.2f} jours")
    print(f"Écart sur la demi-vie : {abs(half_life_hat - TRUE_HALF_LIFE):.2f} jours "
          f"({abs(half_life_hat-TRUE_HALF_LIFE)/TRUE_HALF_LIFE*100:.1f}%)\n")

    print(f"Signal bruit (contrôle négatif) : IC moyen sur tous horizons = {ic_noise.mean():+.4f} "
          f"(écart-type = {ic_noise.std():.4f}) -> {'proche de 0, OK' if abs(ic_noise.mean()) < 0.02 else 'ATTENTION'}\n")

    print(f"Implication pratique : à horizon = demi-vie ({half_life_hat:.0f}j), l'IC est retombé à "
          f"{ic0_hat/2:.3f} (50% du signal initial) -- fenêtre de rebalancement à considérer avant "
          f"que le signal ne se dilue davantage.")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(horizons, ic_signal, "o", color="steelblue", ms=4, alpha=0.7, label="IC empirique (signal réel)")
    ax.plot(horizons, exp_decay(horizons, *popt), color="firebrick", lw=2,
             label=f"Ajustement exponentiel (demi-vie={half_life_hat:.1f}j)")
    ax.plot(horizons, TRUE_IC0 * np.exp(-horizons / TRUE_HALF_LIFE), color="black", lw=1.3, ls="--",
             label=f"Vraie décroissance (demi-vie={TRUE_HALF_LIFE:.0f}j)")
    ax.plot(horizons, ic_noise, "o", color="gray", ms=3, alpha=0.5, label="IC empirique (bruit, contrôle négatif)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Horizon de prévision h (jours)")
    ax.set_ylabel("Information Coefficient")
    ax.set_title("Décroissance de l'IC en fonction de l'horizon de prévision")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("../P21_Decroissance_de_l'Information_Coefficient_(IC_decay)/outputs/ic_decay.png", dpi=140)
    plt.close(fig)
    print("\nGraphique exporté -> ../P21_Decroissance_de_l'Information_Coefficient_(IC_decay)/outputs/ic_decay.png")


if __name__ == "__main__":
    main()
