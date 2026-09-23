"""
hull_white.py
=============
Modèle de taux court Hull-White à un facteur (mean-reverting Gaussien),
utilisé pour simuler l'évolution future de la courbe des taux, 
et donc la valeur de marché (MtM) future d'un swap de taux, 
c'est le moteur de simulation d'exposition dont a besoin tout calcul de CVA/xVA.

dr_t = (theta(t) - a*r_t) dt + sigma dW_t

Courbe initiale supposée PLATE (f(0,t) = r0 constant) pour garder le
calibrage de theta(t) explicite et lisible ; 
le code est structuré pour qu'il soit trivial de remplacer f(0,t),
par une vraie courbe de marché (theta(t) en dépend directement).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class HullWhite:
    a: float       # vitesse de retour à la moyenne
    sigma: float    # volatilité du taux court
    r0: float       # taux court initial (= niveau de la courbe plate)

    """
    Fonction de dérive déterministe calibrée pour que le modèle
    reproduise exactement la courbe initiale f(0,t) = r0 (plate).
    Formule standard (Brigo & Mercurio) : avec df(0,t)/dt = 0 ici,
    theta(t) = a*r0 + sigma^2/(2a) * (1 - exp(-2a t))
    """

    def theta(self, t: np.ndarray) -> np.ndarray:
        return self.a * self.r0 + (self.sigma ** 2) / (2 * self.a) * (1 - np.exp(-2 * self.a * t))

    """
Simule le taux court par Euler-Maruyama sur une grille fine
(substeps_per_period pas entre deux dates de paiement, pour
limiter l'erreur de discrétisation), mais ne renvoie l'état
qu'AUX dates de paiement du swap, c'est là qu'on évalue le
MtM, donc inutile de tout stocker.

Retourne :
r_at_payments : (n_paths, n_payments+1) taux court aux dates de paiement (colonne 0 = t=0)
money_market : (n_paths, n_payments+1) numéraire "compte monétaire" B_t = exp(int_0^t r_s ds), 
utilisé UNIQUEMENT
    pour la vérification de martingale (voir main.py), 
    pas pour l'actualisation de marché du CVA (qui utilise la courbe déterministe P(0,t), convention standard).

Variance réduite par variates antithétiques (même technique
que le pricing engine du projet 1).
"""
    def simulate(self, payment_times: np.ndarray, substeps_per_period: int,
                 n_paths: int, seed: int = 42):
        rng = np.random.default_rng(seed)

        # --- Grille fine de simulation ---
        fine_times = [0.0]
        for i in range(len(payment_times)):
            start = payment_times[i - 1] if i > 0 else 0.0
            end = payment_times[i]
            sub = np.linspace(start, end, substeps_per_period + 1)[1:]
            fine_times.extend(sub.tolist())
        fine_times = np.array(fine_times)
        n_fine = len(fine_times)

        half = n_paths // 2
        r = np.full((n_paths, n_fine), self.r0, dtype=float)
        log_B = np.zeros((n_paths, n_fine), dtype=float)   # log du numéraire, cumulé par trapèzes

        for i in range(1, n_fine):
            dt = fine_times[i] - fine_times[i - 1]
            t_prev = fine_times[i - 1]

            z = rng.standard_normal(half)
            z_full = np.concatenate([z, -z])   # antithétique : réduit la variance à coût nul

            drift = (self.theta(np.array([t_prev])) - self.a * r[:, i - 1]) * dt
            diffusion = self.sigma * np.sqrt(dt) * z_full
            r[:, i] = r[:, i - 1] + drift + diffusion

            # Intégrale du taux court par la méthode des trapèzes,
            # pour un numéraire beaucoup plus précis qu'une simple
            # somme de Riemann à gauche.
            log_B[:, i] = log_B[:, i - 1] + 0.5 * (r[:, i - 1] + r[:, i]) * dt

        # --- Extraction aux dates de paiement uniquement ---
        payment_idx = [0] + [int(np.argmin(np.abs(fine_times - t))) for t in payment_times]
        r_at_payments = r[:, payment_idx]
        money_market = np.exp(log_B[:, payment_idx])

        return r_at_payments, money_market

    """
    Prix zéro-coupon analytique P(t,T) sous Hull-White, connaissant
    r_t (vectorisé sur les trajectoires). Formule fermée standard
    (Brigo & Mercurio, eq. 3.39), spécialisée pour une courbe
    initiale plate P(0,s) = exp(-r0*s).
    """
    def zero_coupon_price(self, t: float, T: np.ndarray, r_t: np.ndarray) -> np.ndarray:

        T = np.asarray(T, dtype=float)
        B_tT = (1 - np.exp(-self.a * (T - t))) / self.a

        P0T = np.exp(-self.r0 * T)
        P0t = np.exp(-self.r0 * t) if t > 0 else 1.0

        A_tT = (P0T / P0t) * np.exp(
            B_tT * self.r0 - (self.sigma ** 2 / (4 * self.a)) * (1 - np.exp(-2 * self.a * t)) * B_tT ** 2
        )

        # r_t : (n_paths,) ; B_tT, A_tT : (n_maturities,) -> broadcast en (n_paths, n_maturities)
        return A_tT[None, :] * np.exp(-B_tT[None, :] * r_t[:, None])
