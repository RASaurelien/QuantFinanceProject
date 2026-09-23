"""
swap.py
=======
Valorisation d'un swap de taux vanille (payer swap : on paie fixe, on
reçoit flottant), à partir des prix zéro-coupon Hull-White. La valeur
au marché à une date de paiement t_k s'exprime uniquement avec des prix
zéro-coupon, pas besoin de simuler la jambe flottante explicitement,
c'est la propriété qui rend ce pricing rapide et vectorisable :

jambe flottante(t_k) = 1 - P(t_k, T_end)     [reprise à chaque reset]
jambe fixe(t_k)      = K * sum_{i>k} tau_i * P(t_k, T_i)
MtM(t_k)             = notional * (jambe flottante - jambe fixe)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from hull_white import HullWhite


@dataclass
class InterestRateSwap:
    notional: float
    maturity: float          # en années
    payment_freq: float       # ex: 0.5 = semestriel
    fixed_rate: float = None  # si None, calculé au taux swap "par" (MtM(0)=0)

    @property
    def payment_times(self) -> np.ndarray:
        n = round(self.maturity / self.payment_freq)
        return np.arange(1, n + 1) * self.payment_freq

    """Taux fixe qui annule le MtM en t=0 (convention standard de cotation d'un swap)."""
    def par_rate(self, hw: HullWhite) -> float:
        times = self.payment_times
        P0 = np.exp(-hw.r0 * times)   # courbe plate : P(0,T) = exp(-r0*T)
        floating_leg = 1.0 - P0[-1]
        annuity = np.sum(self.payment_freq * P0)
        return floating_leg / annuity

    """
    MtM vectorisé (une valeur par trajectoire) à la date de
    paiement t, connaissant le taux court simulé r_t sur chaque
    trajectoire à cette date.
    """
    def mtm(self, hw: HullWhite, t: float, r_t: np.ndarray) -> np.ndarray:

        future_times = self.payment_times[self.payment_times > t + 1e-9]
        if len(future_times) == 0:
            return np.zeros_like(r_t)   # swap arrivé à maturité : plus de cashflows

        P_tT = hw.zero_coupon_price(t, future_times, r_t)   # (n_paths, n_future_payments)

        floating_leg = 1.0 - P_tT[:, -1]
        fixed_leg = self.fixed_rate * self.payment_freq * P_tT.sum(axis=1)

        return self.notional * (floating_leg - fixed_leg)
