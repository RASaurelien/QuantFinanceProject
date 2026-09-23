"""
pricer.py
=========
Pricer Monte Carlo pour une note autocall à mémoire (structure type
"Athena" / "Phoenix"), sur un sous-jacent unique. C'est le produit
structuré le plus vendu en banque de détail/privée en Europe? un
bon exemple de ce que "structuration" veut dire en pratique : combiner
plusieurs briques optionnelles simples (barrières digitales, mémoire de
coupon, capital à risque conditionnel) en un seul produit dont le
payoff dépend du CHEMIN suivi par le sous-jacent, pas seulement de sa
valeur finale.

Mécanique du produit (voir README pour le détail complet) :
- À chaque date d'observation, 
si le sous-jacent >= barrière de coupon, 
un coupon est versé 
(avec MÉMOIRE : tout coupon manqué précédemment est rattrapé).
- Si le sous-jacent >= barrière de rappel (autocall),
à une date d'observation autre que la dernière, 
la note est remboursée par anticipation (capital + coupon dû).
- À l'échéance, si jamais rappelée : 
capital protégé si le sous-jacent >= barrière de protection, 
sinon perte proportionnelle à la baisse du sous-jacent 
(capital à risque "en dessous de la barrière").
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class AutocallNote:
    notional: float = 1000.0
    S0: float = 100.0
    obs_times: list = field(default_factory=lambda: [1.0, 2.0, 3.0])   # dates d'observation (années)
    autocall_barrier: float = 1.00     # en fraction de S0 ; s'applique a toutes les dates SAUF la derniere
    coupon_barrier: float = 0.70        # en fraction de S0 ; condition de versement du coupon (avec memoire)
    capital_barrier: float = 0.70        # en fraction de S0 ; protection du capital a l'echeance
    coupon_rate: float = 0.08             # taux de coupon ANNUEL (proportionnel a la periode entre dates)

    @property
    def maturity(self) -> float:
        return self.obs_times[-1]

"""
Simule le sous-jacent (GBM risque-neutre)
exactement aux dates d'observation, 
inutile de simuler une trajectoire continue,
la loi de S(t) est connue en forme fermée,
a chaque instant sous GBM, 
ce qui rend la simulation O(n_paths x n_obs), 
plutot que O(n_paths x n_steps_fins).
Variates antithetiques (meme technique que le projet 1), 
pour reduire la variance a cout nul.

`spot` : niveau de depart de la simulation. Distinct de `note.S0`
(le niveau de REFERENCE fige a l'emission, qui definit les barrieres en absolu),
crucial pour le calcul du delta : 
bouger le spot courant sans bouger les barrieres, 
est la seule facon economiquement correcte de mesurer une sensibilite de couverture
(voir compute_delta).
"""
def simulate_underlying_at_obs_dates(note: AutocallNote, r: float, q: float, sigma: float,
                                       n_paths: int, seed: int = 42, spot: float = None) -> np.ndarray:
    if spot is None:
        spot = note.S0

    rng = np.random.default_rng(seed)
    obs_times = np.array(note.obs_times)
    dt = np.diff(np.concatenate([[0.0], obs_times]))   # duree entre dates d'observation successives

    half = n_paths // 2
    n_obs = len(obs_times)

    z = rng.standard_normal((half, n_obs))
    z_full = np.concatenate([z, -z], axis=0)   # antithetique

    log_increments = (r - q - 0.5 * sigma ** 2) * dt[None, :] + sigma * np.sqrt(dt)[None, :] * z_full
    log_S = np.log(spot) + np.cumsum(log_increments, axis=1)

    return np.exp(log_S)   # (n_paths, n_obs)

"""
Valorise la note par Monte Carlo : 
pour chaque trajectoire, applique la mécanique complète 
(mémoire de coupon, autocall, protection du capital), 
actualise les flux à leur date de versement, 
moyenne sur toutes les trajectoires.

Les BARRIÈRES restent toujours définies en % de `note.S0` 
(le niveau de référence fixé à l'émission), 
même si `spot` (le niveau de départ de la simulation) diffère, 
c'est la mécanique réelle d'un autocall déjà émis dont le sous-jacent a bougé depuis
l'émission.
"""
def price_autocall(note: AutocallNote, r: float, q: float, sigma: float,
                     n_paths: int = 200_000, seed: int = 42, spot: float = None) -> dict:
    S_paths = simulate_underlying_at_obs_dates(note, r, q, sigma, n_paths, seed, spot=spot)
    n_paths_actual, n_obs = S_paths.shape
    obs_times = np.array(note.obs_times)
    period = np.diff(np.concatenate([[0.0], obs_times]))

    discounted_cashflows = np.zeros(n_paths_actual)
    redemption_time = np.full(n_paths_actual, note.maturity)   # date effective de remboursement (autocall ou maturite)
    is_autocalled = np.zeros(n_paths_actual, dtype=bool)
    final_below_capital_barrier = np.zeros(n_paths_actual, dtype=bool)

    still_alive = np.ones(n_paths_actual, dtype=bool)
    unpaid_coupon = np.zeros(n_paths_actual)

    for i in range(n_obs):
        t = obs_times[i]
        S_t = S_paths[:, i]
        is_last = (i == n_obs - 1)

        coupon_due = still_alive & (S_t >= note.coupon_barrier * note.S0)
        coupon_paid_amount = note.coupon_rate * period[i] * note.notional + unpaid_coupon

        # Coupon verse (avec rattrapage memoire) la ou la condition est remplie
        discounted_cashflows += np.where(coupon_due, coupon_paid_amount * np.exp(-r * t), 0.0)
        unpaid_coupon = np.where(still_alive & ~coupon_due,
                                    unpaid_coupon + note.coupon_rate * period[i] * note.notional,
                                    np.where(coupon_due, 0.0, unpaid_coupon))

        if not is_last:
            autocalled_now = still_alive & (S_t >= note.autocall_barrier * note.S0)
            discounted_cashflows += np.where(autocalled_now, note.notional * np.exp(-r * t), 0.0)
            redemption_time = np.where(autocalled_now, t, redemption_time)
            is_autocalled = is_autocalled | autocalled_now
            still_alive = still_alive & ~autocalled_now
        else:
            # Derniere date : remboursement final pour tout ce qui est encore vivant
            below_barrier = still_alive & (S_t < note.capital_barrier * note.S0)
            above_or_equal = still_alive & (S_t >= note.capital_barrier * note.S0)

            redemption_above = note.notional
            redemption_below = note.notional * (S_t / note.S0)   # capital a risque, proportionnel a la baisse

            discounted_cashflows += np.where(above_or_equal, redemption_above * np.exp(-r * t), 0.0)
            discounted_cashflows += np.where(below_barrier, redemption_below * np.exp(-r * t), 0.0)

            final_below_capital_barrier = below_barrier

    price = discounted_cashflows.mean()
    stderr = discounted_cashflows.std(ddof=1) / np.sqrt(n_paths_actual)

    autocall_prob_by_date = {}
    for i, t in enumerate(obs_times[:-1]):
        autocall_prob_by_date[t] = float(np.mean(redemption_time == t))
    autocall_prob_by_date["maturité (jamais rappelée)"] = float(np.mean(redemption_time == note.maturity))

    return {
        "price": float(price),
        "stderr": float(stderr),
        "price_pct_notional": float(price / note.notional * 100),
        "prob_autocall_total": float(is_autocalled.mean()),
        "prob_capital_loss": float(final_below_capital_barrier.mean()),
        "autocall_prob_by_date": autocall_prob_by_date,
        "expected_life": float(redemption_time.mean()),
        "redemption_times": redemption_time,
        "final_underlying": S_paths[:, -1],
    }

"""
Delta par différences finies centrées avec Common Random Numbers
(même seed pour spot bumpé et non bumpé, 
voir projet 1 pour la justification détaillée de la technique).

IMPORTANT : on bump le SPOT courant, jamais `note.S0` 
(qui fixe les barrières en absolu, définies une fois pour toutes à l'émission).
Bumper S0 par erreur ferait bouger les barrières EN MÊME TEMPS que le sous-jacent, 
le prix redeviendrait invariant par construction
(homogénéité de degré 0 en S0), 
et le delta mesuré serait artificiellement nul, 
ce qui n'a aucun sens économique pour une desk qui doit couvrir une position déjà émise.
"""
def compute_delta(note: AutocallNote, r: float, q: float, sigma: float,
                    n_paths: int = 200_000, seed: int = 42, bump: float = 0.01) -> float:
    price_up = price_autocall(note, r, q, sigma, n_paths, seed, spot=note.S0 * (1 + bump))["price"]
    price_down = price_autocall(note, r, q, sigma, n_paths, seed, spot=note.S0 * (1 - bump))["price"]

    return (price_up - price_down) / (2 * note.S0 * bump)
