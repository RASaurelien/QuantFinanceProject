"""
simulate.py
===========
Simule une session de trading complète : 
le prix milieu suit un mouvement brownien arithmétique, 
les ordres arrivent au marché selon un processus de Poisson 
dont l'intensité décroît exponentiellement avec la distance à laquelle le MM, 
cote (lambda(delta) = A*exp(-k*delta)
- modèle standard d'Avellaneda-Stoikov). 
Le MM est exécuté au bid ou à l'ask dès qu'un ordre "touche" sa cotation, 
avec une probabilité par pas de temps donnée par cette intensité.
"""

from __future__ import annotations
import numpy as np

"""
quote_fn(s, q, time_left) -> (bid, ask) : 
fonction de cotation (Avellaneda-Stoikov ou naïve), 
injectée en paramètre pour comparer les deux stratégies sur
EXACTEMENT le même chemin de prix et le même flux d'ordres 
(comparaison appariée, pas juste deux runs,
indépendants qui pourraient différer par chance).
"""
def run_session(quote_fn, n_steps: int = 2000, T: float = 1.0, s0: float = 100.0,
                  sigma: float = 2.0, A: float = 140.0, k: float = 1.5, seed: int = 42):
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    mid = np.zeros(n_steps + 1)
    mid[0] = s0
    cash = 0.0
    inventory = 0
    cash_path = np.zeros(n_steps + 1)
    inventory_path = np.zeros(n_steps + 1, dtype=int)

    for t in range(n_steps):
        time_left = T - t * dt
        bid, ask = quote_fn(mid[t], inventory, time_left)

        delta_bid = max(mid[t] - bid, 1e-6)
        delta_ask = max(ask - mid[t], 1e-6)
        p_hit_bid = 1 - np.exp(-A * np.exp(-k * delta_bid) * dt)
        p_hit_ask = 1 - np.exp(-A * np.exp(-k * delta_ask) * dt)

        if rng.random() < p_hit_bid:
            inventory += 1
            cash -= bid
        if rng.random() < p_hit_ask:
            inventory -= 1
            cash += ask

        mid[t + 1] = mid[t] + sigma * np.sqrt(dt) * rng.standard_normal()
        cash_path[t + 1] = cash
        inventory_path[t + 1] = inventory

    pnl_path = cash_path + inventory_path * mid
    return mid, inventory_path, pnl_path
