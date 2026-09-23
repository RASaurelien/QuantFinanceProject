"""
main.py
=======
Agent de trading par Q-learning tabulaire, 
sur un marché synthétique à changement de régime 
(même famille de générateur que le moteur debacktesting événementtiel).

Le trading par RL est un terrain très saturé, 
avec un risque élevé de surapprentissage et peu de garantie d'edge réel 
-- ce projet est traité avec la même honnêteté méthodologique que le reste du dépôt : 
train et test sur des simulations INDÉPENDANTES (seeds différentes), 
comparaison systématique à des baselines simples, 
et pas de survente du résultat.

État : (bucket de momentum récent, position courante)
Action : {-1 (short), 0 (flat), +1 (long)}
Récompense : rendement de la position précédente - coût de transaction
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate_prices(n_days: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prices = np.zeros(n_days)
    price = 100.0
    anchor = price
    regime = 0
    days_in_regime = 0

    for t in range(n_days):
        days_in_regime += 1
        if days_in_regime > 20 and rng.random() < 0.03:
            regime = rng.integers(0, 3)
            anchor = price
            days_in_regime = 0

        z = rng.standard_normal()
        if regime == 0:
            ret = 0.0009 + 0.011 * z
        elif regime == 1:
            ret = -0.0009 + 0.011 * z
        else:
            ret = 0.05 * (np.log(anchor) - np.log(price)) + 0.009 * z

        price *= max(1 + ret, 0.01)
        prices[t] = price

    return prices


def momentum_bucket(returns_window: np.ndarray) -> int:
    """Discrétise le momentum récent (rendement cumulé sur la fenêtre) en 5 buckets, via des bornes FIXES (pas ajustées sur les données -- évite une fuite d'information)."""
    momentum = returns_window.sum()
    edges = np.array([-0.03, -0.01, 0.01, 0.03])
    return int(np.searchsorted(edges, momentum))


class QLearningAgent:
    def __init__(self, n_momentum_buckets: int = 5, n_positions: int = 3,
                 alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.15, seed: int = 0):
        self.actions = [-1, 0, 1]
        self.Q = np.zeros((n_momentum_buckets, n_positions, len(self.actions)))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.rng = np.random.default_rng(seed)

    def _pos_idx(self, position: int) -> int:
        return position + 1

    def select_action(self, state: tuple[int, int], greedy: bool = False) -> int:
        m, pos = state
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.actions[int(np.argmax(self.Q[m, self._pos_idx(pos)]))]

    def update(self, state, action, reward, next_state):
        m, pos = state
        m2, pos2 = next_state
        a_idx = self.actions.index(action)
        best_next = np.max(self.Q[m2, self._pos_idx(pos2)])
        td_target = reward + self.gamma * best_next
        self.Q[m, self._pos_idx(pos), a_idx] += self.alpha * (td_target - self.Q[m, self._pos_idx(pos), a_idx])


def run_episode(agent: QLearningAgent, prices: np.ndarray, window: int = 5,
                 cost_bps: float = 2.0, train: bool = True, greedy: bool = False):
    returns = np.diff(np.log(prices))
    n = len(returns)
    position = 0
    equity = [1.0]
    positions_log = []

    for t in range(window, n):
        m = momentum_bucket(returns[t - window:t])
        state = (m, position)
        action = agent.select_action(state, greedy=greedy)

        reward = position * returns[t] - cost_bps * 1e-4 * abs(action - position)
        equity.append(equity[-1] * (1 + reward))
        positions_log.append(action)

        if train and t + 1 < n:
            m2 = momentum_bucket(returns[t - window + 1:t + 1])
            next_state = (m2, action)
            agent.update(state, action, reward, next_state)

        position = action

    return np.array(equity), np.array(positions_log)


def sharpe(equity: np.ndarray) -> float:
    r = np.diff(equity) / equity[:-1]
    return 0.0 if r.std() < 1e-12 else r.mean() / r.std() * np.sqrt(252)


def main():
    import os
    os.makedirs("../P18_Agent_de_trading_Qlearning/outputs", exist_ok=True)

    print("=" * 60)
    print(" Agent de trading Q-learning -- entraînement et évaluation honnête")
    print("=" * 60 + "\n")

    train_prices = simulate_prices(4000, seed=1)
    test_prices = simulate_prices(1500, seed=999)   # seed DIFFÉRENTE : vraie généralisation out-of-sample

    agent = QLearningAgent(seed=0)

    print("Entraînement (20 passes sur la série d'entraînement, epsilon-greedy)...")
    for epoch in range(20):
        run_episode(agent, train_prices, train=True, greedy=False)
    print("Terminé.\n")

    equity_agent, positions = run_episode(agent, test_prices, train=False, greedy=True)

    buy_hold = test_prices[5:] / test_prices[5]

    rng_random = np.random.default_rng(42)
    random_actions = rng_random.choice([-1, 0, 1], size=len(positions))
    returns_test = np.diff(np.log(test_prices))
    random_equity = [1.0]
    pos = 0
    for t, a in enumerate(random_actions):
        r = pos * returns_test[5 + t] - 2e-4 * abs(a - pos)
        random_equity.append(random_equity[-1] * (1 + r))
        pos = a
    random_equity = np.array(random_equity)

    print("--- Performance out-of-sample (test, seed différente de l'entraînement) ---")
    print(f"Agent Q-learning : rendement total = {(equity_agent[-1]-1)*100:+.2f}%  Sharpe = {sharpe(equity_agent):.3f}")
    print(f"Buy & Hold        : rendement total = {(buy_hold[-1]-1)*100:+.2f}%  Sharpe = {sharpe(buy_hold):.3f}")
    print(f"Politique aléatoire : rendement total = {(random_equity[-1]-1)*100:+.2f}%  Sharpe = {sharpe(random_equity):.3f}\n")

    print("Répartition des actions prises par l'agent (test, politique greedy) :")
    for a, label in zip([-1, 0, 1], ["Short", "Flat", "Long"]):
        print(f"  {label:6s} : {np.mean(positions == a)*100:.1f}%")

    if sharpe(equity_agent) <= max(sharpe(buy_hold), sharpe(random_equity)):
        print("\nRemarque honnête : l'agent NE bat PAS les baselines sur ce test out-of-sample.")
        print("C'est un résultat plausible pour du RL tabulaire sur un état aussi pauvre (5 buckets")
        print("de momentum) -- voir la section Limites du README plutôt qu'une conclusion en faveur")
        print("du RL qui ne serait pas soutenue par ce test.")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(equity_agent, color="navy", lw=1.4, label="Agent Q-learning")
    ax.plot(buy_hold, color="gray", lw=1.2, ls="--", label="Buy & Hold")
    ax.plot(random_equity, color="firebrick", lw=1.0, alpha=0.7, label="Politique aléatoire")
    ax.set_xlabel("Jour (test, out-of-sample)")
    ax.set_ylabel("Équité (base 1.0)")
    ax.set_title("Q-learning vs Buy & Hold vs Aléatoire (out-of-sample)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("../P18_Agent_de_trading_Qlearning/outputs/rl_agent_equity.png", dpi=140)
    plt.close(fig)
    print("\nGraphique exporté -> ../P18_Agent_de_trading_Qlearning/outputs/rl_agent_equity.png")


if __name__ == "__main__":
    main()
