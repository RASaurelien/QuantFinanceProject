"""
agent_model.py
===============
Modele de Cont-Bouchaud (2000) : un marche multi-agents ou les
rendements ne sortent PAS d'une hypothese d'efficience, mais d'un
phenomene purement structurel, la formation de clusters d'agents
qui decident ENSEMBLE (herding), sur un graphe aleatoire.

Mecanique :
1. N agents, relies deux a deux avec probabilite p (graphe d'Erdos-Renyi). 
p est calibre pres du SEUIL DE PERCOLATION (degre moyen <k> = p*(N-1) ~ 1), 
le point critique ou la distribution de la taille des clusters devient, 
une loi de puissance, exactement le mecanisme physique qui produit, 
des queues epaisses, sans aucune hypothese sur le comportement "rationnel" des agents.

2. Chaque cluster agit comme un seul trader collectif : 
achete (+1), vend (-1), ou reste inactif (0), 
avec probabilite a chacun pour +1/-1.

3. Le rendement du marche est proportionnel a la demande nette agregee : 
R_t = somme(taille_cluster * decision) / N.

Reference : Cont & Bouchaud, "Herd behavior and aggregate fluctuations
in financial markets", Macroeconomic Dynamics, 2000.
"""

from __future__ import annotations
import numpy as np

"""
Genere un graphe d'Erdos-Renyi (n_agents, p_link), 
et retourne la taille du cluster (composante connexe) de chaque agent, 
via Union-Find (Quick Union avec compression de chemin), 
O(n alpha(n)) plutot qu'un parcours de graphe naif, 
ce qui compte quand on rejoue la simulation a chaque pas de temps.
"""
def find_clusters(n_agents: int, p_link: float, rng: np.random.Generator) -> np.ndarray:
    parent = np.arange(n_agents)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:   # compression de chemin
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Nombre attendu de liens : p_link * n_agents*(n_agents-1)/2, on
    # tire directement ce nombre de paires plutot que de tester toutes
    # les C(n,2) paires (essentiel pour n_agents grand).
    expected_edges = int(p_link * n_agents * (n_agents - 1) / 2)
    ii = rng.integers(0, n_agents, size=expected_edges)
    jj = rng.integers(0, n_agents, size=expected_edges)
    for a, b in zip(ii, jj):
        if a != b:
            union(int(a), int(b))

    roots = np.array([find(i) for i in range(n_agents)])
    _, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
    return counts[inverse]   # taille du cluster de chaque agent


def simulate_step(n_agents: int, p_link: float, activation_prob: float,
                    rng: np.random.Generator) -> float:
    """Un pas de temps : forme les clusters, tire une decision par cluster, agrege la demande nette."""
    cluster_sizes = find_clusters(n_agents, p_link, rng)

    unique_sizes = np.unique(cluster_sizes)
    net_demand = 0.0
    for size in unique_sizes:
        n_clusters_this_size = int((cluster_sizes == size).sum() / size)
        decisions = rng.choice([1, -1, 0], size=n_clusters_this_size,
                                 p=[activation_prob, activation_prob, 1 - 2 * activation_prob])
        net_demand += size * decisions.sum()

    return net_demand / n_agents

"""
Simule n_steps rendements. 
Le degre moyen <k> oscille LENTEMENT autour du seuil critique 1.0 
(sinusoide de tres basse frequence + bruit), 
ce qui cree des PHASES ou le marche est plus ou moins proche de la criticalite, 
et donc des phases de queues plus ou moins epaisses, 
sans jamais l'imposer explicitement. 
C'est ce qui genere le clustering de volatilite en plus des queues epaisses,
avec le meme mecanisme structurel (proximite du seuil de percolation).
"""
def simulate_market(n_agents: int = 2000, n_steps: int = 2000, activation_prob: float = 0.10,
                     seed: int = 42, regime_amplitude: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    returns = np.zeros(n_steps)
    avg_degree_path = np.zeros(n_steps)

    for t in range(n_steps):
        avg_degree = 1.0 + regime_amplitude * np.sin(2 * np.pi * t / 400) + rng.normal(0, 0.03)
        avg_degree = max(avg_degree, 0.3)
        p_link = avg_degree / (n_agents - 1)
        avg_degree_path[t] = avg_degree

        returns[t] = simulate_step(n_agents, p_link, activation_prob, rng)

    return returns, avg_degree_path
