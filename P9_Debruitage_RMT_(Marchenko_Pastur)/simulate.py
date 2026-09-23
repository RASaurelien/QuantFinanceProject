"""
simulate.py
===========
Simule des rendements multi-actifs sous un modèle à facteurs latents,
avec la matrice de covariance VRAIE connue exactement,
ce qui permet de valider objectivement le débruitage RMT plus loin 
(sur des données de marché réelles, on n'a jamais accès à la "vraie" covariance, 
donc on ne peut jamais savoir directement, 
si un débruitage a amélioré les choses ; ici, si).

Modèle : R_t = B F_t + eps_t
- B (N x K)   : sensibilités factorielles (loadings)
- F_t ~ N(0, Sigma_f)  : rendements des K facteurs latents
- eps_t ~ N(0, D)       : bruit idiosyncratique par actif (D diagonale)

Sigma_vraie = B Sigma_f B' + D
"""

from __future__ import annotations
import numpy as np

"""
Retourne (returns, Sigma_true) où returns est un tableau
(n_periods, n_assets) et Sigma_true la matrice de covariance
exacte utilisée pour générer les données.
"""
def simulate_factor_model(n_assets: int = 200, n_periods: int = 500, n_factors: int = 5,
                            seed: int = 42):
    rng = np.random.default_rng(seed)

    # Loadings factoriels : chaque actif a une exposition positive au
    # "facteur marché" (facteur 0) et des expositions aléatoires aux
    # autres facteurs (secteur/style), pour une structure réaliste.
    B = np.zeros((n_assets, n_factors))
    B[:, 0] = rng.uniform(0.5, 1.5, size=n_assets)          # facteur marché : tout le monde y est exposé positivement
    B[:, 1:] = rng.normal(0, 0.6, size=(n_assets, n_factors - 1))   # facteurs sectoriels/style

    # Volatilités factorielles décroissantes (le facteur marché domine, comme en pratique)
    factor_vols = np.array([0.16, 0.10, 0.08, 0.07, 0.06])[:n_factors]
    Sigma_f = np.diag(factor_vols ** 2)

    # Volatilité idiosyncratique par actif (hétérogène, réaliste)
    idio_vols = rng.uniform(0.15, 0.35, size=n_assets)
    D = np.diag(idio_vols ** 2)

    Sigma_true = B @ Sigma_f @ B.T + D

    # Simulation des rendements (périodes iid, cohérent avec l'hypothèse
    # sous-jacente du test de Marchenko-Pastur)
    F = rng.multivariate_normal(np.zeros(n_factors), Sigma_f, size=n_periods)
    eps = rng.normal(0, 1, size=(n_periods, n_assets)) * idio_vols[None, :]
    returns = F @ B.T + eps

    return returns, Sigma_true
