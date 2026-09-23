"""
rmt.py
======
Débruitage d'une matrice de corrélation empirique par la théorie des matrices aléatoires
(Random Matrix Theory), méthode de Laloux, Cizeau,
Bouchaud & Potters (1999) / Bouchaud & Potters, 
*Financial Applications of Random Matrix Theory*.

Principe : quand on estime une matrice de corrélation N x N à partir de seulement T observations, 
avec N proche de T (ratio q = N/T pas négligeable), 
une grande partie du spectre de valeurs propres est du BRUIT D'ESTIMATION PUR,
même si les vrais actifs sous-jacents n'étaient corrélés d'aucune façon, 
on observerait un spectre non-trivial juste à cause du nombre fini d'observations. 
La loi de Marchenko-Pastur donne la distribution exacte de ce spectre de bruit, 
ce qui permet de séparer signal et bruit sans aucune hypothèse, 
sur la vraie structure de corrélation.
"""

from __future__ import annotations
import numpy as np

"""
Bornes du support de la loi de Marchenko-Pastur pour, 
une matrice de corrélation N x N estimée sur T observations iid, q = N/T.
Toute valeur propre dans [lambda_minus, lambda_plus] est compatible,
avec du bruit pur (aucune structure), 
sous l'hypothèse nulle d'actifs non corrélés.
"""
def marchenko_pastur_bounds(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    lambda_plus = sigma2 * (1 + np.sqrt(q)) ** 2
    lambda_minus = sigma2 * (1 - np.sqrt(q)) ** 2
    return lambda_minus, lambda_plus

"""Densité théorique de Marchenko-Pastur, pour superposer à l'histogramme empirique des valeurs propres."""
def marchenko_pastur_density(x: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    lambda_minus, lambda_plus = marchenko_pastur_bounds(q, sigma2)
    density = np.zeros_like(x, dtype=float)
    mask = (x > lambda_minus) & (x < lambda_plus)
    density[mask] = np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus)) / (2 * np.pi * sigma2 * q * x[mask])
    return density

"""
Filtre RMT standard (Laloux et al., 1999) :
1. Décomposition spectrale de la matrice de corrélation empirique.
2. Toute valeur propre au-dessus de lambda_plus (borne MP) est
    considérée comme du signal -> conservée telle quelle.
3. Toutes les valeurs propres restantes (bruit) sont remplacées
    par leur MOYENNE COMMUNE, ce qui préserve la trace totale
    (= N, la variance totale) tout en supprimant la structure
    fallacieuse que le bruit d'estimation introduit entre elles.
4. Reconstruction, puis renormalisation de la diagonale à 1
    (une matrice de corrélation a toujours une diagonale unité).
"""
def denoise_correlation_matrix(corr: np.ndarray, q: float) -> tuple[np.ndarray, dict]:
    n = corr.shape[0]
    eigvals, eigvecs = np.linalg.eigh(corr)   # eigh : matrice symétrique, valeurs propres triées croissant

    _, lambda_plus = marchenko_pastur_bounds(q)

    is_signal = eigvals > lambda_plus
    n_signal = int(is_signal.sum())

    eigvals_denoised = eigvals.copy()
    noise_mean = eigvals[~is_signal].mean()
    eigvals_denoised[~is_signal] = noise_mean

    corr_denoised = eigvecs @ np.diag(eigvals_denoised) @ eigvecs.T

    # Renormalisation : forcer une diagonale exactement à 1 (la
    # reconstruction spectrale peut introduire un écart numérique minime)
    d = np.sqrt(np.diag(corr_denoised))
    corr_denoised = corr_denoised / np.outer(d, d)

    info = {
        "n_signal_eigenvalues": n_signal,
        "n_noise_eigenvalues": n - n_signal,
        "lambda_plus": lambda_plus,
        "noise_mean": noise_mean,
        "top_eigenvalues": eigvals[::-1][:n_signal],
    }
    return corr_denoised, info

"""Reconstruit une covariance à partir d'une corrélation et des volatilités individuelles d'origine."""
def correlation_to_covariance(corr: np.ndarray, vols: np.ndarray) -> np.ndarray:
    return np.outer(vols, vols) * corr
