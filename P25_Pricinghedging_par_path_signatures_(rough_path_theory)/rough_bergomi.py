"""
rough_bergomi.py
=================

Simulation of the rough Bergomi model (Bayer, Friz & Gatheral, 2016,
"Pricing under rough volatility", Quantitative Finance 16(6)), built on the
empirical finding that log-volatility behaves like a fractional Brownian
motion with Hurst exponent H ~ 0.1 (Gatheral, Jaisson & Rosenbaum,
"Volatility is Rough", Quantitative Finance 18(6), 2018, arXiv:1410.3394).

Model
-----
    W^H_t = sqrt(2H) \\int_0^t (t-s)^{H-1/2} dZ_s          (Riemann-Liouville fBm)
    V_t   = xi0 * exp( eta * W^H_t - 0.5 * eta^2 * t^{2H} )
    dS_t  = S_t * sqrt(V_t) * dB_t ,   dB_t = rho dZ_t + sqrt(1-rho^2) dZ'_t

with Z, Z' independent standard Brownian motions and rho < 0 the usual
equity leverage correlation. H < 1/2 is what makes the spot-vol path rough
and non-Markovian in the price, unlike a classical one-factor stochastic
vol model such as Heston: volatility here genuinely remembers the whole
path of Z, not just its current level. That's exactly the setting where a
plain Black-Scholes delta, a function of (t, S_t) only, structurally can't
see information the market has already priced in, and where a richer
path-dependent statistic like the signature has room to help. That gap is
what this project measures (see hedge_experiment.py).

Simulation method
------------------
The stochastic Volterra integral defining W^H is discretized with a
left-point Riemann sum on the Brownian increments:

    W^H_{t_k} ~= sqrt(2H) * sum_{i=0}^{k-1} (t_k - t_i)^{H-1/2} * dZ_i

This is the standard, simple discretization used throughout the
rough-vol literature for illustrative and research code. It's O(dt) biased
relative to the true Volterra integral (the kernel is only evaluated at
left endpoints of each sub-interval, never at t_k - t_k = 0, so there's no
singularity to deal with). For production-grade accuracy at large step
counts, the O(n log n) hybrid scheme of Bennedsen, Lunde & Pakkanen (2017,
"Hybrid scheme for Brownian semistationary processes", Finance &
Stochastics 21) handles the near-singular diagonal kernel weights exactly
and the far ones with this same Riemann-sum idea.

The whole convolution collapses to one matrix multiplication (kernel
matrix K times the increments dZ), fully vectorized across every Monte
Carlo path at once.
"""

from __future__ import annotations
import numpy as np


def simulate_rough_bergomi(n_paths: int, n_steps: int, T: float,
                            S0: float, xi0: float, H: float, eta: float,
                            rho: float, seed: int = 0):
    """
    Simulate N paths of the rough Bergomi model.

    Parameters
    ----------
    n_paths : number of Monte Carlo paths
    n_steps : number of time steps to expiry T
    T       : expiry (years)
    S0      : spot price
    xi0     : flat forward variance curve level (xi0 = V_0)
    H       : Hurst exponent of the driving fBm (~0.1 empirically)
    eta     : vol-of-vol parameter
    rho     : correlation between the variance driver Z and the price
              driver B (rho < 0: leverage effect)
    seed    : RNG seed

    Returns
    -------
    time_grid : (n_steps+1,), t_0=0 ... t_n=T
    S : (n_paths, n_steps+1), asset price paths
    V : (n_paths, n_steps+1), instantaneous variance paths
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    # Brownian increments driving the variance process (Z) and an
    # independent Brownian motion (Z') used to build the correlated price
    # driver B. Shapes: (n_paths, n_steps).
    dZ = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    dZperp = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    dB = rho * dZ + np.sqrt(1 - rho ** 2) * dZperp

    # Riemann-Liouville fBm via the left-point discretized Volterra kernel.
    # K[k, i] = (t_{k+1} - t_i)^{H-1/2} for i <= k, else 0.
    n = n_steps
    K = np.zeros((n, n))
    for k in range(n):
        idx = np.arange(k + 1)               # i = 0..k
        gap = t[k + 1] - t[idx]               # t_{k+1} - t_i > 0 always
        K[k, idx] = gap ** (H - 0.5)
    WH = np.sqrt(2 * H) * (K @ dZ.T).T        # (n_paths, n)

    # Spot variance V_t = xi0 * exp(eta*W^H_t - 0.5*eta^2*t^(2H))
    V = xi0 * np.exp(eta * WH - 0.5 * eta ** 2 * t[1:][None, :] ** (2 * H))
    V = np.concatenate([np.full((n_paths, 1), xi0), V], axis=1)  # prepend V_0 = xi0

    # Log-Euler scheme for the asset price, using the left-point variance
    # over each sub-interval [t_i, t_{i+1}).
    V_left = V[:, :-1]
    dlogS = -0.5 * V_left * dt + np.sqrt(V_left) * dB
    # log(S0) has to be added to every column, not just tacked on as the
    # first one: cumsum(dlogS) alone only gives the log-return since t=0,
    # not the log-price itself.
    cum = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dlogS, axis=1)], axis=1)
    logS = np.log(S0) + cum
    S = np.exp(logS)

    return t, S, V


if __name__ == "__main__":
    t, S, V = simulate_rough_bergomi(
        n_paths=20000, n_steps=60, T=0.25, S0=100.0, xi0=0.04, H=0.10, eta=1.5, rho=-0.7, seed=1
    )
    print("S0 check:                 ", S[:, 0].mean())
    print("S_T mean / std:            ", S[:, -1].mean(), S[:, -1].std())
    print("instantaneous vol range:   ", np.sqrt(V.min()), np.sqrt(V.max()))
    print("avg terminal spot vol:     ", np.sqrt(V[:, -1]).mean())
    logvol = 0.5 * np.log(V)
    inc1 = np.diff(logvol, axis=1)
    print("std of 1-step log-vol increments:", inc1.std())
