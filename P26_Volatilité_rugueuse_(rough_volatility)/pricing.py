"""
pricing.py
===========

European call pricing from the rough Heston characteristic function,
via the classical Heston-style P1/P2 decomposition
(Heston 1993, generalized to any model
whose characteristic function of X_T = log(S_T/S_0) is known):

    C(K, T) = S0 * P1 - K * P2
    (zero rates / dividends, as in the rest of this project)

    P2 = P(X_T > ln(K/S0))
       = 1/2 + 1/pi * int_0^inf Re[ e^{-iu*ln(K/S0)} phi(u) / (iu) ] du
    P1 = 1/2 + 1/pi * int_0^inf Re[ e^{-iu*ln(K/S0)} phi(u-i) / (iu) ] du

P1 uses the characteristic function shifted by -i,
which is exactly the "share measure" change of numeraire.
It is well defined here because we verified phi(-i, T) = E[S_T/S_0] = 1
(martingale property, see Check 3 in rough_heston.py),
so no extra normalization constant is needed.

Both integrals use a single fixed Gauss-Legendre quadrature,
mapped onto [eps, U_MAX] by a change of variables.
The (vectorized) characteristic function is reused,
so an ENTIRE smile needs only two calls to rough_heston_char_function
per maturity (one for P2's arguments, one for P1's shifted arguments),
not one call per strike.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from rough_heston import rough_heston_char_function

N_QUAD = 128
U_MAX = 80.0
EPS = 1e-4
DEFAULT_N_STEPS = 300

# NOTE on numerical stability:
# the explicit fractional Adams-Bashforth-Moulton scheme
# used to solve the Riccati equation (see rough_heston.py)
# becomes stiff for LARGE Fourier arguments u
# (the quadratic term in F grows like u^2),
# and can blow up (NaN/overflow) if n_steps is too small relative to u.
#
# Since |phi(u,T)| decays rapidly in u anyway
# (which is what makes the Fourier inversion converge),
# U_MAX is capped well before the region where the instability would bite.
# The alternative would be pushing it out to u=200+
# and compensating with a much finer (and much more expensive) time grid.


def _quadrature_nodes():
    nodes, weights = np.polynomial.legendre.leggauss(N_QUAD)
    # map from [-1, 1] to [EPS, U_MAX]
    u = 0.5 * (U_MAX - EPS) * (nodes + 1) + EPS
    w = 0.5 * (U_MAX - EPS) * weights
    return u, w


_U_NODES, _W_NODES = _quadrature_nodes()


# Vectorized European call prices
# for an array of strikes K, at a single maturity T.
def rough_heston_call_prices(K: np.ndarray, T: float, S0: float, V0: float,
                             lam: float, theta: float, nu: float, rho: float,
                             H: float, n_steps: int = DEFAULT_N_STEPS
                             ) -> np.ndarray:
    K = np.atleast_1d(np.asarray(K, dtype=float))
    logm = np.log(K / S0)  # (n_K,)

    phi_u = rough_heston_char_function(
        _U_NODES, T, V0, lam, theta, nu, rho, H, n_steps)            # (n_quad,)
    phi_u_shift = rough_heston_char_function(
        _U_NODES - 1j, T, V0, lam, theta, nu, rho, H, n_steps)       # (n_quad,)

    # integrand_j(u) = Re[ e^{-iu*logm_j} * phi(u) / (iu) ],
    # vectorized over (n_quad, n_K)
    exp_term = np.exp(-1j * np.outer(_U_NODES, logm))  # (n_quad, n_K)
    integrand_P2 = np.real(exp_term * (phi_u / (1j * _U_NODES))[:, None])
    integrand_P1 = np.real(exp_term * (phi_u_shift / (1j * _U_NODES))[:, None])

    P2 = 0.5 + (1.0 / np.pi) * (_W_NODES @ integrand_P2)
    P1 = 0.5 + (1.0 / np.pi) * (_W_NODES @ integrand_P1)

    return S0 * P1 - K * P2


def bs_call_price(S, K, T, sigma, r=0.0):
    T = np.maximum(T, 1e-12)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# Invert Black-Scholes (Brent's method) for a single option price.
def implied_vol(price: float, S0: float, K: float, T: float) -> float:
    intrinsic = max(S0 - K, 0.0)
    if price <= intrinsic + 1e-10:
        return np.nan
    try:
        return brentq(lambda sig: bs_call_price(S0, K, T, sig) - price,
                      1e-4, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan


def implied_vol_smile(K: np.ndarray, T: float, S0: float,
                      **model_params) -> np.ndarray:
    prices = rough_heston_call_prices(K, T, S0=S0, **model_params)
    return np.array([implied_vol(p, S0, k, T) for p, k in zip(prices, K)])


if __name__ == "__main__":
    # --- Sanity check: nu -> 0 collapses the model to a deterministic
    # (time-varying) instantaneous variance V_t,
    # so pricing should match Black-Scholes
    # with sigma^2 = average variance over [0, T].
    # We use a tiny but nonzero nu (1e-6),
    # since the Fourier machinery divides by things
    # that assume a genuine stochastic-vol structure.
    S0, T = 100.0, 0.5
    params = dict(lam=1.5, theta=0.04, nu=1e-6, rho=-0.6, H=0.30, n_steps=150)
    V0 = 0.04
    K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    prices = rough_heston_call_prices(K, T, S0, V0=V0, **params)

    # With nu ~ 0, V_t is deterministically mean reverting.
    # With theta = V0 (flat variance curve) and a short-ish maturity,
    # the effective constant vol is close to sqrt(V0),
    # so we compare to a plain BS price at sigma = sqrt(V0).
    bs_ref = bs_call_price(S0, K, T, sigma=np.sqrt(V0))
    print("Sanity check -- nu ~ 0 (near-deterministic variance) "
          "vs Black-Scholes reference:")
    for k, p, b in zip(K, prices, bs_ref):
        print(f"  K={k:6.1f}   rough-Heston price={p:8.4f}   "
              f"BS(sigma=sqrt(V0))={b:8.4f}   diff={p-b:+.4f}")

    print("\nImplied vols recovered from rough-Heston prices "
          "(nu~0 case, should be ~flat at "
          f"sqrt(V0)={np.sqrt(V0):.4f}):")
    ivs = implied_vol_smile(K, T, S0, V0=V0, **params)
    for k, iv in zip(K, ivs):
        print(f"  K={k:6.1f}   implied vol={iv:.4f}")
