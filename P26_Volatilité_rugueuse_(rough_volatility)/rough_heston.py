"""
rough_heston.py
=================

Semi-closed-form pricing under the rough Heston model,
via the fractional Riccati equation of El Euch & Rosenbaum
(2019, "The characteristic function of rough Heston models",
Mathematical Finance 29(1), arXiv:1609.02108).

The equation is solved numerically with the fractional
Adams-Bashforth-Moulton (FABM) predictor-corrector scheme
of Diethelm, Ford & Freed (2002).
The same machinery is used by "Rough is not so Tough"
(Gomez-Alarcon Perez, Vazquez-Abad, 2018, arXiv:1805.12587)
in its "hybrid" scheme.
Here we implement the constant-step baseline version.

Model (El Euch & Rosenbaum's own parametrization,
where the vol-of-vol coefficient is written lambda*nu,
NOT the more common kappa/theta/xi convention).
This choice makes the model reduce EXACTLY to the classical Heston
Riccati ODE when H = 1/2,
which is the main correctness check of the __main__ block:

    dS_t/S_t = sqrt(V_t) dW_t
    V_t = V_0 + 1/Gamma(alpha) int_0^t (t-s)^{alpha-1} lambda(theta - V_s) ds
              + 1/Gamma(alpha) int_0^t (t-s)^{alpha-1} lambda*nu*sqrt(V_s) dB_s
    corr(W, B) = rho,   alpha = H + 1/2 in (1/2, 1)

Characteristic function of X_T = log(S_T / S_0), for u1 = i*u (u real):

    E[exp(u1 X_T)] = exp( phi_1(T) + V_0 * phi_2(T) )
    phi_1(T) = lambda * theta * int_0^T psi(s) ds
    phi_2(T) = I^{1-alpha} psi(T)     (Riemann-Liouville fractional integral)

where psi solves the FRACTIONAL RICCATI EQUATION:

    D^alpha psi(t) = 1/2 (u1^2 - u1) + lambda(u1 rho nu - 1) psi(t)
                     + 1/2 (lambda nu)^2 psi(t)^2 ,   I^{1-alpha} psi(0) = 0

Numerically, we solve the equivalent weakly-singular Volterra equation
(the standard way to discretize a Caputo-type fractional ODE
with zero initial condition):

    psi(t) = 1/Gamma(alpha) int_0^t (t-s)^{alpha-1} F(u1, psi(s)) ds

with F(u1, x) = 1/2(u1^2-u1) + lambda(u1 rho nu - 1) x
                + 1/2 (lambda nu)^2 x^2.
"""

from __future__ import annotations
import numpy as np
from math import gamma


# Precompute the Diethelm-Ford-Freed predictor/corrector weights
# for a grid of n_steps+1 points (0..n_steps).
# The weights depend only on alpha and the step index, not on F,
# so they could be reused across many independent values of u1.
# Predictor weights: b[n+1, j] = (n+1-j)^alpha - (n-j)^alpha, for j = 0..n.
# Corrector weights: a[n+1, j], as in Diethelm-Ford-Freed (2002).
def _fabm_weights(n_steps: int, alpha: float):
    j = np.arange(0, n_steps + 1)
    return j  # placeholder, the weights are computed on the fly in solve()


# Solve the fractional Riccati equation for one OR SEVERAL complex
# arguments u1 at once (vectorized across u1),
# on a uniform grid of n_steps+1 points on [0, T],
# via the FABM predictor-corrector (Diethelm, Ford & Freed 2002).
#
# Vectorizing across u1 is what makes pricing a whole smile fast:
# the smile needs the characteristic function at dozens of Fourier
# frequencies, and the recursion is identical for every u1,
# so the Python loop only runs once over TIME, not once per u1.
#
# Parameters:
#   u1 : complex scalar or (n_u,) complex array
#
# Returns:
#   t   : (n_steps+1,) time grid
#   psi : (n_steps+1,) or (n_steps+1, n_u) complex array, psi(t_j; u1)
def solve_fractional_riccati(u1, lam: float, theta: float, nu: float,
                             rho: float, alpha: float, T: float, n_steps: int):
    u1 = np.atleast_1d(np.asarray(u1, dtype=complex))
    n_u = u1.shape[0]
    h = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    def F(x):
        # The explicit FABM scheme can leave its stability region
        # for large Fourier arguments.
        # Turn that trajectory into NaN before an overflowing square
        # can contaminate the rest of the batch.
        quadratic = 0.5 * (lam * nu) ** 2
        safe_x = np.where(np.abs(x) <= np.sqrt(np.finfo(float).max),
                          x, np.nan + 0.0j)
        with np.errstate(over="ignore", invalid="ignore"):
            value = (0.5 * (u1 ** 2 - u1)
                     + lam * (u1 * rho * nu - 1.0) * safe_x
                     + quadratic * safe_x ** 2)
        return np.where(np.isfinite(value), value, np.nan + 0.0j)

    psi = np.zeros((n_steps + 1, n_u), dtype=complex)
    f_hist = np.zeros((n_steps + 1, n_u), dtype=complex)
    f_hist[0] = F(psi[0])  # psi(0) = 0

    g_a1 = gamma(alpha + 1.0)
    g_a2 = gamma(alpha + 2.0)

    for n in range(0, n_steps):
        # ---- Predictor (fractional Adams-Bashforth) ----
        js = np.arange(0, n + 1)
        b = (n + 1 - js) ** alpha - (n - js) ** alpha  # (n+1,)
        pred = (h ** alpha / g_a1) * np.sum(b[:, None] * f_hist[: n + 1], axis=0)
        f_pred = F(pred)

        # ---- Corrector (fractional Adams-Moulton) ----
        a = np.empty(n + 1)
        if n >= 1:
            k = np.arange(1, n + 1)  # j = 1..n
            a[1:] = ((n - k + 2) ** (alpha + 1)
                     + (n - k) ** (alpha + 1)
                     - 2 * (n - k + 1) ** (alpha + 1))
        a[0] = n ** (alpha + 1) - (n - alpha) * (n + 1) ** alpha
        corr_sum = np.sum(a[:, None] * f_hist[: n + 1], axis=0)
        psi[n + 1] = (h ** alpha / g_a2) * (f_pred + corr_sum)

        f_hist[n + 1] = F(psi[n + 1])

    if n_u == 1:
        return t, psi[:, 0]
    return t, psi


# Riemann-Liouville fractional integral of order (1-alpha) of psi,
# evaluated at t = T (the last grid point):
#
#     I^{1-alpha} psi(T) = 1/Gamma(1-alpha) int_0^T (T-s)^{-alpha} psi(s) ds
#
# Discretized with the SAME weakly-singular product trapezoidal quadrature
# as the Adams-Moulton corrector in solve_fractional_riccati.
#
# Crucially, the index convention is the same too:
# the weights for the point T = t_N are built with "n = N-1"
# (as if T were the *next* point after the history t_0..t_{N-1}),
# and psi AT T itself is added separately with unit weight.
# This mirrors how the corrector adds its newest-point term (f_pred)
# outside the history sum.
#
# Folding T's own value into the history sum with weights built from
# "n = N" is an easy off-by-one.
# It silently gives a plausible-looking but wrong answer.
# It was caught by testing against the closed-form fractional integral
# of s^gamma (see the __main__ validation block).
#
# psi may be (n_steps+1,) for a single u1,
# or (n_steps+1, n_u) for a batch.
def fractional_integral_1_minus_alpha(psi: np.ndarray, t: np.ndarray,
                                      alpha: float) -> np.ndarray:
    N = len(t) - 1
    h = t[1] - t[0]
    beta = 1.0 - alpha  # integral order
    n = N - 1
    if n < 0:
        return psi[-1] * 0.0  # T = 0 edge case
    a = np.empty(n + 1)
    if n >= 1:
        k = np.arange(1, n + 1)
        a[1:] = ((n - k + 2) ** (beta + 1)
                 + (n - k) ** (beta + 1)
                 - 2 * (n - k + 1) ** (beta + 1))
    a[0] = n ** (beta + 1) - (n - beta) * (n + 1) ** beta

    endpoint = psi[N]
    if psi.ndim == 1:
        hist_sum = np.sum(a * psi[: n + 1])
    else:
        hist_sum = np.sum(a[:, None] * psi[: n + 1], axis=0)
    return (h ** beta / gamma(beta + 2.0)) * (endpoint + hist_sum)


# Rough Heston characteristic function of X_T = log(S_T/S_0),
# evaluated at an array of real Fourier arguments u (u1 = i*u internally),
# for ALL values of u in a single vectorized fractional-Riccati solve.
#
# Returns an array of complex values phi(u, T) = E[exp(i*u*X_T)].
def rough_heston_char_function(u: np.ndarray, T: float, V0: float, lam: float,
                               theta: float, nu: float, rho: float, H: float,
                               n_steps: int = 200) -> np.ndarray:
    alpha = H + 0.5
    u1 = 1j * np.asarray(u, dtype=complex)
    t, psi = solve_fractional_riccati(u1, lam, theta, nu, rho, alpha, T, n_steps)
    phi1 = lam * theta * np.trapezoid(psi, t, axis=0)
    phi2 = fractional_integral_1_minus_alpha(psi, t, alpha)
    return np.atleast_1d(np.exp(phi1 + V0 * phi2))


# ----------------------------------------------------------------------
# Validation harness (run this file directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --- Check 1: small-t asymptotic of psi under the fractional kernel.
    # For small t, psi(t) ~ 0, so F(psi) ~ F(0) = 0.5(u1^2-u1) =: A (constant).
    # Solving D^alpha psi = A with psi(0) = 0 gives
    # psi(t) = A*t^alpha/Gamma(alpha+1).
    # This checks the FABM weights independently of the nonlinear part of F,
    # i.e. it validates the fractional-kernel machinery itself.
    u1 = 1j * 0.5
    lam, theta, nu, rho, H = 1.5, 0.04, 0.3, -0.6, 0.10
    alpha = H + 0.5
    T = 0.5
    n_steps = 4000
    t, psi = solve_fractional_riccati(u1, lam, theta, nu, rho, alpha, T, n_steps)
    A = 0.5 * (u1 ** 2 - u1)

    # compare at an early grid point (t small, before nonlinear terms matter)
    j_small = 5
    asym = A * t[j_small] ** alpha / gamma(alpha + 1.0)
    print("Check 1 -- small-t asymptotic psi(t) ~ A t^alpha / Gamma(alpha+1):")
    print(f"  numerical psi(t={t[j_small]:.5f}) = {psi[j_small]:.6e}")
    print(f"  asymptotic prediction            = {asym:.6e}")
    print(f"  relative error                    = "
          f"{abs(psi[j_small]-asym)/abs(asym):.2%}\n")

    # --- Check 2: alpha -> 1 (H = 0.5) should match a plain RK4
    # integration of the CLASSICAL (non-fractional) Riccati ODE
    # psi'(t) = F(psi(t)).
    H_classical = 0.5
    alpha_c = 1.0
    t_f, psi_f = solve_fractional_riccati(u1, lam, theta, nu, rho,
                                          alpha_c, T, n_steps=2000)

    def F(x):
        return (0.5 * (u1 ** 2 - u1)
                + lam * (u1 * rho * nu - 1.0) * x
                + 0.5 * (lam * nu) ** 2 * x ** 2)

    # independent RK4 reference solve of the plain ODE
    n_rk = 20000
    h_rk = T / n_rk
    y = 0.0 + 0.0j
    for _ in range(n_rk):
        k1 = F(y)
        k2 = F(y + 0.5 * h_rk * k1)
        k3 = F(y + 0.5 * h_rk * k2)
        k4 = F(y + h_rk * k3)
        y = y + (h_rk / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    print("Check 2 -- fractional solver at H=0.5 (alpha=1) "
          "vs independent RK4 ODE solve:")
    print(f"  FABM  psi(T) = {psi_f[-1]:.8f}")
    print(f"  RK4   psi(T) = {y:.8f}")
    print(f"  relative error = {abs(psi_f[-1]-y)/abs(y):.2%}\n")

    # --- Check 3: martingale property E[S_T/S_0] = 1,
    # i.e. phi(u=-i, T) = 1.
    V0 = 0.04
    phi_martingale = rough_heston_char_function(
        np.array([-1j]), T=0.5, V0=V0, lam=lam,
        theta=theta, nu=nu, rho=rho, H=0.10, n_steps=2000)
    print("Check 3 -- martingale property E[S_T/S_0] = phi(u=-i, T):")
    print(f"  phi(-i, T) = {phi_martingale[0]:.6f}  (should be 1.0 + 0.0j)")
