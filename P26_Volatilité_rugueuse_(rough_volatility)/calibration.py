"""
calibration.py
================

Calibrates the 5 rough Heston parameters (H, lambda, theta, nu, rho)
to a target implied-volatility smile.
V0 is fixed to the ATM short-term variance, as is standard practice.

The fit is a nonlinear least squares on the implied-vol residuals
(scipy.optimize.least_squares, trust-region-reflective).

No real market data is wired into this project,
so the "target" smile is itself generated
by the SAME rough Heston pricer at KNOWN true parameters,
with a little noise added to mimic bid/ask and quoting-grid frictions.

Recovering the true parameters from the noisy smile
is the calibration correctness test:
if the optimizer converges close to the (theta, lambda, nu, rho, H)
that generated the data,
the whole pricing + calibration pipeline is self-consistent end to end.
This is the same "invert your own model" sanity check
every calibration desk runs before trusting a new pricer on live quotes.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pricing import implied_vol_smile

S0 = 100.0
T = 0.5
K = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)

# "True" parameters used to generate the synthetic target market smile.
TRUE_PARAMS = dict(H=0.10, lam=1.5, theta=0.05, nu=0.4, rho=-0.7)
V0 = 0.04

# Parameter order used throughout the optimizer: [H, lam, theta, nu, rho]
PARAM_NAMES = ["H", "lam", "theta", "nu", "rho"]
LOWER = np.array([0.02, 0.2, 0.01, 0.05, -0.95])
UPPER = np.array([0.45, 5.0, 0.25, 1.50, -0.05])
CALIBRATION_N_STEPS = 150


def smile_from_vector(x: np.ndarray,
                      n_steps: int = CALIBRATION_N_STEPS) -> np.ndarray:
    H, lam, theta, nu, rho = x
    return implied_vol_smile(K, T, S0, V0=V0, lam=lam, theta=theta,
                             nu=nu, rho=rho, H=H, n_steps=n_steps)


def residuals(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    model = smile_from_vector(x)
    r = model - target

    # A strike where the model failed to invert
    # (deep OTM/ITM, or a parameter region where the Fourier pricer
    # becomes numerically unstable, see rough_heston.py)
    # must NOT be silently scored as a perfect fit (residual 0).
    # That would reward the optimizer for drifting into a region
    # where the pricer breaks down, instead of genuinely matching the smile.
    # We penalize with a fixed, large residual instead.
    return np.where(np.isfinite(r), r, 5.0)


def main():
    print("Generating synthetic target market smile from TRUE parameters:")
    print(" ", TRUE_PARAMS)
    true_vec = np.array([TRUE_PARAMS[k] for k in
                         ["H", "lam", "theta", "nu", "rho"]])
    target_clean = smile_from_vector(true_vec)

    rng = np.random.default_rng(7)
    # ~15bp implied-vol noise
    noise = rng.normal(0, 0.0015, size=target_clean.shape)
    target_noisy = target_clean + noise
    print("Target (noisy) implied vols:", np.round(target_noisy, 4))

    # Initial guess: deliberately far from the truth,
    # so the recovery is a meaningful test of the optimizer
    # rather than a trivial local refinement.
    x0 = np.array([0.25, 1.0, 0.035, 0.7, -0.3])
    print("\nInitial guess:            ",
          dict(zip(PARAM_NAMES, np.round(x0, 3))))

    result = least_squares(
        residuals, x0, args=(target_noisy,), bounds=(LOWER, UPPER),
        xtol=1e-7, ftol=1e-7, gtol=1e-5, max_nfev=60, verbose=2
    )
    x_hat = result.x
    print("\nCalibration finished:",
          "success" if result.success else "did NOT converge cleanly")
    print(f"  iterations / function evals: {result.nfev}")
    print(f"  final cost (0.5*sum(residuals^2)): {result.cost:.3e}\n")

    print(f"{'parameter':<10}{'true':>10}{'recovered':>12}{'abs. error':>14}")
    for name, true_v, hat_v in zip(PARAM_NAMES, true_vec, x_hat):
        print(f"{name:<10}{true_v:>10.4f}{hat_v:>12.4f}"
              f"{abs(true_v-hat_v):>14.4f}")

    model_smile = smile_from_vector(x_hat)
    rmse_bp = 1e4 * np.sqrt(np.nanmean((model_smile - target_noisy) ** 2))
    print(f"\nOut-of-the-box fit quality: RMSE = {rmse_bp:.1f} bp of implied vol")

    # ------------------------------------------------------------------
    # Plot: target (noisy) smile vs calibrated model smile
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(K, 100 * target_noisy, "o", color="#333333",
            label="Target market smile (noisy)")
    ax.plot(K, 100 * target_clean, "--", color="#999999", linewidth=1,
            label="True (noise-free) smile")
    ax.plot(K, 100 * model_smile, "-", color="#C44E52", linewidth=2,
            label="Calibrated rough Heston fit")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"Rough Heston calibration (T={T}y) -- RMSE = {rmse_bp:.1f}bp")
    ax.legend()
    fig.tight_layout()
    fig.savefig("calibration_fit.png", dpi=150)
    print("Saved plot to calibration_fit.png")

    # ------------------------------------------------------------------
    # Diagnostic: is the poor recovery of H just this single-maturity smile
    # being unable to pin it down?
    #
    # We compare the ATM skew (d(implied vol)/dK near K=S0)
    # implied by the TRUE vs RECOVERED parameter sets,
    # at OTHER maturities the calibration never saw.
    #
    # If the two sets nearly agree at T=0.5 (by construction)
    # but diverge markedly at short and long maturities,
    # a single-maturity fit under-determines H.
    # A multi-maturity calibration would then be needed
    # (fitting the term structure of ATM skew, which scales as T^{H-1/2})
    # to separate H from the other parameters.
    # ------------------------------------------------------------------
    print("\n=== Diagnostic: ATM skew term structure, true vs recovered params ===")
    print(f"{'T':>6}{'skew (true H)':>16}{'skew (recovered H)':>22}")
    Kskew = np.array([95.0, 100.0, 105.0])
    for Tc in [0.1, 0.5, 1.0, 2.0]:
        iv_true = implied_vol_smile(
            Kskew, Tc, S0, V0=V0,
            **{k: TRUE_PARAMS[k] for k in ["lam", "theta", "nu", "rho", "H"]})
        iv_hat = implied_vol_smile(
            Kskew, Tc, S0, V0=V0,
            lam=x_hat[1], theta=x_hat[2], nu=x_hat[3], rho=x_hat[4], H=x_hat[0])
        skew_true = (iv_true[2] - iv_true[0]) / (Kskew[2] - Kskew[0])
        skew_hat = (iv_hat[2] - iv_hat[0]) / (Kskew[2] - Kskew[0])
        print(f"{Tc:>6.2f}{skew_true:>16.5f}{skew_hat:>22.5f}")

    return dict(true=dict(zip(PARAM_NAMES, true_vec)),
                recovered=dict(zip(PARAM_NAMES, x_hat)),
                rmse_bp=rmse_bp)


if __name__ == "__main__":
    main()
