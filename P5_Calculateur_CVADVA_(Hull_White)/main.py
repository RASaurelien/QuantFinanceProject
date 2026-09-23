"""
main.py
=======
Pipeline complet de calcul CVA/DVA sur un swap de taux vanille :

1. Calibre le taux fixe "par" du swap (MtM(0) = 0).
2. Simule le taux court Hull-White jusqu'à maturité (Monte Carlo, variates antithétiques).
3. Valorise le swap à chaque date de paiement sur chaque trajectoire -> matrice de MtM simulée.
4. Calcule les profils d'exposition (EE, ENE, PFE 95%).
5. Calcule CVA, DVA, CVA bilatérale = CVA - DVA.
6. Fait tourner deux validations indépendantes du moteur desimulation (pas seulement du calcul xVA) :
    a. Cas dégénéré sigma=0 : sans aléa, un swap "par" ne bouge jamais -> exposition nulle partout -> CVA = 0.
    b. Propriété de martingale : le MtM actualisé par le numéraire "compte monétaire" 
        doit avoir une espérance nulle à toute date 
        (résultat théorique de la théorie du pricing risque-neutre), 
        vérifié empiriquement sur les simulations.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hull_white import HullWhite
from swap import InterestRateSwap
from xva import compute_exposure_profile, compute_cva, compute_dva, hazard_rate_from_cds


def run_simulation(hw: HullWhite, swap: InterestRateSwap, n_paths: int, seed: int = 42):
    r_at_payments, money_market = hw.simulate(swap.payment_times, substeps_per_period=20,
                                               n_paths=n_paths, seed=seed)
    times = np.concatenate([[0.0], swap.payment_times])

    mtm_matrix = np.zeros((n_paths, len(times)))
    for k, t in enumerate(times):
        mtm_matrix[:, k] = swap.mtm(hw, t, r_at_payments[:, k])

    return times, mtm_matrix, money_market


def validate_degenerate_case(swap: InterestRateSwap) -> None:
    """sigma=0 : aucun aléa -> le swap 'par' ne bouge jamais -> CVA doit être numériquement nulle."""
    hw_flat = HullWhite(a=0.1, sigma=1e-10, r0=0.03)   # sigma quasi nul (évite une division par 0 ailleurs)
    swap.fixed_rate = swap.par_rate(hw_flat)

    times, mtm_matrix, _ = run_simulation(hw_flat, swap, n_paths=200, seed=1)
    max_abs_mtm = np.max(np.abs(mtm_matrix))
    max_relative = max_abs_mtm / swap.notional   # comparaison en % du notional, plus lisible que l'absolu

    print(f"[validation] Cas dégénéré sigma≈0 : |MtM| max sur toutes trajectoires/dates = {max_abs_mtm:.2e} "
          f"({max_relative:.2e} du notional)")
    print(f"             -> attendu proche de 0 (swap 'par' immobile sans aléa). "
          f"{'OK' if max_relative < 1e-6 else 'ATTENTION : écart inattendu'}\n")


def validate_martingale_property(times: np.ndarray, mtm_matrix: np.ndarray,
                                  money_market: np.ndarray) -> None:
    """
    E[MtM(t) / B(t)] doit être ~0 pour tout t (le MtM actualisé par le
    numéraire risque-neutre est une martingale, MtM(0)=0 par
    construction du taux 'par'). On vérifie ça empiriquement, avec
    l'erreur standard Monte Carlo pour juger si l'écart est
    significatif ou juste du bruit d'échantillonnage.
    """
    discounted = mtm_matrix / money_market
    mean_discounted = np.mean(discounted, axis=0)
    stderr_discounted = np.std(discounted, axis=0) / np.sqrt(discounted.shape[0])

    print("[validation] Propriété de martingale : E[MtM(t)/B(t)] doit être ~0 pour tout t")
    for k, t in enumerate(times):
        n_sigma = abs(mean_discounted[k]) / stderr_discounted[k] if stderr_discounted[k] > 0 else 0.0
        flag = "OK" if n_sigma < 3 else "A SURVEILLER"
        print(f"  t={t:5.2f}  E[MtM/B]={mean_discounted[k]:+10.2f}  "
              f"stderr={stderr_discounted[k]:8.2f}  ({n_sigma:.1f} sigma)  {flag}")
    print()


def plot_exposure_profile(profile, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(profile.times, profile.ee, color="navy", lw=2, marker="o", label="EE (Expected Exposure)")
    ax.plot(profile.times, profile.pfe_95, color="crimson", lw=2, marker="o", ls="--",
             label="PFE 95%")
    ax.plot(profile.times, profile.ene, color="teal", lw=2, marker="o", label="ENE (Expected Negative Exposure)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Temps (années)")
    ax.set_ylabel("Exposition (unités monétaires)")
    ax.set_title("Profil d'exposition du swap de taux (simulation Hull-White)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[main] Profil d'exposition exporté -> {outpath}")


def plot_sample_paths(times: np.ndarray, mtm_matrix: np.ndarray, outpath: str, n_show: int = 40) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i in range(min(n_show, mtm_matrix.shape[0])):
        ax.plot(times, mtm_matrix[i], color="steelblue", alpha=0.25, lw=0.9)
    ax.plot(times, np.mean(mtm_matrix, axis=0), color="black", lw=2.2, label="MtM moyen (≈ 0, martingale)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Temps (années)")
    ax.set_ylabel("MtM du swap")
    ax.set_title(f"{n_show} trajectoires simulées de MtM")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[main] Trajectoires de MtM exportées -> {outpath}")


def main() -> None:
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print(" Calculateur CVA/DVA, Swap de taux (Hull-White)")
    print("=" * 60 + "\n")

    # --- Paramètres du marché et du swap ---
    hw = HullWhite(a=0.10, sigma=0.012, r0=0.03)
    swap = InterestRateSwap(notional=10_000_000, maturity=5.0, payment_freq=0.5)
    swap.fixed_rate = swap.par_rate(hw)
    print(f"Taux fixe 'par' du swap (5Y, notional 10M) : {swap.fixed_rate * 100:.4f}%\n")

    N_PATHS = 50_000

    # --- Validations du moteur de simulation (indépendantes du calcul xVA) ---
    validate_degenerate_case(InterestRateSwap(notional=10_000_000, maturity=5.0, payment_freq=0.5))

    times, mtm_matrix, money_market = run_simulation(hw, swap, n_paths=N_PATHS)
    validate_martingale_property(times, mtm_matrix, money_market)

    # --- Profil d'exposition ---
    profile = compute_exposure_profile(times, mtm_matrix)
    print(f"EPE (Expected Positive Exposure moyenne) : {profile.epe:,.0f}")
    print(f"PFE 95% max sur l'horizon                 : {np.max(profile.pfe_95):,.0f}\n")

    # --- CVA / DVA ---
    # Contrepartie : notation "BBB" typique, spread CDS 150 bps, recovery 40%
    # Banque elle-même : meilleure signature, spread 60 bps, recovery 40%
    lambda_counterparty = hazard_rate_from_cds(spread_bps=150, recovery=0.40)
    lambda_bank = hazard_rate_from_cds(spread_bps=60, recovery=0.40)

    cva = compute_cva(profile, discount_rate=hw.r0, hazard_rate=lambda_counterparty, recovery=0.40)
    dva = compute_dva(profile, discount_rate=hw.r0, hazard_rate_own=lambda_bank, recovery_own=0.40)
    bilateral = cva - dva

    print(f"Intensité de défaut contrepartie (150 bps, R=40%) : {lambda_counterparty*100:.3f}%/an")
    print(f"Intensité de défaut banque (60 bps, R=40%)         : {lambda_bank*100:.3f}%/an\n")
    print(f"CVA (coût du risque de défaut contrepartie)  : {cva:>12,.0f}")
    print(f"DVA (bénéfice du risque de défaut propre)     : {dva:>12,.0f}")
    print(f"CVA bilatérale (CVA - DVA)                     : {bilateral:>12,.0f}")
    print(f"CVA en % du notional                            : {cva / swap.notional * 1e4:.2f} bps\n")

    # --- Graphiques ---
    plot_exposure_profile(profile, "outputs/exposure_profile.png")
    plot_sample_paths(times, mtm_matrix, "outputs/sample_paths.png")


if __name__ == "__main__":
    main()
