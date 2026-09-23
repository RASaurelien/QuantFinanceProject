"""
main.py
=======
Valorise une note autocall réaliste, 
exécute deux validations analytiques 
(cas dégénérés où le prix Monte Carlo doit coïncider avec une formule fermée triviale), 
calcule le delta, et produit les graphiques utiles à une présentation de structuration : 
probabilité de rappel par date, 
distribution du payoff final, 
sensibilité du prix à la barrière de coupon.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pricer import AutocallNote, price_autocall, compute_delta


R, Q, SIGMA = 0.03, 0.02, 0.22

"""
Cas dégénéré 1 : 
barrières de coupon et d'autocall placées hors de portée (jamais atteintes), 
barrière de capital à 0 
(jamais franchie à la baisse en pratique, cf. asymptote). 
La note se réduit alors à une obligation zéro-coupon pure : 
prix = notional * exp(-r*T).
"""
def validate_pure_zero_coupon_bond() -> None:
    note = AutocallNote(notional=1000, S0=100, obs_times=[1.0, 2.0, 3.0],
                          autocall_barrier=1e6, coupon_barrier=1e6, capital_barrier=0.0, coupon_rate=0.0)
    result = price_autocall(note, R, Q, SIGMA, n_paths=200_000)

    theoretical = note.notional * np.exp(-R * note.maturity)
    error_pct = abs(result["price"] - theoretical) / theoretical * 100

    print("[validation 1] Cas dégénéré : obligation zéro-coupon pure")
    print(f"  Prix Monte Carlo : {result['price']:.4f}")
    print(f"  Prix théorique (notional * exp(-rT)) : {theoretical:.4f}")
    print(f"  Écart : {error_pct:.3f}%  {'OK' if error_pct < 0.5 else 'ATTENTION'}\n")

"""
Cas dégénéré 2 : 
barrière d'autocall quasi nulle -> la note est presque sûrement rappelée, 
à la toute première date d'observation.
Le prix doit alors converger vers (notional + coupon) 
actualisé à la première date, à la probabilité de rappel près (quasi 100%).
"""
def validate_immediate_autocall() -> None:
    note = AutocallNote(notional=1000, S0=100, obs_times=[1.0, 2.0, 3.0],
                          autocall_barrier=0.01, coupon_barrier=0.01, capital_barrier=0.70, coupon_rate=0.08)
    result = price_autocall(note, R, Q, SIGMA, n_paths=200_000)

    theoretical = (note.notional + note.coupon_rate * note.notional) * np.exp(-R * note.obs_times[0])
    error_pct = abs(result["price"] - theoretical) / theoretical * 100

    print("[validation 2] Cas dégénéré : rappel quasi certain dès la première date")
    print(f"  Probabilité de rappel à t=1 : {result['autocall_prob_by_date'][1.0]*100:.2f}% (attendu : ~100%)")
    print(f"  Prix Monte Carlo : {result['price']:.4f}")
    print(f"  Prix théorique ((notional+coupon) * exp(-r*t1)) : {theoretical:.4f}")
    print(f"  Écart : {error_pct:.3f}%  {'OK' if error_pct < 0.5 else 'ATTENTION'}\n")


def plot_autocall_probabilities(result: dict, outpath: str) -> None:
    labels = [f"Année {t:.0f}" if isinstance(t, float) else t for t in result["autocall_prob_by_date"].keys()]
    probs = [p * 100 for p in result["autocall_prob_by_date"].values()]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = ["navy"] * (len(labels) - 1) + ["firebrick"]
    ax.bar(labels, probs, color=colors)
    for i, p in enumerate(probs):
        ax.text(i, p, f"{p:.1f}%", ha="center", va="bottom")
    ax.set_ylabel("Probabilité (%)")
    ax.set_title("Probabilité de rappel anticipé par date d'observation")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[main] Graphique des probabilités de rappel exporté -> {outpath}")

"""Diagramme de payoff classique de structuration : valeur de remboursement à l'échéance en fonction de S_T/S0 (SI jamais rappelée avant)."""
def plot_payoff_diagram(note: AutocallNote, outpath: str) -> None:
    perf = np.linspace(0.3, 1.4, 300)
    S_T = perf * note.S0

    redemption = np.where(S_T >= note.capital_barrier * note.S0, note.notional,
                             note.notional * S_T / note.S0)
    coupon_final = np.where(S_T >= note.coupon_barrier * note.S0, note.coupon_rate * note.notional, 0.0)
    total_payoff = redemption + coupon_final

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(perf * 100, total_payoff, color="navy", lw=2)
    ax.axvline(note.capital_barrier * 100, color="firebrick", ls="-", lw=1.2,
                label=f"Barrière de protection ({note.capital_barrier*100:.0f}%)")
    ax.axvline(100, color="gray", ls=":", lw=1.2, label="Niveau initial (100%)")
    ax.set_xlabel("Performance du sous-jacent à l'échéance (%)")
    ax.set_ylabel("Remboursement (+ coupon final si dû)")
    ax.set_title("Diagramme de payoff à l'échéance (si jamais rappelée avant)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[main] Diagramme de payoff exporté -> {outpath}")

"""Sensibilité du prix à la barrière de capital, utile pour une desk qui doit choisir où placer la barrière."""
def plot_sensitivity(note: AutocallNote, outpath: str) -> None:
    barriers = np.linspace(0.5, 0.9, 9)
    prices = []
    for b in barriers:
        n = AutocallNote(**{**note.__dict__, "capital_barrier": b})
        prices.append(price_autocall(n, R, Q, SIGMA, n_paths=80_000)["price_pct_notional"])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(barriers * 100, prices, "o-", color="darkorange", lw=2)
    ax.set_xlabel("Barrière de protection du capital (% du niveau initial)")
    ax.set_ylabel("Prix (% du notional)")
    ax.set_title("Sensibilité du prix à la barrière de protection")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[main] Sensibilité à la barrière exportée -> {outpath}")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" Pricer Autocall (Athena/Phoenix à mémoire), Monte Carlo")
    print("=" * 60 + "\n")

    validate_pure_zero_coupon_bond()
    validate_immediate_autocall()

    note = AutocallNote(notional=1000, S0=100, obs_times=[1.0, 2.0, 3.0],
                          autocall_barrier=1.00, coupon_barrier=0.70,
                          capital_barrier=0.70, coupon_rate=0.08)

    result = price_autocall(note, R, Q, SIGMA, n_paths=300_000)
    delta = compute_delta(note, R, Q, SIGMA, n_paths=150_000)

    print("- Note de base (S0=100, autocall 100%, coupon/protection 70%, coupon 8%/an)-")
    print(f"Prix : {result['price']:.2f}  ({result['price_pct_notional']:.2f}% du notional)  "
          f"[erreur standard : {result['stderr']:.2f}]")
    print(f"Delta : {delta:.4f}  (variation de prix pour 1 unité de S0)")
    print(f"Probabilité de rappel anticipé (avant échéance) : {result['prob_autocall_total']*100:.1f}%")
    print(f"Probabilité de perte en capital à l'échéance : {result['prob_capital_loss']*100:.1f}%")
    print(f"Durée de vie moyenne attendue : {result['expected_life']:.2f} ans\n")

    print("Probabilité de rappel par date :")
    for date, prob in result["autocall_prob_by_date"].items():
        label = f"Année {date:.0f}" if isinstance(date, float) else date
        print(f"  {label:25s} {prob*100:5.1f}%")
    print()

    plot_autocall_probabilities(result, str(output_dir / "autocall_probabilities.png"))
    plot_payoff_diagram(note, str(output_dir / "payoff_diagram.png"))
    plot_sensitivity(note, str(output_dir / "barrier_sensitivity.png"))


if __name__ == "__main__":
    main()
