"""
main.py
=======
Simule les événements, calcule les CAR par groupe (surprise positive,
surprise négative, placebo), teste la significativité du CAR à
+60 jours, et trace le profil de dérive post-annonce.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate import simulate_events, PRE_EVENT
from event_study import compute_car_matrix, car_ttest


def main():
    import os
    os.makedirs("../P20_Event_study_derive_post_annonce_de_resultats_(PEAD)/outputs", exist_ok=True)

    print("=" * 60)
    print(" Event study -- dérive post-annonce de résultats (PEAD)")
    print("=" * 60 + "\n")

    events, t_event = simulate_events(n_events=2000)
    print(f"{len(events)} événements simulés "
          f"({(~events['is_placebo']).sum()} avec vraie dérive, {events['is_placebo'].sum()} placebo)\n")

    car_matrix = compute_car_matrix(events)
    day_60_idx = int(np.searchsorted(t_event, 60))

    groups = {
        "Surprise positive (vraie dérive)": (~events["is_placebo"]) & (events["surprise"] > 0),
        "Surprise négative (vraie dérive)": (~events["is_placebo"]) & (events["surprise"] < 0),
        "Placebo (surprise positive, PAS de vraie dérive)": events["is_placebo"] & (events["surprise"] > 0),
        "Placebo (surprise négative, PAS de vraie dérive)": events["is_placebo"] & (events["surprise"] < 0),
    }

    print(f"{'Groupe':<48}{'N':>6}{'CAR[0,+60]':>13}{'t-stat':>9}{'p-value':>10}  Verdict")
    print("-" * 100)
    car_by_group = {}
    for name, mask in groups.items():
        sub_car = car_matrix[mask.to_numpy()]
        car_by_group[name] = sub_car.mean(axis=0)
        res = car_ttest(sub_car, day_60_idx)
        sig = "significatif" if res["p_value"] < 0.05 else "non significatif"
        print(f"{name:<48}{res['n']:>6}{res['mean_car']*100:>12.3f}%{res['t_stat']:>9.2f}"
              f"{res['p_value']:>10.4f}  {sig}")

    print("\n-> Les groupes 'vraie dérive' doivent être fortement significatifs, avec un CAR")
    print("   qui continue de dériver APRÈS le jour de l'annonce (jour 0) -- c'est exactement")
    print("   la définition de PEAD : le marché sous-réagit initialement à la surprise, puis")
    print("   le prix continue de s'ajuster pendant plusieurs semaines. Les groupes placebo")
    print("   doivent être plats et non significatifs -- contrôle négatif de la méthodologie.\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Surprise positive (vraie dérive)": "seagreen",
              "Surprise négative (vraie dérive)": "firebrick",
              "Placebo (surprise positive, PAS de vraie dérive)": "gray",
              "Placebo (surprise négative, PAS de vraie dérive)": "lightgray"}
    styles = {"Surprise positive (vraie dérive)": "-", "Surprise négative (vraie dérive)": "-",
              "Placebo (surprise positive, PAS de vraie dérive)": "--",
              "Placebo (surprise négative, PAS de vraie dérive)": "--"}

    for name, car in car_by_group.items():
        ax.plot(t_event, car * 100, label=name, color=colors[name], ls=styles[name], lw=1.8)

    ax.axvline(0, color="black", lw=1, ls=":")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Jours autour de l'annonce (0 = annonce)")
    ax.set_ylabel("CAR moyen (%)")
    ax.set_title("Dérive post-annonce de résultats (PEAD) : CAR moyen par groupe")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("../P20_Event_study_derive_post_annonce_de_resultats_(PEAD)/outputs/pead_car.png", dpi=140)
    plt.close(fig)
    print("Graphique exporté -> ../P20_Event_study_derive_post_annonce_de_resultats_(PEAD)/outputs/pead_car.png")


if __name__ == "__main__":
    main()
