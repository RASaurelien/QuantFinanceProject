"""
main.py
=======
Pipeline complet de prédiction du régime de volatilité (calme/stress)
du jour suivant, à partir de features de marché rétrospectives :

1. Simule un processus GBM à changement de régime markovien (régime VRAI connu, 
    permet une validation objective, 
    impossible sur données réelles où le régime n'est jamais directement observé).
2. Construit des features glissantes sans fuite temporelle.
3. Split train/test CHRONOLOGIQUE (pas de shuffle, une prédiction ne doit jamais 
    être validée avec des données futures).
4. Entraîne un Random Forest, compare à une baseline naïve 
    ("persistance" : le régime de demain = le régime d'aujourd'hui).
5. Interprétabilité SHAP : quelles features pèsent vraiment dans la décision, 
    plutôt qu'un modèle boîte noire dont on ne sait pas pourquoi il prédit ce qu'il prédit.

Choix de Random Forest plutôt qu'un réseau de neurones profond : 
sur un problème tabulaire de cette taille, un RF est aussi performant (ou plus), 
s'entraîne en quelques secondes, et surtout reste interprétable via SHAP,
un point explicitement recherché plutôt qu'un "trading bot deep learning" opaque.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                              classification_report)
import shap

from simulate import simulate_regime_switching_returns
from features import build_features_and_target

"""Baseline la plus simple qui soit : demain = aujourd'hui. Tout modèle ML doit au moins la battre."""
def naive_persistence_baseline(df: pd.DataFrame, valid_index: pd.Index) -> pd.Series:
    return df.loc[valid_index, "regime"]


def plot_regime_timeline(dates, true_regime, pred_regime, outpath: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].fill_between(dates, 0, 1, where=(true_regime == 1), color="firebrick", alpha=0.5,
                          step="mid", label="Régime stress (vrai)")
    axes[0].set_yticks([])
    axes[0].set_title("Régime VRAI (connu par construction de la simulation)")
    axes[0].legend(loc="upper right")

    axes[1].fill_between(dates, 0, 1, where=(pred_regime == 1), color="navy", alpha=0.5,
                          step="mid", label="Régime stress (prédit, out-of-sample)")
    axes[1].set_yticks([])
    axes[1].set_title("Régime PRÉDIT par le Random Forest (échantillon test)")
    axes[1].legend(loc="upper right")

    fig.suptitle("Détection du régime de volatilité : vérité terrain vs prédiction")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[main] Timeline des régimes exportée -> {outpath}")


def plot_confusion_matrix(cm: np.ndarray, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Calme (0)", "Stress (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
    ax.set_title("Matrice de confusion (test, out-of-sample)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[main] Matrice de confusion exportée -> {outpath}")


def main() -> None:
    import os
    os.makedirs("../outputs", exist_ok=True)

    print("=" * 60)
    print(" Prédiction du régime de volatilité (Random Forest + SHAP)")
    print("=" * 60 + "\n")

    # --- 1) Simulation ---
    df = simulate_regime_switching_returns(n_days=2500)
    print(f"[main] {len(df)} jours simulés. Répartition des régimes : "
          f"{(df['regime'] == 0).sum()} calme / {(df['regime'] == 1).sum()} stress "
          f"({df['regime'].mean()*100:.1f}% du temps en stress)\n")

    # --- 2) Features + cible ---
    X, y = build_features_and_target(df)
    print(f"[main] {X.shape[0]} observations exploitables, {X.shape[1]} features, "
          f"après retrait du warmup des fenêtres glissantes.\n")

    # --- 3) Split chronologique (70% train / 30% test), PAS de shuffle ---
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"[main] Train : {X_train.index[0].date()} -> {X_train.index[-1].date()} ({len(X_train)} jours)")
    print(f"[main] Test  : {X_test.index[0].date()} -> {X_test.index[-1].date()} ({len(X_test)} jours)\n")

    # --- 4) Baseline naïve : persistance (demain = aujourd'hui) ---
    baseline_pred = naive_persistence_baseline(df, X_test.index)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    # --- 5) Random Forest ---
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    rf_acc = accuracy_score(y_test, pred)
    rf_auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)

    print("--- Résultats (out-of-sample) ---")
    print(f"Baseline naïve (persistance)  : accuracy = {baseline_acc*100:.1f}%")
    print(f"Random Forest                  : accuracy = {rf_acc*100:.1f}%   AUC = {rf_auc:.3f}")
    print(f"Gain vs baseline                : {(rf_acc - baseline_acc)*100:+.1f} points\n")
    print("Rapport de classification détaillé :")
    print(classification_report(y_test, pred, target_names=["Calme", "Stress"]))

    if rf_acc <= baseline_acc:
        print("Remarque : l'accuracy globale favorise la baseline naïve, attendu, puisque les régimes\n"
              "sont très persistants par construction (peu de jours de transition). Voir ci-dessous\n"
              "l'analyse spécifique sur les jours de transition, où cette baseline est structurellement\n"
              "TOUJOURS fausse : c'est là que la valeur ajoutée du modèle doit se voir, ou pas.\n")

    # --- 6bis) Analyse ciblée : jours de TRANSITION de régime ---
    # Un jour de transition = le régime de demain (y_test) diffère du régime
    # d'aujourd'hui (df.loc[..., 'regime']). Par construction, la baseline
    # "persistance" a une accuracy de 0% EXACTEMENT sur ces jours (elle prédit
    # toujours "pas de changement"). C'est le sous-ensemble qui répond à la
    # vraie question utile : le modèle anticipe-t-il les BASCULES de régime,
    # pas seulement leur continuation ?
    today_regime = df.loc[X_test.index, "regime"].to_numpy()
    is_transition = (y_test.to_numpy() != today_regime)
    n_transitions = is_transition.sum()

    if n_transitions > 0:
        rf_acc_transitions = accuracy_score(y_test.to_numpy()[is_transition], pred[is_transition])
        rf_acc_stable = accuracy_score(y_test.to_numpy()[~is_transition], pred[~is_transition])
        print("--- Analyse ciblée : jours de transition de régime ---")
        print(f"Jours de transition dans le test : {n_transitions} / {len(y_test)} "
              f"({n_transitions/len(y_test)*100:.1f}%)")
        print(f"Baseline naïve sur ces jours       : 0.0% (fausse par construction)")
        print(f"Random Forest sur ces jours          : {rf_acc_transitions*100:.1f}%")
        print(f"Random Forest sur les jours stables    : {rf_acc_stable*100:.1f}%\n")

    # --- 6) Interprétabilité SHAP ---
    print("[main] Calcul des valeurs SHAP (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Pour un classifieur binaire, shap_values peut être une liste [classe0, classe1]
    # selon la version de shap -- on prend systématiquement la classe "stress" (1).
    sv_stress = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]

    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(sv_stress, X_test, show=False, plot_size=None)
    plt.title("Importance des features (SHAP), prédiction du régime de stress")
    plt.tight_layout()
    plt.savefig("../outputs/shap_summary.png", dpi=140)
    plt.close(fig)
    print("[main] Résumé SHAP exporté -> ../outputs/shap_summary.png")

    # --- 7) Graphiques de diagnostic ---
    plot_regime_timeline(X_test.index, y_test.to_numpy(), pred, "../outputs/regime_timeline.png")
    plot_confusion_matrix(cm, "../outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()
