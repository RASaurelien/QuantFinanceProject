# VaR / Expected Shortfall, Backtest et Test de Kupiec (Python)

Trois méthodes de calcul de la Value-at-Risk (VaR) et de l'Expected
Shortfall (ES/CVaR), gaussienne, historique, Monte Carlo (Student-t),
comparées par un **backtest glissant honnête** (out-of-sample strict) et
validées par le **test de Kupiec** (Proportion of Failures), la
procédure de validation de modèle de risque standard en régulation
bancaire (Bâle).

## Pourquoi ce projet

La VaR est LA mesure de risque réglementaire (Bâle II/III), mais une
VaR mal calibrée est pire qu'inutile : elle donne un faux sentiment de
sécurité. Ce projet ne se contente pas de calculer une VaR, il
**backteste** chaque méthode pour vérifier si elle tient ses promesses
statistiques, exactement comme l'exige un régulateur ou un desk de
risque avant de valider un modèle interne.

## Résultat principal

Sur des données à queues épaisses (GARCH(1,1), innovations de Student
à 5 degrés de liberté, réaliste pour des rendements actions
quotidiens), backtest glissant sur 1700 jours hors échantillon, VaR à
99% (taux de dépassement attendu : 1%) :

| Méthode | Dépassements observés | Taux observé | p-value (Kupiec) | Verdict |
|---|---|---|---|---|
| Gaussienne | 31 / 1700 | 1.82 % | 0.0022 | **REJETÉE** (mal calibrée) |
| Historique | 27 / 1700 | 1.59 % | 0.0248 | **REJETÉE** (mal calibrée) |
| **Monte Carlo (Student-t)** | **19 / 1700** | **1.12 %** | **0.6323** | **Non rejetée** (bien calibrée) |

La VaR gaussienne sous-estime systématiquement le risque de queue
(près de 2× trop de dépassements), résultat attendu sur des données à
queues épaisses, mais démontré ici plutôt qu'affirmé. La méthode
historique fait mieux mais reste rejetée : sur une fenêtre de 500
jours, elle est trop lente à intégrer un changement de régime de
volatilité (elle "regarde en arrière" par construction). Seule la
méthode Monte Carlo, qui ajuste explicitement une distribution à
queues épaisses avant de simuler, passe le test.
(ATTENTION, le script peu prendre un peu de temps)

## Méthode

### Les trois calculs de VaR/ES

1. **Gaussienne** (forme fermée) : `VaR = -(mu - z_alpha * sigma)`,
   rapide mais suppose des rendements normaux.
2. **Historique** : quantile empirique des rendements passés, aucune
   hypothèse de distribution, mais limitée par la taille de
   l'échantillon et lente à réagir à un changement de régime.
3. **Monte Carlo** : ajuste une loi de Student par maximum de
   vraisemblance sur la fenêtre glissante, puis simule 20 000 tirages
   pour estimer VaR/ES, capture les queues épaisses tout en lissant
   le bruit d'échantillonnage de la méthode historique.

### Backtest glissant (out-of-sample strict)

À chaque date `t`, la VaR est estimée **uniquement** à partir des 500
jours précédents `[t-500, t-1]`, puis comparée au rendement **réalisé**
au jour `t`. Utiliser la même fenêtre pour estimer et valider la VaR
serait un backtest en échantillon, il ne dirait rien de la
performance réelle du modèle en production.

### Test de Kupiec (1995), Proportion of Failures

Sous H0 ("le modèle est bien calibré"), le nombre de dépassements suit
une loi binomiale de paramètre `p = 1-alpha`. Le test du rapport de
vraisemblance :

```
LR = -2 * ln[ (1-p)^(n-x) p^x / (1-x/n)^(n-x) (x/n)^x ]  ~ chi2(1) sous H0
```

compare le taux de dépassement observé au taux théorique attendu.

## Structure

```
  simulate.py       # GARCH(1,1) à innovations Student (queues épaisses réalistes)
  var_methods.py   # VaR/ES : gaussienne, historique, Monte Carlo (Student-t)
  backtest.py      # Backtest glissant + test de Kupiec (POF)
  main.py          # Pipeline complet, graphiques
requirements.txt
outputs/           # Graphiques générés par main.py
```

## Usage

```bash
pip install -r requirements.txt
python main.py    # ~70-90 secondes (backtest sur 1700 jours x 3 méthodes)
```

Les graphiques sont écrits dans `outputs/`, à côté de `main.py`, quel que soit
le dossier depuis lequel le script est lancé.

## Limites connues

- Fenêtre de backtest unique (500 jours), un vrai processus de
  validation testerait la sensibilité du résultat à plusieurs tailles
  de fenêtre.
- Le test de Kupiec vérifie le TAUX de dépassement, pas leur
  regroupement dans le temps (clustering des violations, symptôme
  qu'un modèle réagit mal aux changements de régime), le test de
  Christoffersen (indépendance des dépassements) serait un complément
  naturel.
- Un seul niveau de confiance testé (99%), la pratique bancaire
  (Bâle) teste généralement plusieurs niveaux (95%, 97.5%, 99%) et sur
  plusieurs horizons (1 jour, 10 jours).
- La méthode Monte Carlo suppose une Student-t i.i.d. par fenêtre, un
  vrai modèle de risque de production combinerait souvent cette
  approche avec un GARCH filtré (VaR conditionnelle), plutôt qu'un
  ajustement de distribution non-conditionnel sur la fenêtre brute.
