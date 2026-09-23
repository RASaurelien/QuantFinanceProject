# Résolveur EDP Options Américaines (Crank-Nicolson + Brennan-Schwartz)

Résolution de l'équation de Black-Scholes-Merton par différences finies pour
pricer des options **américaines** (exercice anticipé possible), avec
validation croisée contre la formule fermée de Black-Scholes en mode
européen. Suite directe du projet 1 (pricing engine C++), même structure
`Option`.

## Pourquoi ce projet

Le pricing analytique (Black-Scholes) ne fonctionne que pour des options
**européennes**. Dès qu'un contrat autorise l'exercice anticipé (la quasi-
totalité des options sur actions cotées aux US, par exemple), il n'existe
pas de formule fermée : il faut résoudre un problème de complémentarité
linéaire (LCP), à chaque instant, la valeur de l'option est le maximum
entre la valeur de continuation (attendre) et la valeur d'exercice
immédiat (le payoff).

Ce projet montre :
1. La compréhension du lien EDP ↔ pricing (pas seulement l'utilisation
   d'une formule).
2. Un choix d'algorithme **motivé par la performance** (Brennan-Schwartz
   plutôt que PSOR, voir ci-dessous).
3. Un résultat économique vérifiable : la prime d'exercice anticipé
   s'annule exactement pour un call sans dividende (résultat classique de
   la théorie des options, retrouvé numériquement).

## Modèle et méthode

EDP de Black-Scholes-Merton :
```
dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + (r-q)*S*dV/dS - r*V = 0
```

Discrétisée par **Crank-Nicolson** (moyenne des schémas implicite et
explicite, précision d'ordre 2 en temps et en espace, inconditionnellement
stable). Le contrat américain impose la contrainte :
```
V(S,t) >= payoff(S)   pour tout (S,t)
```

### Brennan-Schwartz plutôt que PSOR

La méthode la plus répandue pour résoudre le LCP à chaque pas de temps
est le **PSOR** (Projected Successive Over-Relaxation) : une boucle
itérative qui converge vers la solution, avec un paramètre de relaxation
à calibrer et un coût de `O(N * nb_iterations)` par pas de temps.

Ce projet utilise à la place l'algorithme de **Brennan & Schwartz (1977)** :
la matrice du schéma de Crank-Nicolson est une M-matrice à diagonale
dominante, ce qui permet d'appliquer la contrainte `max(., payoff)`
**directement pendant la substitution arrière** de l'algorithme de Thomas
(élimination tridiagonale classique). Résultat : la contrainte américaine
est résolue en `O(N)` par pas de temps, en une seule passe, sans aucune
itération ni paramètre à régler. C'est net dans les timings : le passage
européen → américain ne coûte quasiment rien de plus (voir `Temps de
calcul` à l'exécution).

## Structure

```
include/
  Option.hpp        # (repris du projet 1)
  BlackScholes.hpp   # (repris du projet 1, sert de référence de validation)
  AmericanPDE.hpp     # Interface du solveur EDP
src/
  BlackScholes.cpp
  AmericanPDE.cpp      # Crank-Nicolson + Thomas + projection Brennan-Schwartz
  main.cpp             # CLI, validation, export CSV de la frontière d'exercice
Makefile
```

## Compilation & usage

```bash
make
./american_pde [N] [M] [S0] [K] [T] [r] [sigma] [q] [call|put]

# Exemple : put américain 1 an, avec dividende
./american_pde 400 400 100 100 1.0 0.05 0.25 0.03 put
```

Un fichier `exercise_boundary.csv` (colonnes `t, S_boundary`) est généré à
chaque exécution, directement traçable (matplotlib, Excel...) pour
visualiser la frontière d'exercice anticipé dans le temps.

## Validation faite

- **EDP européenne vs Black-Scholes analytique** : écart < 0.001 sur le
  prix (erreur de discrétisation résiduelle attendue avec N=M=400,
  décroît si on raffine la grille).
- **Call américain sans dividende (q=0)** : prime d'exercice anticipé
  ≈ 0 (10⁻⁵ près, bruit numérique), conforme au théorème classique
  selon lequel il n'est jamais optimal d'exercer un call américain sans
  dividende avant l'échéance.
- **Put américain avec dividende** : prime d'exercice anticipé positive
  et significative (~3 % du prix européen dans l'exemple par défaut),
  frontière d'exercice croissante vers K à l'approche de la maturité,
  comportement conforme à la théorie.
- Garde-fou automatique : le prix américain ne peut jamais être inférieur
  au prix européen (alerte si incohérence détectée).

## Limites connues

- Grille en S uniforme (pas de raffinement local autour du strike),
  un maillage non-uniforme réduirait l'erreur de discrétisation à N fixé.
- Conditions aux limites à `Smax` approximées (valeur intrinsèque) plutôt
  que dérivées rigoureusement, impact négligeable car `Smax` est choisi
  loin de la zone d'intérêt (3x le strike par défaut).
- Un seul sous-jacent (EDP 1D), les options multi-actifs demandent soit
  une EDP à plusieurs dimensions (coût exponentiel), soit du Monte Carlo
  (voir projet 1) avec régression (Longstaff-Schwartz) pour l'exercice
  anticipé.
