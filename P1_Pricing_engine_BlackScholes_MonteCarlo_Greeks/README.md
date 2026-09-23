# Pricing Engine, Black-Scholes \& Monte Carlo (C++)

Moteur de pricing d'options vanilles européennes en C++17, comparant un
pricer analytique (Black-Scholes-Merton fermé) à un pricer par simulation
Monte Carlo parallélisé, avec calcul complet des Greeks (Delta, Gamma,
Vega, Theta, Rho) sur les deux moteurs.



## Pourquoi ce projet

Il valide trois choses en une fois :

1. Le prix Monte Carlo **converge** vers le prix fermé de Black-Scholes (validation croisée).
2. Le moteur est **optimisé** : antithetic variates + Common Random Numbers
pour les Greeks + parallélisation OpenMP.
3. Le code est **réutilisable** : `MonteCarlo::price()` fonctionne pour
n'importe quel payoff vanille sans formule fermée, peut servir de base pour de futurs projets sur des options exotiques.

## Modèle

Sous la mesure risque-neutre, le sous-jacent suit un mouvement brownien
géométrique (GBM) :

```
dS\_t = (r - q) S\_t dt + sigma S\_t dW\_t
```

Le prix terminal se simule directement (sans discrétisation intermédiaire,
car GBM a une solution exacte) :

```
S\_T = S0 \* exp\[(r - q - 0.5\*sigma^2)\*T + sigma\*sqrt(T)\*Z],   Z \~ N(0,1)


```

## Techniques de réduction de variance / performance

|Technique|Où|Effet|
|-|-|-|
|**Antithetic variates**|`simulateSum()`|Chaque tirage Z est couplé à -Z ; réduit la variance de l'estimateur de prix sans biais, à coût quasi nul.|
|**Common Random Numbers (CRN)**|`priceWithGreeks()`|Les scénarios "bumpés" (S0+h, sigma+h, ...) réutilisent exactement les mêmes tirages aléatoires que le scénario de base → le bruit Monte Carlo s'annule dans la différence finie, donc les Greeks convergent bien plus vite que le prix brut.|
|**Parallélisation OpenMP**|`simulateSum()`|Un flux RNG (mt19937\_64) indépendant par thread, sans verrou, avec réduction OpenMP sur les sommes.|
|**-O3 -march=native**|`Makefile`|Vectorisation SIMD et instructions spécifiques au CPU.|

## 

## Structure

```
include/
  Option.hpp        # Structure du contrat + résultat de pricing
  BlackScholes.hpp   # Interface du pricer fermé
  MonteCarlo.hpp      # Interface du pricer par simulation
src/
  BlackScholes.cpp    # Formules fermées (prix + Greeks analytiques)
  MonteCarlo.cpp      # Simulation GBM + Greeks par différences finies (CRN)
  main.cpp            # CLI, comparaison, timings
Makefile
```

## 

## Compilation \& usage

```bash
make                 # compile avec -O3 -march=native -fopenmp
./pricer \[nPaths] \[S0] \[K] \[T] \[r] \[sigma] \[q] \[call|put]

# Exemple : call ATM, 2M chemins
./pricer 2000000 100 100 0.5 0.03 0.20 0.0 call
```

Tous les arguments sont optionnels (valeurs par défaut : call ATM 6 mois).

## 

## Validation

* Écart Monte Carlo / Black-Scholes < 1 erreur standard sur toutes les
configurations testées (call/put, avec/sans dividende).
* Parité call-put vérifiée numériquement à 10⁻⁷ près :
`C - P = S0\*e^(-qT) - K\*e^(-rT)`.

## 

## Limites connues

* GBM à volatilité constante : pas de smile de volatilité (voir pour un projet de calibration SVI/SABR pour une extension réaliste).
* Options européennes uniquement, le projet suivant de la roadmap (EDP/différences finies) traite l'exercice anticipé américain.
* Les Greeks Monte Carlo sont estimés par différences finies : biaisés à l'ordre O(h²) près, ce qui est acceptable ici mais mériterait des pathwise/likelihood-ratio estimators dans une version plus avancée.

