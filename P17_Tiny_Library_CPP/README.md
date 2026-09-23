# Librairie de Pricing Générique, C++ Templates

Moteur de pricing Monte Carlo générique (`MonteCarloEngine<Model,
Payoff>`) où le modèle et le payoff sont résolus **à la compilation**
plutôt qu'à l'exécution, avec validation contre Black-Scholes
analytique et benchmark contre l'équivalent à dispatch dynamique
(`virtual`). Projet optionnel/avancé du Tier 3, démontre une
compétence C++ plus poussée que les projets précédents (pricing
engine, EDP), qui utilisaient déjà des classes concrètes mais pas de
généricité par templates.

## Le principe : zero-cost abstraction

`MonteCarloEngine<BlackScholesModel, CallPayoff>` et
`MonteCarloEngine<BlackScholesModel, AsianCallPayoff>` sont deux
**types C++ distincts**, générés séparément par le compilateur à la
compilation. Le compilateur voit le code complet du payoff à
l'intérieur de la boucle de simulation et peut l'inliner, même
principe que la STL ou Eigen. Avec une interface `virtual`
équivalente, chaque appel au payoff passe par une indirection de table
de fonctions virtuelles que le compilateur ne peut pas voir à travers.

`if constexpr` permet en plus au **même** moteur générique de gérer
aussi bien un payoff terminal (`Call`, `Put`, ne dépend que de `S_T`)
qu'un payoff path-dépendant (`Asian`, dépend de toute la trajectoire),
sans dupliquer le code de la classe `MonteCarloEngine`.

## Validation

| Payoff | Monte Carlo (template) | Black-Scholes (analytique) | Écart |
|---|---|---|---|
| Call | 9.4192 | 9.4134 | 0.0058 |
| Put | 6.4690 | 6.4580 | 0.0110 |

Écarts cohérents avec le bruit Monte Carlo attendu à 2M chemins (pas
de biais systématique). Le call asiatique (5.2652) est correctement
inférieur au call européen (9.4134), la moyenne arithmétique sur 252
fixings réduit mécaniquement la variance de la trajectoire finale,
donc la valeur de l'optionalité.

## Benchmark : template vs virtual, résultat honnête

| Approche | Temps (2M chemins) | Facteur |
|---|---|---|
| Template (dispatch statique) | 77.45 ms | 1.00x |
| Virtual (dispatch dynamique) | 80.28 ms | 1.04x |

Le gain mesuré est **modeste** (4%), pas spectaculaire, et c'est un
résultat honnête à expliquer plutôt qu'à cacher : pour un payoff aussi
trivial qu'un Call (une soustraction + un `max`), le coût dominant de
la boucle est la génération du nombre aléatoire gaussien
(`std::normal_distribution`), pas l'appel au payoff lui-même.
L'indirection `virtual` ne représente donc qu'une petite fraction du
temps total. Le bénéfice des templates serait beaucoup plus visible
sur un payoff plus coûteux à évaluer (produit structuré avec plusieurs
conditions, comme le pricer autocall du Tier 2) ou sur un modèle moins
coûteux que la génération gaussienne elle-même, ce que ce benchmark
ne teste pas, et le README le dit plutôt que d'arrondir le résultat
dans le sens de la conclusion attendue.

## Structure

```
include/
  pricing_lib.hpp        # Payoffs (types), BlackScholesModel, MonteCarloEngine<Model,Payoff>
  black_scholes_ref.hpp    # Formule fermée, référence de validation uniquement
src/
  main.cpp                   # Validation + démonstration Asian + benchmark
```

## Compilation & usage

```bash
g++ -std=c++17 -O3 -march=native -Iinclude -o pricer src/main.cpp
./pricer
```

## Limites connues

- Un seul modèle (Black-Scholes/GBM), l'intérêt du système de
  templates grandit avec le nombre de modèles réellement échangés
  (Heston, SABR...), pas démontré ici faute de temps.
- Le gain de performance mesuré est modeste (voir ci-dessus), la
  vraie valeur ajoutée ici est la **composabilité** (nouveaux payoffs
  sans toucher au moteur, sécurité de type à la compilation) plus que
  la vitesse brute sur ce cas précis.
- Pas de Greeks ni de réduction de variance (antithétique) dans ce
  moteur générique, volontairement minimal pour rester lisible ;
  les projets 1 et 2 du dépôt couvrent déjà ces techniques séparément.
