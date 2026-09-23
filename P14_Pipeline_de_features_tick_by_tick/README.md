# Pipeline de Features Tick-by-Tick Haute Performance (Rust)

Pipeline de calcul de features en streaming sur un flux de ticks haute
fréquence, microprix, order flow imbalance, volatilité réalisée,
conçu en mémoire **O(1) par tick**, testé jusqu'à 5 millions de ticks.
Complémentaire aux projets de recherche quantitative du dépôt : ici
l'angle est l'**ingénierie**, pas l'estimation statistique.

## Pourquoi O(1) par tick, pas juste "ça marche"

Un pipeline qui stocke une fenêtre glissante de prix dans un `Vec`
(l'approche la plus naturelle) fonctionne très bien sur 100k lignes et
explose en mémoire, ou devient prohibitivement lent, sur 100M. Ce
projet utilise deux techniques pour rester en coût constant, quel que
soit le nombre de ticks déjà traités :

- **Algorithme de Welford (1962)** pour la moyenne/variance en ligne
  des rendements, un seul passage, stable numériquement, sans somme
  de carrés qui explose.
- **Moyenne mobile exponentielle (EMA)** pour le microprix lissé
  un seul état à mettre à jour, pas de fenêtre à stocker.
- Le générateur de ticks est un **itérateur** (`Iterator<Item=Tick>`),
  pas un vecteur pré-calculé : les ticks sont produits à la demande,
  jamais tous chargés en RAM.

## Résultats mesurés

- **21.4 millions de ticks/seconde** en traitement streaming complet
  (microprix, imbalance, EMA, vol réalisée), sur 5 millions de ticks.
- Benchmark naïf vs streaming : une fenêtre glissante recalculée
  entièrement à chaque tick (`Vec::remove(0)` + recalcul complet de
  l'écart-type, coût O(n·W)) est **4x plus lente** que l'approche
  Welford O(1), même sur un test volontairement modeste (200k ticks,
  fenêtre 500), l'écart se creuse encore avec n et W plus grands.

## Structure

```
src/
  generator.rs    # Flux de ticks synthétique (itérateur, mémoire O(1))
  features.rs       # StreamingFeatureEngine : microprix, imbalance, Welford, EMA
  main.rs             # Pipeline complet + benchmark naïf vs streaming
python/
  plot_results.py      # Visualisation des features (microprix, imbalance, vol)
outputs/                 # Graphique (committé : preuve visuelle pour le README)
```

## Usage

```bash
cargo build --release
./target/release/tier2_tick_pipeline    # traite 5M ticks, exporte un échantillon CSV

cd python && python plot_results.py      # génère le graphique
```

## Limites connues

- Données synthétiques (marche aléatoire + spread/tailles stochastiques), 
  un vrai flux ITCH/FIX aurait une structure d'événements plus riche
  (ajouts/annulations d'ordres, pas seulement des snapshots bid/ask).
- Le benchmark naïf vs streaming compare des métriques légèrement
  différentes (vol sur tout l'historique vs vol sur fenêtre),
  documenté explicitement dans la sortie du programme plutôt que
  présenté comme une comparaison parfaitement à périmètre égal.
- Pas de vrai parsing depuis un fichier externe (le pipeline consomme
  directement l'itérateur de génération), une extension naturelle
  serait un vrai parseur zero-copy depuis un fichier CSV/binaire sur
  disque, avec `mmap` pour éviter la copie mémoire.