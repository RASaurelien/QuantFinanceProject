# Lead-Lag Matrix Python & Rust

Deux implémentations de la matrice lead-lag basée sur l'estimateur de
Hayashi-Yoshida, pour données tick asynchrones (HFT).

## Contenu

- `leadlag.py` prototype Python (avec numpy/pandas/matplotlib). Sert de
  référence pour valider la logique et itérer rapidement. Produit
  `leadlag_heatmap.png`.
- `rust_src/main.rs` + `rust_src/Cargo.toml` prototype Rust, parallélisée sur
  les coeurs CPU avec des threads scopés (`std::thread::scope`), "pensée" pour du
  volume réel (des millions de ticks). Produit `leadlag_matrix.csv`.

## Lancer la version Python

```bash
pip install numpy pandas matplotlib
python3 leadlag.py
```

## Lancer la version Rust

```bash
mkdir leadlag_rs && cd leadlag_rs
mkdir src
cp ../rust_src/Cargo.toml .
cp ../rust_src/main.rs src/
cargo build --release
./target/release/leadlag_rs
```

Nécessite `rustc` >= 1.63 (pour `std::thread::scope`). Aucune dépendance
externe critique, seuls `rand`/`rand_distr` (simulation) et `csv`/`serde`
(export) sont utilisés ; retirer ces deux derniers si vous branchez le code
sur un flux de données déjà en mémoire (le cœur de l'algo n'en dépend pas).

## Benchmark (6 actifs, 1800s de données, ~18k ticks/actif, 81 lags testés)

| Version | Temps de calcul de la matrice |
|---|---|
| Python (numpy, boucle pure Python par paire/lag) | ~250 s |
| Rust (threads scopés, two-pointer O(n) par paire) | ~0.9 s |

Soit environ **280x plus rapide**. L'écart vient de deux facteurs cumulés :
l'algorithme two-pointer en Rust (pas de recopie/interpolation des séries) et
l'absence d'overhead d'interprète Python dans la boucle chaude, en plus de la
parallélisation multi-coeur.

## Notes pour une mise en prod réelle

- **Convention de signe** : `lag > 0` dans `corr(i, j, lag)` signifie qu'on
  teste "i mène j" (le rendement futur de j, à t+lag, est corrélé au
  rendement présent de i). Un score de leadership positif pour la paire
  (i, j) signifie que i mène j.
- **Rolling window** : la structure lead-lag n'est pas stable dans le temps.
  En production, on recalcule cette matrice sur des fenêtres glissantes
  (ex. toutes les 5-15 minutes) plutôt qu'une fois pour toute la journée.
- **Coût algorithmique** : O(n_paires × n_lags × (n_i + n_j)) par fenêtre.
  Pour un univers de N actifs, cela grandit en O(N²), au-delà de quelques
  dizaines d'actifs, il vaut la peine de restreindre les paires testées
  (ex. via un premier filtre de corrélation contemporaine, ou une
  structure sectorielle/factorielle) plutôt que de tester toutes les paires.
- **Données réelles** : remplacer `simulate_asynchronous_ticks` par un
  chargement depuis vos flux de marché (ITCH, FIX, base tick interne, etc.) ;
  le reste du pipeline (intervalles, HY, matrice) ne change pas.
