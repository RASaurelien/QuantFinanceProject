# Optimisation de Portefeuille, Markowitz + Black-Litterman (Python + R)

Pipeline bi-langage : **Python** calcule la frontière efficiente de
Markowitz par formules fermées, exporte les statistiques vers **R**, qui
calibre le modèle bayésien de Black-Litterman (rendements d'équilibre +
mise à jour à partir de vues d'investisseur) et renvoie le portefeuille
optimal correspondant.

## Pourquoi ce projet

C'est le projet qui répond directement à une critique classique de
Markowitz : ses poids optimaux sont extrêmement sensibles aux
rendements espérés estimés, et sans contrainte de vente à découvert,
il produit souvent des positions absurdes (voir résultat ci-dessous).
Black-Litterman corrige ce problème en partant des rendements que le
marché implique déjà (équilibre) et en les ajustant par une mise à jour
bayésienne à partir de vues explicites, c'est l'approche standard en
gestion d'actifs institutionnelle.

## Résultat qui valide la démarche

Sur le même jeu de données (6 classes d'actifs), le portefeuille de
tangence Markowitz "naïf" (rendements historiques) donne :

| Actif | Markowitz (tangence) | Black-Litterman |
|---|---|---|
| SPY | 11.8 % | 27.1 % |
| EFA | 31.8 % | 13.4 % |
| AGG | 70.8 % | 26.9 % |
| GLD | 8.7 % | 14.9 % |
| EEM | 18.3 % | 11.4 % |
| **VNQ** | **-41.4 %** | **6.3 %** |

Le Markowitz naïf recommande une **position courte de 41 % sur
l'immobilier coté**, un artefact classique du problème mal conditionné
(petites erreurs d'estimation sur mu amplifiées par l'inversion de la
matrice de covariance). Black-Litterman, parti des poids de marché et
ajusté par seulement deux vues, donne une allocation entièrement
positive et beaucoup plus défendable. **Le projet démontre le problème
qu'il résout, pas seulement la théorie.**

## Pipeline

```
Python (mécanique numérique)                R (inférence bayésienne)
──────────────────────────────              ──────────────────────────
rendements (réels/synthétiques)
  │
  ▼
mu, Sigma annualisés
  │
  ▼
Markowitz : frontière efficiente,
min-variance, tangence
(formules fermées, théorème
des deux fonds)
  │
  ▼
export CSV (mu, Sigma, poids marché) ───►  lecture CSV
                                             │
                                             ▼
                                            Pi = delta * Sigma %*% w_mkt
                                            (rendements d'équilibre,
                                            reverse optimization)
                                             │
                                             ▼
                                            vues (P, Q, Omega)
                                             │
                                             ▼
                                            mise à jour bayésienne
                                            (conjugaison gaussienne)
                                             │
                                             ▼
                                            portefeuille de tangence
                                            sous mu_posterior
                                             │
lecture CSV  ◄─────────────────────────── export CSV + graphique (base R)
  │
  ▼
graphiques finaux (frontière +
comparaison des poids)
```

## Les deux vues d'investisseur (exemple)

- **Vue relative** : les actions émergentes (EEM) surperformeront les
  actions US (SPY) de 3 points par an.
- **Vue absolue** : l'or (GLD) délivrera un rendement de 7 % par an.

Incertitude des vues (`Omega`) calibrée selon la formule standard de
He & Litterman (1999) : proportionnelle à la variance a priori du
portefeuille de vue lui-même, sans confiance subjective à fixer à la
main.

## Choix techniques

- **Formules fermées plutôt qu'un solveur QP**, aussi bien côté Python
  (`markowitz.py`) que côté R (`black_litterman.R`) : sans contrainte
  long-only, le problème moyenne-variance a une solution analytique
  exacte (théorème des deux fonds). Même logique que le choix
  Brennan-Schwartz du projet EDP (Tier 1 #2), préférer une formule
  fermée à un solveur itératif quand la structure du problème le permet.
- **R en base R uniquement** (pas de `quadprog`, pas de `ggplot2`) :
  l'algèbre matricielle bayésienne et le graphique en barres sont
  entièrement faisables avec les fonctions de base, ce qui rend le
  script portable sans accès à CRAN.

## Structure

```
data_source.py          # Rendements réels ou synthétiques, poids de marché
markowitz.py            # Frontière efficiente et portefeuilles Markowitz
main.py                 # Orchestration Python -> R -> graphiques
black_litterman.R       # Modèle Black-Litterman en R de base
data/                   # Fichiers CSV d'échange, générés à l'exécution
outputs/                # Graphiques générés à l'exécution
.vscode/launch.json     # Configurations de débogage Python et R
requirements.txt        # Dépendances Python
```

## Installation et exécution

Depuis le dossier qui contient `main.py` :

```powershell
python -m pip install -r requirements.txt
python main.py
```

Le mode par défaut utilise les données synthétiques et ne nécessite pas
de connexion réseau. Pour tenter de télécharger trois ans de données
réelles via `yfinance` :

```powershell
python main.py --real
```

Le pipeline nécessite aussi `Rscript` dans le `PATH` :

```powershell
Rscript --version
```

Le script Python appelle automatiquement `black_litterman.R`, puis écrit
les fichiers suivants dans le dossier du projet :

```text
data/mu.csv
data/cov.csv
data/market_weights.csv
data/bl_results.csv
outputs/bl_returns_comparison.png
outputs/efficient_frontier.png
outputs/weights_comparison.png
```

## Exécution dans VS Code

Ouvrir directement le dossier qui contient `main.py`, puis :

1. Ouvrir `main.py`.
2. Appuyer sur `F5`.
3. Choisir **Debug main.py** dans la liste des configurations.

Le profil **Debug main.py (real data)** lance la variante avec `--real`.
Les messages apparaissent dans le terminal intégré. Le bouton **Run
Python File** fonctionne également si l'interpréteur Python sélectionné
contient les dépendances du fichier `requirements.txt`.

Sous Linux/macOS, les commandes équivalentes sont :

```bash
pip install -r requirements.txt
python main.py
python main.py --real
```

R utilise uniquement ses fonctions de base : aucun package R
supplémentaire n'est nécessaire. `Rscript` doit être disponible dans le
`PATH` (`sudo apt install r-base-core` sur Ubuntu/Debian).

## Limites connues

- Pas de contrainte long-only : documenté et même utilisé comme preuve
  du problème (voir résultat ci-dessus). Une extension naturelle serait
  d'ajouter une contrainte `w >= 0` via un solveur QP (`quadprog` en R,
  `scipy.optimize.minimize` en Python) et de comparer les trois
  allocations.
- Poids de marché approximatifs (pas de vraies capitalisations
  flottantes) en l'absence de données réelles.
- Deux vues seulement, à titre d'exemple, le mécanisme bayésien
  s'étend directement à un nombre quelconque de vues (il suffit
  d'ajouter des lignes à `P` et `Q`).
- `Omega` diagonale (vues supposées indépendantes), Black-Litterman
  permet en théorie une matrice pleine si les vues sont corrélées.

