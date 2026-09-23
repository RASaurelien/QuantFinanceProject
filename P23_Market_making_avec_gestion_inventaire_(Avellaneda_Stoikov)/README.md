# Market Making avec Gestion d'Inventaire Avellaneda-Stoikov

Implémentation du modèle de référence du market making (Avellaneda &
Stoikov, 2008), comparé à une cotation naïve à spread fixe sur des
sessions **Monte Carlo appariées** (même chemin de prix, même flux
d'ordres) pour isoler proprement l'effet de la gestion d'inventaire.
Deuxième projet du volet "quant trader" du dépôt, sur un terrain
complètement différent de mean-reversion : ici le problème n'est
pas de prédire une direction, mais de **survivre au risque
d'inventaire** en cotant en continu.

## Le modèle

Le teneur de marché ne cote pas symétriquement autour du prix milieu
`s` : il décale ses cotations selon son inventaire courant `q` via un
**prix de réservation** :

```
r(s, q, t) = s - q * gamma * sigma² * (T-t)
```

Long (`q>0`) → le MM baisse ses deux cotations pour encourager les
ventes et décourager les achats (se délester) ; short (`q<0`) →
l'inverse. Le spread optimal total combine un terme lié au risque
d'inventaire restant et un terme lié à la liquidité du marché
(intensité d'arrivée des ordres) :

```
delta = gamma*sigma²*(T-t) + (2/gamma)*ln(1 + gamma/k)
```

## Résultat (300 sessions, Monte Carlo apparié)

| | P&L moyen | Écart-type P&L | Sharpe | Écart-type inventaire |
|---|---|---|---|---|
| Avellaneda-Stoikov | 63.28 | 6.90 | **9.165** | **1.46** |
| Naïf (spread fixe) | 64.79 | 13.14 | 4.930 | 3.32 |

**P&L moyen quasi identique** (les deux stratégies partent du même
spread de départ, donc capturent un flux d'ordres comparable), mais
l'écart-type du P&L est **divisé par 2** et l'écart-type d'inventaire
réduit de **56%** grâce à Avellaneda-Stoikov quasi doublement du
Sharpe. C'est exactement l'effet recherché : ce n'est pas une
stratégie qui "gagne plus", c'est une stratégie qui **gère mieux le
risque pour un P&L comparable**, la vraie question du market making,
où le risque dominant n'est pas directionnel mais l'accumulation
incontrôlée d'inventaire.

## Pourquoi la comparaison est appariée (et pas juste deux runs séparés)

Les deux stratégies sont testées sur **exactement** le même chemin de
prix et le même tirage d'arrivées d'ordres (même seed) à chaque
session sinon, une différence de performance pourrait venir du
hasard de la simulation plutôt que de la logique de cotation
elle-même. Le spread naïf est aussi calibré pour démarrer à la même
largeur que le spread Avellaneda-Stoikov à `t=0, q=0` : la seule
différence structurelle entre les deux est l'ajustement (ou non) à
l'inventaire.

## Structure

```
src/
  avellaneda_stoikov.py    # Prix de réservation, spread optimal, cotations AS et naïves
  simulate.py                 # Session de trading : prix milieu brownien, arrivées Poisson, exécution
  main.py                        # Monte Carlo apparié, comparaison, graphiques
outputs/                          # Graphique (committé)
```

## Usage

```bash
cd src && python main.py
```

## Limites connues

- Marché à un seul MM (pas de compétition entre teneurs de marché,
  qui influencerait l'intensité effective des arrivées d'ordres).
- Paramètres `A` et `k` (intensité d'arrivée) fixes et supposés connus
  en pratique, ils doivent être calibrés sur des données réelles de
  carnet d'ordres (régression log-linéaire de l'intensité d'exécution
  observée contre la distance au mid).
- Pas de gestion du risque de sélection adverse (le flux d'ordres est
  purement poissonnien, indépendant de l'information) une extension
  reconnue du modèle de base traite ce point (Avellaneda-Stoikov
  suppose un marché "non informé", ce qui n'est pas toujours réaliste).