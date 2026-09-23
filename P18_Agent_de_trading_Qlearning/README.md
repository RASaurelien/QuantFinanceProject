# Agent de Trading par Q-Learning, Évaluation Honnête (Python)

Agent de Q-learning tabulaire sur un marché synthétique à changement de
régime, avec entraînement et test sur des **simulations indépendantes**
(seeds différentes) pour une vraie évaluation out-of-sample. Dernier
projet de la roadmap, volontairement en priorité basse : le trading par
RL est un terrain saturé, à edge non garanti, ce projet le traite avec
la même rigueur méthodologique que le reste du dépôt plutôt que d'en
survendre le résultat.

## Résultat, honnête, pas flatteur

| Politique | Rendement total (test) | Sharpe |
|---|---|---|
| Agent Q-learning | +5.92% | 0.181 |
| Buy & Hold | **+71.82%** | **0.635** |
| Politique aléatoire | -37.13% | -0.518 |

**L'agent ne bat pas le Buy & Hold.** Il bat largement la politique
aléatoire (ce qui prouve qu'il a appris *quelque chose*, voir
répartition des actions ci-dessous), mais reste loin d'un edge
exploitable. C'est le résultat le plus probable pour du Q-learning
tabulaire avec un état aussi pauvre (5 buckets de momentum × 3
positions = 15 états) sur un problème où le signal est faible et bruité, 
un constat cohérent avec la littérature sur le RL appliqué au
trading, pas une anomalie de ce projet en particulier.

**Ce que l'agent a réellement appris** : la répartition des actions en
test (Flat 85%, Long 6.6%, Short 8.4%) montre qu'il a surtout appris à
**rester à l'écart** plutôt qu'à identifier des opportunités, un
comportement rationnel compte tenu du rapport signal/bruit, mais qui
explique pourquoi il ne capture pas le fort rendement du marché
sous-jacent sur la période de test (le régime tendanciel a dominé,
récompensant une exposition longue soutenue que l'agent n'a pas prise).

## Méthodologie (ce qui rend ce résultat crédible)

- **Train et test sur des simulations indépendantes** (seeds 1 et 999), 
  pas de fuite entre les deux, contrairement à un split
  train/test sur la même trajectoire qui sous-estimerait le risque de
  surapprentissage.
- **Bornes de discrétisation fixes**, non ajustées sur les données,
  évite une forme subtile de fuite d'information (calibrer les
  quantiles sur l'ensemble de la série reviendrait à utiliser de
  l'information future).
- **Deux baselines**, pas une seule : Buy & Hold (référence de marché)
  ET politique aléatoire (référence de "l'agent a-t-il appris quelque
  chose du tout"), pour distinguer "pas d'edge" de "l'agent est cassé".

## Structure

```
src/main.py    # Environnement, agent Q-learning, entraînement, évaluation, graphique
outputs/          # Graphique (committé)
```

## Usage

```bash
cd src && python main.py
```

## Limites connues

- État très pauvre (5 buckets de momentum, pas de volatilité, pas de
  volume, pas de contexte de régime), un agent avec un espace d'état
  plus riche (ou une fonction d'approximation, DQN) pourrait capturer
  davantage de structure, au prix d'un risque de surapprentissage plus
  élevé sur des données aussi limitées.
- Une seule paire train/test, un vrai test de robustesse répéterait
  l'entraînement sur plusieurs seeds et regarderait la distribution des
  performances, comme le fait le projet de backtesting événementiel
  (Tier 2 #11).
- Récompense = rendement immédiat, pas de shaping ni d'objectif
  ajusté du risque (Sharpe différentiel, drawdown pénalisé), un choix
  de récompense différent changerait probablement le comportement
  appris.
- Coûts de transaction simples (bps fixes), cohérents avec les autres
  projets du dépôt mais pas calibrés sur un vrai carnet d'ordres.
