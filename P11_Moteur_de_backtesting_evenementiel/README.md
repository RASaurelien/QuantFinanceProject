# Moteur de Backtesting Événementiel (Rust + Python)

Moteur de backtesting **event-driven** au sens propre (file
d'événements FIFO explicite, pas des calculs vectorisés qui masquent
l'ordre causal), avec gestion d'ordres, slippage, commissions, et deux
stratégies systématiques (momentum et mean-reversion) confrontées sur
un banc d'essai à changement de régime. Prolongement du savoir-faire
Rust déjà démontré (lead-lag, volatilité réalisée), appliqué cette
fois à l'infrastructure de trading plutôt qu'à l'estimation statistique.

## Pourquoi "event-driven" et pas juste un calcul vectorisé

Un backtest vectorisé (calculer un signal sur toute la série d'un
coup, puis le décaler d'une période) est rapide à écrire mais cache
facilement du **look-ahead bias** : une ligne de code à peine
différente peut, sans erreur apparente, laisser fuiter de
l'information future dans le signal du jour. L'architecture
événementielle élimine structurellement ce risque : chaque composant
(stratégie, portefeuille, exécution) ne voit que l'événement qu'on lui
transmet, dans l'ordre chronologique strict :

```
Market(jour t) -> Strategy -> Signal -> Portfolio -> Order -> Execution -> Fill -> Portfolio
```

La stratégie ne reçoit jamais que `&history[..=t]` ; le portefeuille ne
décide de la taille de position qu'après avoir vu le signal ; le prix
d'exécution n'est connu qu'après l'ordre. C'est plus verbeux qu'un
`shift(-1)` en pandas, mais c'est la seule architecture qui rend une
fuite d'information future *structurellement difficile*, pas juste
"évitée si on fait attention".

## Résultat principal, pourquoi un seul backtest ne suffit jamais

Sur un premier scénario (seed fixe), le Mean-Reversion écrase le
Momentum (Sharpe +0.71 vs -0.33). Sur un **autre** scénario (seed
différent), c'est l'inverse presque exact (Momentum +66.9% vs
Mean-Reversion -49.4%). Plutôt que de choisir le scénario qui raconte
la plus belle histoire, ce projet fait tourner **30 scénarios
indépendants** :

| Stratégie | Sharpe moyen | Écart-type | Meilleure sur |
|---|---|---|---|
| Momentum | -0.111 | 0.388 | 14 / 30 scénarios |
| Mean-Reversion | +0.009 | 0.336 | 16 / 30 scénarios |

**Aucune des deux stratégies ne domine universellement** et c'est le
résultat correct, pas un défaut du moteur : chaque stratégie a été
conçue pour gagner dans un régime de marché précis (tendance pour l'une,
range pour l'autre) et perdre dans l'autre. Un backtest sur un seul
chemin de prix aurait pu, par hasard, "prouver" la supériorité de
n'importe laquelle des deux, c'est exactement le piège que l'analyse
multi-scénarios est censée éviter.

## Architecture

```
event.rs         # Bar, Direction, Event (Market/Signal/Order/Fill)
data.rs           # Prix simulés à changement de régime (tendance haussière/baissière/range)
strategy.rs        # Trait Strategy + MovingAverageCrossover + MeanReversion
portfolio.rs         # Signal -> Order (dimensionnement), Fill -> mise à jour cash/position
execution.rs           # Order -> Fill (slippage + commission)
metrics.rs               # Sharpe, drawdown maximal, rendement annualisé
backtest.rs                # Boucle événementielle (VecDeque<Event>)
main.rs                       # Orchestration, rapport, robustesse multi-seeds, export CSV
python/plot_results.py           # Courbes d'équité et drawdowns (à partir du CSV Rust)
```

## Compilation & usage

```bash
cargo build --release
./target/release/tier2_backtester      # affiche le rapport, exporte equity_curves.csv

python plot_results.py                  # transforme le CSV en equity_curves.png et drawdowns.png
```

## Frictions de marché modélisées

- **Slippage** (2 bps par défaut) : le prix d'exécution est
  systématiquement décalé défavorablement par rapport au sens de
  l'ordre.
- **Commission** (1 bp par défaut) : coût proportionnel au notionnel
  échangé.

Les ignorer est l'erreur la plus commune en backtesting : une
stratégie qui fait beaucoup d'allers-retours (Mean-Reversion, 208
transactions dans l'exemple détaillé) peut sembler très rentable sans
frictions et nettement moins avec, ce moteur les intègre par défaut,
pas en option qu'on oublierait d'activer.

## Limites connues

- Position sizing par fraction fixe du capital (95%), pas de
  vol-targeting ni de dimensionnement à la Kelly, qui ajusteraient la
  taille de position à la volatilité courante plutôt qu'à un
  pourcentage constant.
- Exécution au prix du jour même (ajusté du slippage), pas au prix
  d'ouverture du jour suivant, simplification standard pour un
  premier moteur, mais un backtest de production distinguerait le
  moment du signal (clôture t) du moment d'exécution (ouverture t+1).
- Un seul actif, deux stratégies simples, le moteur est conçu pour
  être étendu (portefeuille multi-actifs, stratégies plus élaborées)
  sans changer l'architecture événementielle centrale.
- Pas de test de significativité formel sur les 30 scénarios (les
  écarts-types donnent une idée de la dispersion, mais un test
  statistique en bonne et due forme comparerait les distributions de
  Sharpe entre les deux stratégies).

## Prochaine étape (Tier 2 #12)

Pricer de produit structuré autocall (Python + VBA), pour prolonger
le lien avec le livre de Guliano de la liste de référence et
diversifier vers les produits structurés, avec un volet VBA/Excel très
demandé en banque de financement.
