# Marché Multi-Agents, Percolation et Faits Stylisés (Cont-Bouchaud)

Modèle de Cont & Bouchaud (2000) : les faits stylisés des marchés
financiers (queues épaisses, clustering de volatilité) émergent d'un
**mécanisme purement structurel** la formation de clusters d'agents
sur un graphe aléatoire, près du seuil de percolation, sans aucune
hypothèse d'efficience ou de rationalité individuelle. Deuxième projet
du dépôt exploitant le point fort "physique statistique" du profil
(après le débruitage RMT, Tier 2 #9), sur un terrain différent : la
physique des transitions de phase plutôt que la théorie des matrices
aléatoires.

## Mécanique du modèle

1. **N agents** reliés deux à deux avec probabilité `p` (graphe
   d'Erdős-Rényi), calibrée près du **seuil de percolation**
   (degré moyen `<k> = p(N-1) ≈ 1`), le point critique où la taille
   des clusters (composantes connexes) suit une loi de puissance.
2. Chaque cluster agit comme **un seul trader collectif** : achète
   (+1), vend (-1), ou reste inactif, avec la même décision pour tous
   ses membres (herding).
3. Rendement du marché = demande nette agrégée / N.
4. Le degré moyen oscille lentement autour du seuil critique (phases
   plus ou moins proches de la criticalité), ce qui génère le
   clustering de volatilité **sans hypothèse GARCH imposée**, juste
   la proximité variable au point critique.

## Résultats

| Fait stylisé | Résultat mesuré | Verdict |
|---|---|---|
| Queues épaisses | Excès de kurtosis = 10.84 (0 = gaussien) | Jarque-Bera rejette la normalité (p ≈ 0) |
| Pas d'autocorrélation des rendements | ACF moyenne (retards 1-20) = 0.0000 | Cohérent avec l'absence d'arbitrage trivial |
| Clustering de volatilité | ACF moyenne des \|rendements\| = +0.099 | Positive et significative |
| Loi de puissance des clusters | Exposant mesuré -1.30 (théorie champ moyen : -2.5) | Signature qualitative confirmée (R²=0.70) |

Les trois premiers résultats reproduisent exactement les signatures
empiriques attendues. Le dernier confirme le **mécanisme** (une vraie
loi de puissance, pas une coïncidence), même si l'exposant précis
diffère de la valeur théorique asymptotique, voir limites ci-dessous.

## Structure

```
src/
  agent_model.py    # Union-Find pour les clusters de percolation + simulation du marché
  main.py             # Faits stylisés (kurtosis, ACF), loi de puissance, graphiques
outputs/                # Graphique (committé : preuve visuelle pour le README)
```

## Usage

```bash
cd src && python main.py    # ~10 secondes (2000 agents x 2000 pas de temps)
```

## Limites connues

- L'exposant de loi de puissance mesuré (-1.30) diffère de la valeur
  théorique en champ moyen (-2.5) : effet de taille finie (N=5000,
  une seule réalisation du graphe aléatoire), la convergence vers
  l'exposant asymptotique demanderait N beaucoup plus grand et un
  moyennage sur de nombreuses réalisations, coûteux en temps de calcul.
- Le lien "taille de cluster -> décision de trading" est simplifié
  (probabilité fixe, pas de dépendance à la richesse ou à l'aversion
  au risque de l'agent), le modèle original de Cont-Bouchaud et ses
  extensions (Lux-Marchesi) ajoutent une hétérogénéité comportementale
  plus riche.
- Impact de prix linéaire (rendement proportionnel à la demande nette),
  une hypothèse simplificatrice standard à ce niveau de modèle, pas
  calibrée sur une vraie fonction d'impact de marché.