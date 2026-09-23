# Décroissance de l'Information Coefficient (IC Decay)

Combien de temps un signal prédictif garde-t-il son pouvoir de
prévision ? Question centrale de tout desk de recherche avant de
déployer un signal troisième et dernier projet "researcher" du
dépôt, prolongeant la question posée dans le projet régime de
volatilité (Tier 2 #8) : *"le signal détecte-t-il un état, ou
anticipe-t-il un changement ?"*, reformulée ici comme une courbe de
décroissance mesurable.

## Définition

```
IC(h) = corr_rang( S_t , R_{t, t+h} )
```

Corrélation de rang (Spearman, robuste aux outliers) entre le signal
au temps `t` et le rendement réalisé sur l'horizon `h`. Un bon signal
quant a typiquement un IC autour de 0.05-0.15 à court horizon, et
décroît avec `h` à mesure que l'information s'incorpore dans les prix
ou devient obsolète.

## Résultat (signal simulé, IC vrai=0.12 à h=1, demi-vie vraie=12j)

| Quantité | Vraie valeur | Estimée | Écart |
|---|---|---|---|
| IC à h=1 | 0.120 | 0.120 | ~0% |
| Demi-vie | 12.0 jours | 10.57 jours | 11.9% |

Le contrôle négatif (signal pur bruit, IC vrai=0 à tout horizon) donne
un IC moyen de -0.0004, confirme que l'estimateur ne détecte pas de
signal fantôme là où il n'y en a pas.

**Implication pratique** : à horizon = demi-vie, l'IC est retombé à
50% de sa valeur initiale, c'est l'ordre de grandeur de la fenêtre de
rebalancement à considérer : plus lent, on laisse une grande partie de
l'edge inexploitée ; plus rapide que nécessaire, les coûts de
transaction dominent sans gain d'information proportionnel.

## Structure

```
src/main.py    # Simulation signal+rendements à décroissance connue, IC empirique, ajustement, graphique
outputs/          # Graphique (committé)
```

## Usage

```bash
cd src && python main.py
```

## Limites connues

- Décroissance exponentielle imposée par construction dans la
  simulation, un vrai signal peut décroître différemment (par
  paliers, avec un rebond, etc.) ; l'ajustement exponentiel est un
  point de départ standard, pas une loi universelle.
- Un seul régime testé (décroissance stationnaire), en pratique, la
  demi-vie d'un signal peut elle-même varier dans le temps
  (non-stationnarité de l'edge), ce qui nécessiterait une analyse
  glissante plutôt qu'un ajustement unique sur tout l'historique.