# Recherche Factorielle Cross-Sectionnelle, Fama-MacBeth (Python)

Régressions de Fama & MacBeth (1973) à deux étapes pour estimer les
primes de risque de 4 facteurs de style (Value, Momentum, Quality,
Low-Vol) sur un panel simulé, avec erreurs-types corrigées **Newey-West**
et un facteur **placebo** (prime vraie = 0) comme contrôle négatif.
Premier projet du dépôt orienté **recherche empirique** plutôt que
pricing ou infrastructure, format le plus proche d'un vrai papier de
recherche factorielle (asset pricing empirique).

## Méthodologie

**Étape 1** (répétée chaque mois `t`) : régression cross-sectionnelle
OLS des rendements du mois sur les expositions factorielles des
actifs → une prime de facteur estimée `λ_{k,t}` par mois.

**Étape 2** : la prime de facteur retenue est la moyenne de la série
`λ_{k,t}` dans le temps. L'erreur-type ne peut pas supposer
l'indépendance temporelle des `λ_{k,t}`, d'où la correction
**Newey-West** (HAC), implémentée à la main (noyau de Bartlett, règle
de troncature de Newey-West 1994).

## Résultat (20 ans simulés, 500 actifs, contrôle négatif inclus)

| Facteur | Prime vraie | Prime estimée | t-stat (NW) | Verdict |
|---|---|---|---|---|
| Value | 0.300% | -0.016% | -0.12 | non significatif |
| **Momentum** | 0.500% | 0.549% | **4.36** | **significatif** |
| Quality | 0.200% | 0.191% | 1.49 | non significatif |
| LowVol | 0.150% | 0.193% | 1.40 | non significatif |
| **Placebo** | 0.000% | 0.099% | 0.78 | non significatif (correct) |

Seul Momentum (la plus grosse prime vraie) atteint la significativité
sur 20 ans, et le placebo est correctement rejeté. **Ce n'est pas un
échec de la méthode** : c'est un phénomène bien documenté en finance
empirique, détecter une prime de quelques dizaines de points de base
par mois avec confiance nécessite énormément de données, même quand la
prime existe réellement.

## Vérification de cohérence asymptotique

Pour distinguer "la méthode est mal implémentée" de "l'échantillon de
20 ans manque de puissance", un stress-test pur (T=1200 mois, 100 ans,
pas un scénario réaliste) montre Value, Momentum et Quality devenir
significatifs avec des estimations qui se rapprochent des vraies
valeurs, confirmant que la méthode elle-même n'est pas biaisée.
LowVol (plus petite prime vraie) reste non significatif même à
T=1200 sur cette réalisation précise : rappel que même un échantillon
énorme n'élimine pas complètement le bruit pour un effet de petite
taille sur un seul tirage.

## Structure

```
src/
  simulate.py       # Panel simulé, 4 facteurs de style + 1 placebo, primes vraies connues
  fama_macbeth.py     # Régressions cross-sectionnelles + agrégation Newey-West
  main.py               # Pipeline complet, validation, vérification asymptotique, graphiques
outputs/                  # Graphique (committé)
```

## Usage

```bash
cd src && python main.py
```

## Limites connues

- Expositions factorielles fixes dans le temps (simplification), en
  pratique, les caractéristiques (B/M, momentum 12 mois...) sont
  recalculées chaque mois à partir des données brutes, avec leur
  propre bruit de mesure.
- Un seul tirage aléatoire par configuration, une étude de puissance
  statistique complète ferait tourner plusieurs centaines de
  simulations et regarderait la distribution des t-stats, pas un seul
  résultat.
- Pas de contrôle pour la corrélation entre facteurs (multicolinéarité)
  ni de neutralisation sectorielle, deux raffinements standards d'une
  vraie étude factorielle académique.