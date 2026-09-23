# Event Study, Dérive Post-Annonce de Résultats (PEAD)

Méthodologie d'event study (MacKinlay, 1997) appliquée à la dérive
post-annonce de résultats (Post-Earnings Announcement Drift, Ball &
Brown 1968 / Bernard & Thomas 1989), l'une des anomalies les plus
robustes et les plus étudiées de la finance empirique. Deuxième projet
du dépôt orienté recherche, complémentaire à l'approche
cross-sectionnelle du projet Fama-MacBeth (R1) : ici, l'unité
d'analyse est un **événement discret**, pas une coupe transversale
périodique.

## Méthodologie

1. **Modèle de marché** estimé sur une fenêtre d'ESTIMATION de 200
   jours **avant** l'événement (jamais sur la fenêtre d'événement,
   sinon l'estimation absorberait elle-même la dérive qu'on cherche à
   mesurer) : `r_{i,t} = alpha_i + beta_i * r_{m,t} + eps_{i,t}`.
2. **Rendement anormal (AR)** sur la fenêtre d'événement (-20 à +70
   jours) : écart entre le rendement réalisé et le rendement attendu
   par le modèle de marché.
3. **CAR** (rendement anormal cumulé) : somme des AR depuis le début
   de la fenêtre.
4. **Test de Brown & Warner (1985)** : t-test cross-sectionnel sur le
   CAR moyen à un horizon donné, écart-type basé sur la dispersion
   cross-sectionnelle des CAR individuels.

## Résultat (2000 événements simulés, CAR à +60 jours)

| Groupe | N | CAR[0,+60] | t-stat | Verdict |
|---|---|---|---|---|
| Surprise positive (vraie dérive) | 689 | +2.36% | 3.23 | **significatif** |
| Surprise négative (vraie dérive) | 711 | -2.17% | -2.93 | **significatif** |
| Placebo, surprise positive | 302 | +0.75% | 0.74 | non significatif |
| Placebo, surprise négative | 298 | +0.26% | 0.24 | non significatif |

Les deux groupes à vraie dérive sont fortement significatifs, avec un
CAR qui continue de dériver **après** le jour de l'annonce (jour 0),
exactement la signature de PEAD : le marché sous-réagit à la surprise
initiale, puis le prix continue de s'ajuster pendant plusieurs
semaines. Les deux groupes placebo (même distribution de surprise,
mais AUCUNE vraie dérive injectée) restent plats et non significatifs,
contrôle négatif qui valide que la méthodologie ne "trouve" pas un
effet fantôme là où il n'y en a pas.

## Structure

```
src/
  simulate.py       # Événements simulés, fenêtre d'estimation + fenêtre d'événement, PEAD + placebo
  event_study.py      # Modèle de marché, calcul AR/CAR, test de Brown & Warner
  main.py                # Pipeline complet, tests par groupe, graphique
outputs/                   # Graphique (committé)
```

## Usage

```bash
cd src && python main.py
```

## Limites connues

- Magnitude de la dérive calibrée pour une démonstration nette
  (significative sur 2000 événements), l'ampleur réelle de PEAD dans
  la littérature est généralement plus modeste et nécessite des
  échantillons de plusieurs milliers à dizaines de milliers
  d'événements réels pour être détectée avec cette confiance (voir le
  projet R1 pour une discussion de ce compromis puissance/réalisme).
- Une seule spécification de fenêtre (200j estimation, -20/+70j
  événement), une étude de robustesse ferait varier ces choix.
- Modèle de marché à un facteur (CAPM simplifié), la littérature
  académique utilise souvent un modèle à plusieurs facteurs
  (Fama-French) pour le calcul des rendements attendus, ce qui
  changerait légèrement les AR mesurés.