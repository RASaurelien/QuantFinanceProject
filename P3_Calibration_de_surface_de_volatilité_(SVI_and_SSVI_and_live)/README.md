# Calibration de Surface de Volatilité, SVI (Python)

Pipeline complet : cotations d'options (prix) → volatilité implicite
(inversion Black-Scholes) → calibration d'un smile paramétrique SVI par
maturité → vérification de non-arbitrage → surface 3D.

Suite naturelle des projets 1 (pricing) et 2 (EDP américaine) : après
avoir su *pricer avec une vol donnée*, ce projet retrouve *la vol que le
marché implique réellement*.

## Pourquoi ce projet

C'est l'un des exercices les plus demandés en entretien quant desk /
quant research, parce qu'il combine plusieurs briques :
- inversion numérique robuste (Black-Scholes n'a pas de formule fermée
  pour sigma → root-finding),
- calibration non-linéaire sous contraintes (least squares + bornes),
- compréhension économique du résultat (un smile mal calibré peut
  introduire de l'arbitrage pas juste "un mauvais fit").

## Pipeline

```
prix marché (bruités)
     │  implied_vol()  [Brent, robuste aux cotations proches des bornes d'arbitrage]
     ▼
volatilité implicite marché, par strike et maturité
     │  calibrate_svi_slice()  [least_squares, 5 paramètres, bornes économiques]
     ▼
SVI raw calibré par maturité :  w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
     │
     ▼
smile lisse + surface 3D + vérification calendar arbitrage
```

## Données : réelles si possible, synthétiques sinon (avec bascule automatique)

- `--ticker AAPL` : tente une vraie chaîne d'options via `yfinance`
  (Yahoo Finance). Fonctionne sur un poste avec accès réseau standard.
- Sans `--ticker`, ou si la récupération réseau échoue : bascule
  automatiquement sur un jeu de données **synthétique mais réaliste**,
  généré par un aller-retour complet `vol vraie → prix bruité → vol
  implicite recalculée`. Ce round-trip est volontairement le même
  chemin qu'un pipeline de marché réel, ce qui permet de valider
  honnêtement le calibrateur : si le SVI recalibré retrouve les
  paramètres "vrais" à quelques points de base près malgré le bruit de
  cotation, le pipeline est correct de bout en bout.

*(Dans l'environnement où ce projet a été développé, les hôtes Yahoo
Finance n'étaient pas joignables le fallback synthétique a donc servi
de validation par défaut. Sur ta machine, `--ticker` devrait fonctionner
directement.)*

## Structure

```
src/
  black_scholes.py   # Pricing BS + implied_vol() (inversion par Brent)
  svi.py             # Paramétrisation SVI raw + calibration + checks no-arbitrage
  data_source.py      # Chaîne d'options réelle (yfinance) ou synthétique
  main.py             # Pipeline CLI : calibration, plots, export CSV
requirements.txt
outputs/               # Généré à l'exécution (plots + CSV)
```

## Usage

```bash
pip install -r requirements.txt

python src/main.py                    # données synthétiques
python src/main.py --ticker AAPL      # données réelles si le réseau le permet
```

Génère dans `outputs/` :
- `smile_by_maturity.png` un smile par maturité, points de marché vs SVI calibré
- `vol_surface_3d.png` surface de volatilité 3D complète
- `svi_params.csv` table des 5 paramètres calibrés par maturité

## Résultats obtenus (mode synthétique, 5 maturités de 1 mois à 2 ans)

- RMSE de calibration entre 0.02 % et 0.06 % de vol implicite selon la
  maturité, le SVI retrouve quasiment exactement les paramètres
  "vrais" utilisés pour générer les données, malgré le bruit de
  cotation ajouté (~25 bps).
- Contrainte butterfly (`a + b·σ·√(1-ρ²) ≥ 0`) respectée sur toutes les
  maturités : pas de variance négative implicite.
- Contrainte calendar (variance totale croissante en T) respectée sur
  toute la grille de strikes testée.

## Limites connues

- Chaque maturité est calibrée **indépendamment** : la vérification
  calendar est faite *a posteriori*, pas imposée pendant l'optimisation.
  Une extension naturelle est une calibration jointe multi-maturités
  avec la contrainte calendar explicitement dans l'optimiseur (SLSQP
  avec contraintes d'inégalité plutôt que least_squares).
- SVI raw ne modélise pas la dynamique du smile dans le temps (comment
  il évolue si le spot bouge), seulement sa forme statique à un
  instant donné. Pour une modélisation dynamique, voir SABR ou un
  modèle à volatilité locale (Dupire).
- Le taux sans risque et le dividende sont supposés constants et plats
  par maturité, simplification raisonnable ici, mais une vraie
  desk utiliserait une courbe de taux et un calendrier de dividendes.