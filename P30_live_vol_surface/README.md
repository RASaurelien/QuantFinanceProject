# Surface de volatilité live — SVI + SSVI (Python)

Un seul script, `live_vol_surface.py` : récupère une chaîne d'options réelle
(Yahoo Finance), recalcule la volatilité implicite à partir des prix, calibre
un smile SVI par échéance et une surface SSVI globale, vérifie l'absence
d'arbitrage, puis affiche soit une fenêtre 3D rafraîchie en continu, soit un
rapport statique (graphiques + CSV + pricing Dupire/Monte Carlo).

Aucune dépendance à Interactive Brokers : conçu pour tourner tel quel sur
n'importe quel poste avec un accès réseau standard, sans configuration.

## Pourquoi ce projet

Suite naturelle d'un pipeline de pricing classique : après avoir su *pricer
avec une vol donnée*, ce projet retrouve *la vol que le marché implique
réellement*, et va jusqu'à en tirer une surface de vol locale utilisable en
simulation. Il combine :
- inversion numérique robuste de Black (Brent, pas de Newton — le marché ne
  donne jamais directement une vol, seulement des prix) ;
- calibration non-linéaire sous contraintes (SVI par tranche, puis SSVI
  globale avec pénalité de non-arbitrage butterfly intégrée à l'optimisation) ;
- vol locale de Dupire dérivée de la SSVI et simulation Monte Carlo
  (Euler-Maruyama, variates antithétiques) pour valider le pricing.

## Pipeline

```
chaîne d'options Yahoo Finance (bid/ask/last, par échéance et strike)
     │  implied_vol()  [Black-76, inversion par Brent]
     ▼
volatilité implicite marché, par strike (log-moneyness k) et maturité T
     │  fit_svi()            │  fit_ssvi()
     ▼                        ▼
SVI par échéance          SSVI globale (toutes maturités calibrées ensemble,
(smile local, 5 params)   contrainte de non-arbitrage butterfly pénalisée)
     │                        │
     └──────────┬─────────────┘
                ▼
     surface 3D + smiles + vérif calendar/butterfly
                │
                ▼  dupire_local_vol() (différences finies sur w_SSVI)
     vol locale de Dupire
                │
                ▼  simulate_paths() (Euler-Maruyama, antithétique)
     Monte Carlo : prix d'options européennes et asiatiques
```

## Données : Yahoo Finance, avec bascule synthétique automatique

- Par défaut (`--symbol SPY`) : récupère spot, huit échéances réparties
  (~1 semaine à 1 an) et les cotations hors de la monnaie via `yfinance`.
  Les spreads bid/ask trop larges, les échéances à moins de 5 jours et les
  IV recalculées aberrantes (hors [3 %, 200 %]) sont écartés.
- `--demo` : bascule sur un jeu **synthétique mais réaliste**, généré par un
  aller-retour complet SVI vrai → prix bruités (Black) → IV recalculée — le
  même chemin qu'un vrai pipeline de marché, utile pour valider le
  calibrateur hors connexion ou vérifier qu'il retrouve les paramètres
  "vrais" malgré le bruit.
- En mode `--report` sans `--demo` : si Yahoo échoue (réseau, marché fermé,
  ticker sans options), le script bascule automatiquement sur le jeu
  synthétique plutôt que de s'arrêter.

## Usage

```bash
python live_vol_surface.py                     # fenêtre 3D live, SPY, Yahoo
python live_vol_surface.py --symbol AAPL --interval 60
python live_vol_surface.py --demo               # fenêtre 3D live, données synthétiques

python live_vol_surface.py --report              # rapport statique, Yahoo (bascule sync auto)
python live_vol_surface.py --report --demo        # rapport statique, données synthétiques
python live_vol_surface.py --report --T-mc 1.0 --sims 50000
```

Les dépendances manquantes (`numpy`, `scipy`, `matplotlib`, `yfinance`)
s'installent automatiquement au premier lancement.

### Options

| Option | Défaut | Effet |
|---|---|---|
| `--symbol` | `SPY` | Ticker Yahoo Finance |
| `--interval` | `30` | Secondes entre deux rafraîchissements (mode live) |
| `--rate` | `0.04` | Taux sans risque annuel (forward, pricing) |
| `--div` | `0.0` | Rendement du dividende annuel |
| `--demo` | — | Données synthétiques, hors-ligne |
| `--report` | — | Rapport statique au lieu de la fenêtre live |
| `--outdir` | `outputs` | Dossier d'export du rapport |
| `--T-mc` | `0.5` | Maturité (années) du pricing Monte Carlo |
| `--sims` | `20000` | Nombre de trajectoires Monte Carlo |

## Mode live (fenêtre interactive)

Surface SSVI 3D (points marché + courbes SVI par échéance superposées) à
gauche, smile détaillé de l'échéance sélectionnée à droite. Rafraîchie toutes
les `--interval` secondes.
- **LOCK UPDATES** : fige le rafraîchissement pour manipuler la vue 3D.
- **NEXT EXPIRY** : fait défiler l'échéance affichée dans le panneau de droite.
- Le titre du graphique 3D indique en direct si la SSVI calibrée respecte la
  condition suffisante de non-arbitrage butterfly (Gatheral & Jacquier 2014,
  Thm 4.2).

## Mode rapport (`--report`)

Génère dans `--outdir` :
- `smile_by_maturity.png` — un smile par échéance, marché vs SVI vs SSVI
- `vol_surface_3d.png` — surface SSVI 3D + cotations marché
- `svi_params.csv` — les 5 paramètres SVI par échéance, RMSE, condition butterfly
- `ssvi_params.csv` — les 3 paramètres SSVI globaux et le résultat du test de non-arbitrage

Affiche aussi en console :
- la calibration SVI par échéance (RMSE, condition butterfly) ;
- la vérification calendar (variance totale croissante en T) pour le SVI par
  tranche et pour la SSVI ;
- la vol locale de Dupire à quelques points, puis des prix Monte Carlo
  (calls européens à 0,9F / F / 1,1F, un call asiatique) comparés au prix
  Black obtenu avec l'IV SSVI — les deux doivent coller, ce qui valide la
  cohérence du pipeline vol implicite → vol locale → simulation.

## Structure

Un seul fichier :

```
live_vol_surface.py
  1. Black-76 + implied_vol()          — pricing et inversion de l'IV
  2. SVI (par échéance)                — Gatheral 2004
  3. SSVI (globale)                    — Gatheral & Jacquier 2014
  4. fetch_yahoo() / fetch_demo()      — sources de données
  5. redraw() / main()                 — fenêtre live matplotlib
  6. dupire_local_vol() / simulate_paths() / mc_call()
  7. report()                          — rapport statique (PNG/CSV/MC)
```

## Limites connues

- Yahoo est différé (~15 min) : le mode « live » est un rafraîchissement
  périodique, pas un flux temps réel.
- Le taux sans risque et le dividende sont constants et plats
  (`--rate`, `--div`) — pas de courbe de taux ni de calendrier de dividendes.
- La condition de non-arbitrage butterfly vérifiée pour la SSVI est
  *suffisante* mais pas nécessaire (Thm 4.2 de Gatheral & Jacquier) : une
  SSVI qui échoue au test n'est pas forcément arbitrable, mais une qui le
  passe est garantie sans arbitrage.
- Le calendar check du SVI par tranche compare des paramétrisations
  calibrées indépendamment — c'est une vérification *a posteriori*, pas une
  contrainte imposée à l'optimiseur (contrairement à la SSVI, où la pénalité
  butterfly est intégrée à la calibration).
