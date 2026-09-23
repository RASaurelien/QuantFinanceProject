# Pairs Trading Mean-Reversion, Scan d'Univers et Gestion du Risque (Python)

Contrairement au mean-reversion générique du backtester événementiel
(Tier 2 #11, une seule paire donnée à l'avance), ce projet est pensé
**comme un trader le construirait** : scan de cointégration sur tout
un univers d'actifs, sélection des meilleures paires, puis backtest
avec une vraie gestion du risque par position (stop-loss, sizing par
volatilité du spread), comparée explicitement à une version naïve
pour juger si elle apporte réellement quelque chose.

## Le scanner fonctionne : validation nette

Univers de 20 actifs, dont 3 vraies paires cointégrées noyées parmi
190 paires possibles. Le scanner (Engle-Granger, classé par t-stat
ADF) retrouve les **3/3 vraies paires dans le top 5** :

| Paire | β | t-stat ADF | Vraie paire ? |
|---|---|---|---|
| asset_0-asset_1 | 1.043 | -9.13 | OUI |
| asset_2-asset_3 | 0.798 | -8.93 | OUI |
| asset_4-asset_5 | 1.105 | -8.35 | OUI |
| asset_7-asset_9 | 0.138 | -4.37 | non (fausse détection) |
| asset_7-asset_14 | 0.109 | -4.16 | non (fausse détection) |

Les 3 vraies paires dominent clairement le classement (t-stat autour
de -9 vs -4 pour les fausses détections), le scanner fait son travail.

## La gestion du risque : un résultat honnête, pas flatteur

| | Sharpe | Max Drawdown |
|---|---|---|
| Sans gestion du risque | 0.084 | 8.72 |
| Avec gestion du risque (stop-loss + sizing par vol) | -0.280 | 18.09 |

**La gestion du risque n'améliore PAS la performance agrégée dans ce
backtest précis** et c'est un résultat défendable, pas un bug :

1. **Le stop-loss coupe des reversions qui finissent par se produire.**
   Dans cette simulation, la relation de cointégration ne se casse
   **jamais** réellement (le bruit idiosyncratique reste stationnaire
   tout du long), donc chaque déclenchement de stop-loss est, par
   construction de ce backtest, une perte pure sans contrepartie : le
   stop protège contre un scénario (rupture de la cointégration) qui
   ne s'est simplement pas produit ici. Sa valeur d'assurance
   n'apparaît que dans les scénarios où la rupture a lieu, ce que ce
   backtest ne teste pas (voir Limites).
2. **Le sizing inverse-vol amplifie l'exposition en période de faible
   volatilité du spread**, ce qui peut aggraver une perte autant qu'il
   peut l'atténuer selon l'issue réalisée, la normalisation du risque
   ne rend pas une stratégie perdante gagnante, elle change juste son
   profil de risque.

C'est une tension réelle et documentée dans la littérature sur le
mean-reversion : contrairement au trend-following (où un stop-loss est
généralement bénéfique, car il coupe des pertes qui continuent), un
stop-loss sur une stratégie de retour à la moyenne coupe parfois
précisément la trade qui allait devenir gagnante.

## Structure

```
src/
  universe.py    # 20 actifs, 3 vraies paires cointégrées noyées dans l'univers
  scanner.py       # Scan Engle-Granger sur toutes les paires, classement par t-stat ADF
  strategy.py        # Entrée/sortie z-score, stop-loss, sizing inverse-vol avec plafond de levier
  main.py               # Scan, sélection, backtest avec/sans gestion du risque, graphique
outputs/                  # Graphique (committé)
```

## Usage

```bash
cd src && python main.py
```

## Limites connues

- **La simulation ne teste jamais une vraie rupture de cointégration**
  pour évaluer honnêtement la valeur du stop-loss, il faudrait
  injecter un scénario où une paire cesse d'être cointégrée en cours
  de route (dérive permanente du spread) et vérifier que le stop
  limite bien les dégâts dans CE cas précis, même s'il coûte en
  moyenne sur les cas où la relation reste stable. C'est l'extension
  la plus importante à faire avant de tirer une conclusion générale.
- Deux fausses détections dans le top 5 (paires non cointégrées mais
  avec un t-stat ADF qui franchit le seuil par hasard), sur 190 paires
  testées, c'est un taux de faux positifs attendu sans correction pour
  tests multiples (Bonferroni ou Benjamini-Hochberg), volontairement
  omise ici pour rester lisible.
- Coûts de transaction non modélisés (contrairement au moteur de
  backtesting événementiel, Tier 2 #11), les ajouter pénaliserait
  davantage la version avec stop-loss (plus de rotations).