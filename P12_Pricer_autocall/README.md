# Pricer Autocall (Athena/Phoenix à mémoire), Python + VBA/Excel

Pricing Monte Carlo d'une note structurée autocall à mémoire de coupon
(le produit structuré le plus vendu en banque de détail/privée en
Europe), avec le moteur rigoureux en Python et une présentation
Excel/VBA pour le volet client, exactement la répartition d'un vrai
desk de structuration.

## Le produit (mécanique complète)

À chaque date d'observation annuelle :
1. **Coupon conditionnel avec mémoire** : si le sous-jacent est
   au-dessus de la barrière de coupon, un coupon est versé, et tout
   coupon manqué lors d'une date précédente est rattrapé en même temps.
2. **Rappel anticipé (autocall)** : si le sous-jacent est au-dessus de
   la barrière de rappel (à toute date sauf la dernière), la note est
   remboursée immédiatement à 100% du notional + coupon dû.
3. **À l'échéance, si jamais rappelée** : capital protégé si le
   sous-jacent est au-dessus de la barrière de protection ; sinon,
   perte proportionnelle à la baisse du sous-jacent.

C'est un produit **path-dependent**, son prix dépend de la trajectoire
entière du sous-jacent aux dates d'observation, pas seulement de sa
valeur finale, d'où le recours au Monte Carlo plutôt qu'à une formule
fermée.

## Validations (deux cas dégénérés, formule fermée exacte)

| Cas dégénéré | Prix Monte Carlo | Prix théorique | Écart |
|---|---|---|---|
| Barrières hors de portée → obligation zéro-coupon pure | 913.9312 | `notional·e^(-rT)` = 913.9312 | **0.000%** |
| Barrière d'autocall quasi nulle → rappel quasi certain à t=1 | 1048.0812 | `(notional+coupon)·e^(-r·t1)` = 1048.0812 | **0.000%** |

Ces deux cas réduisent le produit complexe à une obligation simple
dont le prix se calcule à la mai,— la concordance exacte valide toute
la mécanique (mémoire de coupon, autocall, actualisation) d'un coup.

## Résultat de la note de référence

Note 3 ans, autocall à 100%, coupon/protection à 70%, coupon 8%/an,
S0=100, r=3%, q=2%, σ=22% :

- **Prix : 100.01% du notional** (quasi au pair, cohérent avec un
  produit structuré pour être vendu à 100% à l'émission)
- **Probabilité de rappel anticipé : 59.3%** (avant l'échéance)
- **Probabilité de perte en capital : 17.1%**
- **Durée de vie moyenne attendue : 1.93 ans** (sur une maturité
  maximale de 3 ans, la plupart des scénarios rappellent avant terme)

## Une subtilité de Greeks qui vaut la peine d'être racontée

Un premier calcul naïf du delta (bumper `S0` en gardant les barrières
en `% de S0`) donne **exactement zéro**, pas un bug, une vraie
propriété mathématique : puisque toutes les barrières ET le payoff
sont définis en proportion de `S0`, le prix est homogène de degré 0 en
`S0` (l'invariance d'échelle est automatique). Mais ce n'est pas le
delta qui intéresse une desk : les barrières d'une note déjà émise
sont **fixées en absolu** à l'émission, alors que le spot bouge
librement ensuite. Le pricer distingue donc explicitement le niveau de
référence (`S0`, fige les barrières) du spot courant (bumpé pour le
calcul du delta), delta obtenu : **2.9975**, économiquement
significatif.

## Structure

```
python/
  pricer.py          # AutocallNote, simulation, price_autocall(), compute_delta()
  main.py              # Validations, pricing de référence, graphiques
excel/
  autocall_pricer.xlsx   # Inputs (modifiable) + Payoff Diagram (formules + graphique) + Pricing Summary
vba/
  AutocallPricer.bas       # Pricer Monte Carlo en VBA, importable dans le classeur
outputs/                     # Graphiques (committés : preuve visuelle pour le README)
requirements.txt
```

## Usage

**Python (moteur de référence, exécuté et validé) :**
```bash
pip install -r requirements.txt
cd python && python main.py
```

**Excel :**
Ouvrir `excel/autocall_pricer.xlsx`, feuille *Inputs* pour modifier
les paramètres (cellules jaunes), feuille *Payoff Diagram* pour le
diagramme de payoff (100% formules, se met à jour automatiquement),
feuille *Pricing Summary* pour les résultats de référence du moteur
Python.

**VBA (à tester sur ta machine) :**
Dans l'éditeur VBA d'Excel (Alt+F11) : clic droit sur le projet →
*Importer un fichier* → `AutocallPricer.bas`, puis créer un bouton
(onglet Développeur → Insérer → Bouton) sur la feuille *Pricing
Summary* assigné à la macro `RunPricing`.

## Limites connues

- Un seul sous-jacent, les autocalls "worst-of" (sur le pire de
  plusieurs sous-jacents, la structure la plus commercialisée en
  pratique) demanderaient une simulation multivariée corrélée.
- Volatilité et taux constants, un vrai desk utiliserait une surface
  de volatilité complète (voir Tier 1 #3) et une courbe de taux, pas
  des valeurs plates.
- Le VBA utilise Box-Muller "brut" (pas de variates antithétiques ni
  de Common Random Numbers), plus lent à converger que la version
  Python à nombre de trajectoires égal, compromis assumé pour rester
  lisible en macro Excel.
- Delta uniquement, pas de vega, de gamma, ni d'analyse de wrong-way
  risk sur la corrélation défaut/sous-jacent (voir Tier 1 #5 pour ce
  type d'analyse côté CVA).
