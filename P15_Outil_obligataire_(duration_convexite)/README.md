# Outil Obligataire — Prix, Duration, Convexité (VBA + Excel)

Fonctions Excel/VBA pour la valorisation obligataire (prix, duration de
Macaulay, duration modifiée, convexité, approximation du choc de prix,
interpolation de courbe), exécutables depuis un classeur Excel via un
bouton **« Lancer le pricing »**.

## Pourquoi ce projet

VBA reste très présent dans les entretiens M&A/corporate finance et dans
beaucoup d'outils front-office de desks obligataires : un utilitaire
directement exploitable dans un classeur Excel est un bon complément aux
projets de recherche / trading quantitative du dépôt.

## Structure

```
vba/BondPricer.bas        # Fonctions (BondPrice, MacaulayDuration, ModifiedDuration,
                             BondConvexity, PriceChangeApprox, InterpolateYield)
                             + macro RunBondPricing (bouton)
excel/bond_pricer.xlsx    # Classeur : feuille Inputs, feuille Results, bouton
python/validate.py        # Mêmes formules en Python, testées contre 3 cas connus
```

## Le classeur `bond_pricer.xlsx`

| Feuille | Contenu |
|---|---|
| `Inputs` | Paramètres modifiables (cellules jaunes) : nominal, coupon, YTM, fréquence, maturité, choc de taux, courbe de taux et maturité cible à interpoler. |
| `Results` | Remplie par la macro : prix, duration de Macaulay, duration modifiée, convexité, variation de prix approchée vs exacte, erreur, taux interpolé, et table de sensibilité (chocs de −200 à +200 bps). Le bouton **« Lancer le pricing »** s'y trouve. |

Le classeur est livré **sans code** (format `.xlsx`) : le VBA vient du fichier
`BondPricer.bas`, à importer une fois (étapes ci-dessous).

## Exécuter le VBA depuis Excel

1. Ouvrir `bond_pricer.xlsx` dans Excel.
2. `Alt+F11` (éditeur VBA) → clic droit sur le projet → **Importer un fichier...** → choisir `BondPricer.bas`.
3. Aller sur la feuille `Results` et cliquer sur **Lancer le pricing**.
   Si le bouton n'apparaît pas : `Développeur` → `Insérer` → `Bouton (contrôle de formulaire)`, le dessiner, puis lui affecter `RunBondPricing`.
4. Modifier les cellules jaunes de `Inputs`, recliquer sur le bouton : `Results` se met à jour.

Les fonctions s'utilisent aussi directement en cellule une fois le module
importé, par ex. `=BondPrice(100;0,05;0,04;2;10)` (séparateur `;` avec Excel
en français, `,` en anglais).

## Validation (Python)

Les formules sont validées séparément en Python (`python python/validate.py`)
contre trois cas connus en forme fermée :

| Test | Résultat | Attendu | Verdict |
|---|---|---|---|
| Zéro-coupon 10 ans : duration de Macaulay | 10.000000 | = maturité (exact) | OK |
| Obligation au pair (coupon = ytm) | prix = 100.000000 | = 100 (exact) | OK |
| Approximation duration+convexité vs repricing exact (+100 bps) | erreur 0.012 pt | < erreur duration seule (0.34 pt) | Convexité réduit l'erreur de 96.6% |

**Important** : cette validation porte sur les formules (identiques dans les
deux langages), pas sur l'exécution VBA elle-même. Le `.bas` et la macro
`RunBondPricing` n'ont pas été exécutés dans l'environnement de développement
(pas d'Excel) : à tester directement dans Excel. Avec les paramètres par
défaut de `Inputs` (coupon 5 %, YTM 4 %, 10 ans, semestriel), le prix attendu
est 108,1757.

## Limites connues

- VBA non exécuté dans l'environnement de développement (voir ci-dessus).
- Courbe de taux plate implicite pour le pricing (un seul `ytm`) — pas
  d'actualisation par une vraie courbe zéro-coupon terme par terme.
- Pas de day-count convention réelle (Act/360, 30/360...) — maturité
  traitée en années décimales simples.
- `InterpolateYield` suppose une courbe triée par maturité croissante.

## Complexité
O(n) 
Formule fermée sur n flux de trésorerie.