# Débruitage de Matrice de Corrélation par Théorie des Matrices Aléatoires

Application de la loi de **Marchenko-Pastur** (théorie des matrices
aléatoires) pour séparer signal et bruit dans une matrice de
corrélation d'actifs estimée sur un échantillon fini, puis validation
de l'intérêt pratique du débruitage sur un portefeuille de variance
minimale. C'est le projet qui exploite directement le point fort
"physique statistique" du profil, la loi de Marchenko-Pastur vient
originellement de la physique des systèmes désordonnés, pas de la
finance.

## Pourquoi ce projet

Quand on estime une matrice de corrélation N×N à partir de seulement T
observations, avec N proche de T, une grande partie du spectre de
valeurs propres est du **bruit d'estimation pur** même si les vrais
actifs sous-jacents n'étaient corrélés d'aucune façon, on observerait
un spectre non trivial juste à cause du nombre fini d'observations. La
loi de Marchenko-Pastur donne la distribution exacte de ce bruit,
ce qui permet de le filtrer sans aucune hypothèse ad hoc sur la vraie
structure de corrélation, contrairement à un simple seuillage
(thresholding) arbitraire.

C'est un problème très concret en gestion de portefeuille : la matrice
de covariance utilisée dans Markowitz (voir Tier 1 #6) est **exactement**
le genre d'objet mal estimé que ce projet corrige.

## Résultat principal validation par portefeuille

Le test qui compte n'est pas "le spectre a l'air plus propre après
débruitage" (n'importe quelle transformation peut donner cette
impression), mais l'effet sur une décision réelle : le portefeuille de
variance minimale.

| Covariance utilisée | Variance réelle du portefeuille (×10⁻⁴) |
|---|---|
| Brute (échantillon, bruitée) | 51.34 |
| **Débruitée (RMT)** | **36.87** |
| Oracle (vraie covariance, inconnue en pratique) | 32.78 |

Le débruitage réduit la variance **réelle** du portefeuille de **28.2 %**
par rapport à la covariance brute, et referme l'écart à l'oracle de
**56.6 % à 12.5 %**. La variance réelle est mesurée avec la vraie
covariance du modèle générateur, impossible sur données de marché
réelles, mais essentiel ici pour prouver que le débruitage aide
vraiment, pas seulement en apparence.

## Validation de la détection signal/bruit

Le modèle générateur utilise exactement **5 facteurs latents**
(1 facteur marché + 4 facteurs sectoriels/style). Le filtre RMT détecte
**exactement 5 valeurs propres** au-dessus de la borne de bruit
`lambda_+`, la méthode retrouve le bon nombre de facteurs sans
qu'on le lui indique, uniquement à partir du spectre observé.

## Méthode

1. **Loi de Marchenko-Pastur** : pour une matrice de corrélation N×N
   estimée sur T observations iid (`q = N/T`), les valeurs propres
   compatibles avec du bruit pur se répartissent dans
   `[λ₋, λ₊] = [(1-√q)², (1+√q)²]`.
2. **Filtre spectral** (Laloux, Cizeau, Bouchaud & Potters, 1999) :
   les valeurs propres au-dessus de `λ₊` sont conservées (signal), les
   autres remplacées par leur moyenne commune (préserve la trace totale
   = variance totale, tout en supprimant la fausse structure introduite
   par le bruit d'estimation).
3. **Reconstruction** : `C_débruitée = V Λ_débruitée V'`, renormalisée
   pour une diagonale exactement égale à 1.
4. **Validation** : portefeuille de variance minimale (formule fermée,
   même théorème des deux fonds que Tier 1 #6) sur covariance brute,
   débruitée, et oracle, comparaison de la variance **réelle**.

## Structure

```
src/
  simulate.py      # Modèle à facteurs latents (covariance VRAIE connue)
  rmt.py             # Loi de Marchenko-Pastur, filtre de débruitage spectral
  portfolio.py        # Portefeuille de variance minimale (formule fermée) + évaluation
  main.py               # Pipeline complet, graphiques
requirements.txt
outputs/                 # Graphiques (committés : preuve visuelle pour le README)
```

## Usage

```bash
pip install -r requirements.txt
cd src && python main.py
```

## Limites connues

- Le remplacement des valeurs propres de bruit par leur moyenne est la
  méthode la plus simple (Laloux et al., 1999) ; des variantes plus
  raffinées existent (shrinkage optimal de Ledoit-Wolf, RIE Rotationally
  Invariant Estimator de Bun, Bouchaud & Potters, 2017), qui optimisent
  directement l'erreur quadratique espérée plutôt que d'appliquer une
  règle de coupure binaire.
- Rendements simulés iid (indépendants dans le temps), la loi de
  Marchenko-Pastur suppose cette hypothèse ; des rendements
  autocorrélés ou à hétéroscédasticité conditionnelle (GARCH, voir
  Tier 2 #7) biaiseraient légèrement le spectre de bruit théorique.
- Un seul régime testé (`q=0.4`), l'intérêt du débruitage croît avec
  `q` (se rapprochant de 1) et devient marginal quand `T >> N`.
