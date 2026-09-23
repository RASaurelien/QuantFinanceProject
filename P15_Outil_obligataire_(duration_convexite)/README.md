# Outil Obligataire, Prix, Duration, Convexité (VBA + Python)

Fonctions Excel/VBA pour la valorisation obligataire (prix, duration
de Macaulay, duration modifiée, convexité), avec les formules
validées séparément en Python contre des cas connus en forme fermée.

## Pourquoi ce projet

VBA reste très présent dans les entretiens M&A/corporate finance et
dans beaucoup d'outils front-office de desks obligataires, un projet
"utilitaire" directement exploitable dans un classeur Excel est un bon
complément aux projets de recherche quantitative du dépôt.

## Validation (Python, formules identiques ligne à ligne)

Le VBA n'a pas pu être exécuté dans cet environnement (pas d'Excel),
les mêmes formules ont donc été ré-implémentées en Python pour être
testées contre trois cas de référence :

| Test | Résultat | Attendu | Verdict |
|---|---|---|---|
| Zéro-coupon 10 ans : duration de Macaulay | 10.000000 | = maturité (exact) | OK |
| Obligation au pair (coupon = ytm) | prix = 100.000000 | = 100 (exact) | OK |
| Approximation duration+convexité vs repricing exact (+100 bps) | erreur 0.012 pt | < erreur duration seule (0.34 pt) | Convexité réduit l'erreur de 96.6% |

**Important** : cette validation porte sur les formules (identiques
dans les deux langages), pas sur l'exécution VBA elle-même, à
retester directement dans Excel avant tout usage en entretien.

## Structure

```
vba/BondPricer.bas     # BondPrice, MacaulayDuration, ModifiedDuration,
                          BondConvexity, PriceChangeApprox, InterpolateYield
python/validate.py      # Mêmes formules en Python, testées contre 3 cas connus
```

## Usage

**Excel** : `Alt+F11` → Importer le fichier `.bas` → utiliser les
fonctions directement en cellule, ex. `=BondPrice(100, 0.05, 0.04, 2, 10)`.

**Validation** : `python python/validate.py`

## Limites connues

- VBA non exécuté dans cet environnement (voir ci-dessus), formules
  validées, pas le fichier `.bas` lui-même.
- Courbe de taux plate implicite pour le pricing (un seul `ytm`), pas
  d'actualisation par une vraie courbe zéro-coupon terme par terme.
- Pas de day-count convention réelle (Act/360, 30/360...), maturité
  traitée en années décimales simples.