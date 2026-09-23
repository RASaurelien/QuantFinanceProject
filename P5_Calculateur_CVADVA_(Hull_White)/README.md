# Calculateur CVA/DVA, Swap de Taux (Hull-White, Python)

Calcul de la CVA (Credit Valuation Adjustment) et de la DVA (Debit
Valuation Adjustment) d'un swap de taux vanille, par simulation Monte
Carlo de l'exposition future sous un modèle de taux court Hull-White à
un facteur. Prolongement direct du travail déjà fait sur *The xVA
Challenge* de Jon Gregory.

## Pourquoi ce projet

La CVA est l'ajustement qui a le plus changé le métier de trader
dérivés depuis 2008 : le prix d'un produit n'est plus seulement sa
valeur "risque de marché", il faut aussi pricer le risque que la
contrepartie fasse défaut avant l'échéance. Ce projet implémente le
calcul de bout en bout, pas une formule toute faite, mais la chaîne
complète : modèle de taux → simulation d'exposition → intégration sur
une courbe de survie.

Peu d'étudiants en L3 ont un projet xVA fonctionnel avec un vrai moteur
de simulation de taux (plutôt qu'une exposition supposée constante),
c'est un signal fort pour du sell-side, du risque de contrepartie, ou
de la structuration.

## Pipeline

```
Hull-White (dr = (theta(t)-a*r)dt + sigma*dW)
     │  simulation Monte Carlo (antithétique), taux court à chaque date de paiement
     ▼
Prix zéro-coupon P(t,T) analytiques (formule fermée Hull-White)
     │  valorisation du swap à chaque date de paiement, sur chaque trajectoire
     ▼
Matrice de MtM simulée (n_paths x n_dates)
     │  EE(t) = E[max(MtM,0)]  |  ENE(t) = E[min(MtM,0)]  |  PFE 95%(t)
     ▼
Profil d'exposition
     │  intégration sur la courbe de survie (intensité de défaut constante, calibrée sur spread CDS)
     ▼
CVA = (1-R_c) Σ EE(t_k) P(0,t_k) PD_c(t_k)      DVA = (1-R_b) Σ |ENE(t_k)| P(0,t_k) PD_b(t_k)
```

## Résultats obtenus (swap 5Y, notional 10M, contrepartie BBB à 150 bps)

- Taux fixe "par" du swap : 3.0226 %
- EPE (exposition positive moyenne) : ~98 700
- PFE 95 % maximum sur l'horizon : ~632 000
- **CVA : 7 263** (≈ 7.26 bps du notional)
- **DVA : 2 860**
- **CVA bilatérale (CVA − DVA) : 4 403**

Ordre de grandeur cohérent avec la pratique de marché (quelques points
de base à quelques dizaines de bps de notional pour une contrepartie
BBB sur un swap 5 ans).

## Deux validations indépendantes du moteur de simulation

Au-delà de "le nombre a l'air raisonnable", deux checks vérifient que
le moteur de simulation lui-même est correct, indépendamment du calcul
xVA :

1. **Cas dégénéré (sigma → 0)** : sans aléa, un swap coté au taux
   "par" ne bouge jamais, son MtM doit rester nul à toute date, sur
   toute trajectoire. Résultat mesuré : écart maximal de 1.1×10⁻⁹ du
   notional (bruit numérique pur).

2. **Propriété de martingale** : sous la mesure risque-neutre, le MtM
   actualisé par le numéraire "compte monétaire" `B(t) = exp(∫r ds)`
   doit avoir une espérance nulle à toute date, c'est un résultat
   fondamental de la théorie du pricing sans arbitrage, indépendant du
   produit valorisé. Vérifié empiriquement : l'écart à zéro reste sous
   0.3 écart-type Monte Carlo sur toutes les dates testées.

## Structure

```
src/
  hull_white.py    # Simulation du taux court + prix zéro-coupon analytiques
  swap.py           # Valorisation du swap (jambe fixe / flottante via zéro-coupons)
  xva.py             # Profils d'exposition (EE/ENE/PFE) + calcul CVA/DVA
  main.py             # Pipeline complet, validations, graphiques
requirements.txt
outputs/               # Graphiques générés (committés : preuve visuelle pour le README)
```

## Compilation & usage

```bash
pip install -r requirements.txt
python src/main.py
```

## Limites connues

- Intensité de défaut constante (calibrée sur un seul point CDS),
  une vraie desk utiliserait une courbe de spreads CDS complète
  (bootstrap terme par terme).
- Pas de wrong-way risk modélisé (corrélation entre l'exposition et la
  probabilité de défaut de la contrepartie), extension naturelle :
  corréler le processus de hasard rate au taux court simulé.
- CVA actualisée avec une courbe déterministe P(0,t) (convention
  marché standard), alors que la simulation elle-même utilise un
  numéraire stochastique, c'est volontaire (voir la validation de
  martingale, qui teste précisément la cohérence entre les deux), mais
  une desk plus avancée intégrerait aussi le risque de taux dans le
  facteur d'actualisation du CVA lui-même (stochastic discounting).
- Un seul produit (swap) et une seule contrepartie, pas de nettting
  set multi-produits ni de collatéral (CSA), qui sont les deux
  raffinements suivants naturels d'un moteur CVA de desk.