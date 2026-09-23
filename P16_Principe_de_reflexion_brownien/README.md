# Principe de Réflexion du Mouvement Brownien, Validation Numérique

Extension de l'exercice initial sur le principe de réflexion : simulation
Monte Carlo, vérification empirique de la loi du maximum courant et du
temps de premier passage, avec visualisation directe de l'argument de
réflexion sur une trajectoire.

## Le principe

Pour un mouvement brownien standard `W_t` et un niveau `a > 0` :

```
P(M_T >= a) = 2 * P(W_T >= a),   où M_T = max_{0≤s≤T} W_s
```

**Argument de réflexion** (Désiré André, 1887 / Lévy) : toute
trajectoire qui touche `a` avant `T` peut être "réfléchie" après
l'instant de premier passage `τ_a` (on remplace `W_s` par `2a - W_s`
pour `s > τ_a`) pour produire une autre trajectoire brownienne tout
aussi probable. Comme les trajectoires qui finissent au-dessus de `a`
en `T` ont *nécessairement* touché `a` avant, et que chacune a une
réfléchie tout aussi probable qui finit en dessous, on obtient le
facteur 2, sans calcul, juste par symétrie.

## Deux conséquences vérifiées numériquement

1. **Loi du maximum courant** (demi-normale) :
   `f_{M_T}(m) = sqrt(2/(πT)) exp(-m²/2T)`, `m ≥ 0`
2. **Loi du temps de premier passage** (loi de Lévy) :
   `f_{τ_a}(t) = (a/√(2πt³)) exp(-a²/2t)`

## Résultats (a=0.5, T=1, 20 000 trajectoires, 3000 pas)

| Quantité | Empirique | Théorique | Écart |
|---|---|---|---|
| P(M_T ≥ a) | 0.6121 | 0.6171 | 0.0050 |
| Fraction ayant touché a avant T | 0.6120 | 0.6171 | cohérent avec P(M_T≥a) |

L'écart résiduel (~0.5 point) est un biais de discrétisation temporelle
attendu : sur une grille discrète, le maximum simulé sous-estime
légèrement le vrai maximum continu (une trajectoire peut dépasser `a`
entre deux pas de grille sans que la simulation ne le détecte), écart
qui diminue avec un pas de temps plus fin (vérifié : passer de 500 à
3000 pas réduit l'écart de 0.0195 à 0.0050, facteur ~4, cohérent avec
la convergence attendue pour ce type de biais).

## Structure

```
src/main.py    # Simulation, les 2 vérifications, 4 graphiques diagnostiques
outputs/          # Graphique (committé)
```

## Usage

```bash
cd src && python main.py
```

## Correction apportée (normalisation conditionnelle)

Le panneau "Distribution du temps de premier passage" comparait initialement
l'histogramme empirique de `tau_hit` (calculé uniquement sur les
trajectoires ayant touché `a` avant T, et normalisé par `density=True`
pour intégrer à 1 sur ce seul sous-ensemble) à la densité de Lévy
**inconditionnelle** (qui intègre à 1 sur tout `t ∈ [0, +∞)`). Comme
seulement `P(τ_a≤T) ≈ 61.7%` de la masse théorique tombe dans `[0,T]`,
la courbe théorique apparaissait sous-estimée d'un facteur `≈ 1/0.617
≈ 1.62` par rapport à l'histogramme, deux quantités non comparables
en l'état, malgré une forme identique. La courbe théorique trace
désormais la densité **conditionnelle** `f_τ(t) / P(τ_a≤T)`, seule
quantité directement comparable à l'histogramme de `tau_hit`. Les deux
courbes se superposent maintenant correctement.

## Limites connues

- Biais de discrétisation documenté ci-dessus, inhérent à toute
  simulation à pas de temps fini d'un objet en temps continu.
- Un seul niveau `a` testé, le code se généralise directement à
  plusieurs niveaux/temps pour une étude de sensibilité plus large.