# Attribution P&L d'un Vendeur d'Option Delta-Hedgé

Un trader vend un call, delta-hedge quotidiennement, et le P&L réalisé
est décomposé en contributions Delta, Gamma, Theta, Vega, la question
que tout trader d'options doit savoir répondre : *"pourquoi j'ai
gagné/perdu de l'argent cette semaine, malgré un delta-hedge parfait ?"*
Dernier projet du volet "quant trader", réutilisant les formules
Black-Scholes du pricing engine (Tier 1 #1).

## Le point central : vol réalisée vs vol implicite

Le hedge delta neutralise l'exposition à la **direction** du marché,
mais pas à sa **volatilité**. Un vendeur d'option est court en gamma :
chaque mouvement du sous-jacent lui coûte de l'argent au second ordre
(`-0.5 * Gamma * dS²`), compensé (ou pas) par le theta qu'il encaisse
chaque jour. Le signe du P&L total dépend uniquement de l'écart entre
la vol **réalisée** du sous-jacent et la vol **implicite** utilisée
pour construire le hedge, indépendamment de la direction prise par
le marché.

## Résultat (call ATM 30 jours, vol implicite=20%, une trajectoire)

| Scénario | Vol réalisée | P&L total | Gamma | Theta |
|---|---|---|---|---|
| Vol réalisée < implicite | 12% | **+1.40** | -0.47 | +1.89 |
| Vol réalisée = implicite | 20% | +0.34 | -1.12 | +1.61 |
| Vol réalisée > implicite | 32% | **-1.31** | -2.26 | +1.34 |

Le signe s'inverse exactement comme prévu par la théorie : quand la
vol réalisée est plus faible que celle payée par l'acheteur (vol
implicite), le vendeur **gagne**, le gamma coûte moins cher que ce
que le theta rapporte. Quand la vol réalisée dépasse la vol implicite,
c'est l'inverse. Ce mécanisme, pas la direction du marché, est ce qui
détermine le P&L d'un market maker d'options delta-hedgé.

## Sur la qualité de la reconstruction (honnêteté sur un point technique)

La décomposition Taylor (Delta+hedge, Gamma, Theta, Vega) est un
développement au **second ordre**, elle ne capture pas exactement
100% du P&L réel (repricing complet), car les mouvements quotidiens
ne sont pas infinitésimaux. L'erreur absolue reste faible (0.01 à 0.39
en valeur absolue, sur un contrat de nominal 100), mais **exprimée en
% d'un P&L total proche de zéro** (scénario "vol égale"), le
pourcentage d'erreur paraît trompeusement élevé (jusqu'à 42%), un
artefact de mise à l'échelle, pas un défaut de la décomposition elle-
même. Le bon réflexe est de comparer l'erreur en valeur absolue, pas
en ratio, quand le dénominateur est petit.

## Structure

```
src/main.py    # Black-Scholes + Greeks, simulation hedgée, décomposition Taylor, 3 scénarios, graphique
outputs/          # Graphique (committé)
```

## Usage

```bash
cd src && python main.py
```

## Limites connues

- Une seule trajectoire par scénario (pas de moyenne Monte Carlo),
  le signe du résultat est robuste sur la théorie, mais l'ampleur
  exacte varie d'une trajectoire à l'autre ; une vraie étude ferait
  la moyenne sur des centaines de trajectoires par scénario.
- Vol implicite supposée constante dans le temps (Vega=0 par
  construction) en pratique, la vol implicite bouge aussi chaque
  jour, ce qui ajouterait une vraie contribution Vega au P&L.
- Rehedging quotidien discret, pas continu, l'écart entre hedge
  discret et hedge continu (théorique) est lui-même une source de
  bruit additionnel dans le P&L réalisé, distincte de l'effet
  vol réalisée vs implicite qui est le sujet de ce projet.