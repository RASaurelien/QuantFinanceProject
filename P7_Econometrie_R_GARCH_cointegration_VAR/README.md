# Économétrie R GARCH, Cointégration, VAR

Trois techniques économétriques classiques, chacune **implémentée à la
main** (MLE par `optim()`, test ADF, VAR par OLS matriciel) plutôt
qu'appelée depuis un package spécialisé (`rugarch`, `urca`, `vars`),
choix imposé par l'absence d'accès à CRAN dans l'environnement de
développement, mais qui a une vraie valeur pédagogique : comprendre
exactement ce que fait chaque test plutôt que lire une p-value produite
par une boîte noire. Prolongement direct du cours d'économétrie déjà
suivi (Bourbonnais).

## Les trois modules

### 1. GARCH(1,1), `garch.R`

Volatilité conditionnelle, estimée par maximum de vraisemblance
(`optim()` sur la log-vraisemblance gaussienne, avec pénalité douce
pour forcer la stationnarité `alpha+beta<1`).

**Validation** : simulation à partir de paramètres connus
(`omega=0.05, alpha=0.10, beta=0.85`), ré-estimés à
`omega=0.052, alpha=0.107, beta=0.845`, écart de **0.22 %** sur la
persistance de la volatilité (`alpha+beta`).

**Diagnostic classique reproduit** : l'ACF des rendements est quasi
nulle, celle des rendements **au carré** est fortement positive sur
plusieurs retards, la signature empirique du clustering de volatilité
que GARCH capture et qu'un bruit blanc ne capture pas.

### 2. Cointégration `cointegration.R`

Méthode d'Engle-Granger en deux étapes (régression de long terme +
test ADF sur le résidu), appliquée à **deux paires simulées** :

- une paire cointégrée (tendance stochastique commune) → t-stat ADF
  **-16.6** (largement sous le seuil -2.86) → cointégration détectée ✓
- une paire **non** cointégrée, contrôle négatif (deux marches
  aléatoires indépendantes) → t-stat ADF **-2.22** (au-dessus du
  seuil) → pas de cointégration détectée, comme attendu ✓

Le contrôle négatif est la partie qui compte : sans lui, un test qui
"détecte toujours de la cointégration" pourrait sembler fonctionner
alors qu'il ne teste rien. Un signal de pairs trading (z-score du
spread, seuils ±2σ) est dérivé de la paire cointégrée.

### 3. VAR(1) bivarié `var_model.R`

Estimation par OLS matriciel (`A_hat = (X'X)^-1 X'Y`), sur un système
simulé interprété comme croissance du PIB / taux d'intérêt court
terme. Fonctions de réponse impulsionnelle (IRF) calculées par
récursion sur les puissances de la matrice compagnon, orthogonalisées
par décomposition de Cholesky (identification récursive standard).

**Validation** : matrice `A` retrouvée avec une erreur quadratique
moyenne de **0.00058** par rapport à la matrice génératrice, les IRF
estimées se superposent presque exactement aux IRF théoriques (voir
graphique).

## Structure

```
R/
  utils.R            # Test ADF implémenté à la main (régression + valeurs critiques MacKinnon)
  garch.R              # GARCH(1,1) : simulation + MLE + diagnostics ACF
  cointegration.R       # Engle-Granger (2 étapes) + contrôle négatif + pairs trading
  var_model.R            # VAR(1) OLS + IRF orthogonalisées (Cholesky)
outputs/                   # Graphiques (committés : preuve visuelle pour le README)
```

## Usage

Les scripts sont conçus pour fonctionner quel que soit le répertoire de lancement, grâce à des chemins calculés depuis leur emplacement sur disque. Depuis la racine du projet, la commande la plus robuste est :

```bash
Rscript "V1/files 7. Modélisation GARCH  cointégration  VAR/garch.R"
Rscript "V1/files 7. Modélisation GARCH  cointégration  VAR/cointegration.R"
Rscript "V1/files 7. Modélisation GARCH  cointégration  VAR/var_model.R"
```

Ou, si vous êtes déjà dans le dossier du projet :

```bash
Rscript garch.R
Rscript cointegration.R
Rscript var_model.R
```

Pour exécuter les trois scripts en une seule commande et régénérer les
graphiques, utilisez le lanceur `run_all.R` :

```bash
Rscript run_all.R
```

Les anciennes images sont supprimées avant chaque exécution, puis les
fichiers `garch_diagnostics.png`, `cointegration.png` et `var_irf.png`
sont recréés dans `../outputs`.

Aucune dépendance externe : les trois scripts tournent en **base R
uniquement** (pas de `rugarch`, `urca`, `vars`, `ggplot2`).

## Limites connues

- Valeurs critiques ADF hardcodées (table asymptotique de MacKinnon,
  cas "constante sans tendance") plutôt que calculées par simulation
  (bootstrap), approximation standard mais moins précise en petit
  échantillon qu'une procédure `urca::ur.df` complète.
- GARCH(1,1) simple (innovations gaussiennes), pas d'extension
  GJR-GARCH/EGARCH pour l'asymétrie (effet de levier), ni de
  distribution à queues épaisses (Student-t), pourtant plus réalistes
  empiriquement.
- VAR(1) seulement (un seul retard), le choix du nombre de retards
  optimal (AIC/BIC) n'est pas implémenté, ni les tests de causalité de
  Granger.
- Cointégration testée sur une seule paire simulée par cas ; en
  pratique, un vrai signal de pairs trading nécessiterait un scan
  systématique sur un univers d'actifs avec correction pour tests
  multiples (le risque de "cointégration fallacieuse" trouvée par
  hasard sur beaucoup de paires testées).
