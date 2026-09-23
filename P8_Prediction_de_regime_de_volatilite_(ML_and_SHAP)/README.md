# Prédiction du Régime de Volatilité, Random Forest + SHAP (Python)

Classifieur de régime de marché (calme / stress) à horizon 1 jour, à
partir de features de volatilité réalisée glissantes, avec
interprétabilité SHAP plutôt qu'un modèle boîte noire.

## Pourquoi ce projet, et pourquoi pas un réseau de neurones

Sur un problème tabulaire de cette taille (14 features, quelques
milliers d'observations), un Random Forest est aussi performant qu'un
réseau de neurones, voire plus, avec beaucoup moins de risque de
sur-apprentissage et surtout reste **interprétable**. C'est un choix
délibéré : éviter le piège classique du "trading bot deep learning"
générique et illisible, au profit d'un modèle dont on peut *expliquer*
chaque prédiction (SHAP), ce qui est aussi ce qu'attend un desk de
risque en pratique (un modèle qu'on ne peut pas expliquer ne passe pas
la validation interne).

## Ce que le projet montre vraiment (et ce n'est pas ce qu'on attend au premier abord)

Le résultat le plus intéressant n'est pas "le modèle bat une baseline", 
c'est l'inverse, et c'est plus instructif :

| | Accuracy globale |
|---|---|
| Baseline naïve ("demain = aujourd'hui") | **98.1 %** |
| Random Forest | 93.6 % (AUC = 0.976) |

La baseline naïve **gagne** en accuracy globale. Ce n'est pas un échec
du modèle : les régimes sont très persistants par construction
(probabilité de rester dans le même état > 95 % par jour), donc
"demain = aujourd'hui" est une baseline extrêmement forte presque tout
le temps, c'est un rappel utile que l'accuracy globale peut être un
indicateur trompeur sur des séries à forte persistance.

**La vraie question n'est pas "bat-il la persistance en général" mais
"anticipe-t-il les bascules de régime"**, le sous-ensemble de jours où
la baseline naïve est *structurellement* fausse à 100 % (elle prédit
toujours "pas de changement") :

| | Accuracy sur les jours de transition (1.9 % du test) |
|---|---|
| Baseline naïve | 0.0 % (fausse par construction) |
| Random Forest | 7.1 % |

Le Random Forest ne prédit quasiment pas non plus le jour *exact*
d'une bascule. **C'est le résultat scientifiquement correct, pas un
bug** : dans une chaîne de Markov sans mémoire (ce qui est le modèle
sous-jacent ici), l'instant de la transition ne dépend d'aucune
information passée par construction, aucune feature calculée sur
l'historique ne peut porter de signal avant-coureur sur le jour précis
du saut. Le modèle fait ce qu'il peut faire (classer correctement
l'appartenance au régime avec un AUC de 0.976, 95.3 % sur les jours
stables) et pas ce qui est mathématiquement impossible (prédire un
événement sans mémoire).

## Pipeline

```
Simulation GBM à changement de régime markovien (régime VRAI connu)
     │
     ▼
Features glissantes (vol réalisée multi-horizons, skew/kurtosis,
autocorrélation, drawdown...) -- STRICTEMENT rétrospectives
     │
     ▼
Split chronologique train/test (PAS de shuffle -- anti-fuite temporelle)
     │
     ▼
Random Forest (class_weight="balanced")  vs  baseline "persistance"
     │
     ▼
Évaluation globale + évaluation ciblée sur les jours de transition
     │
     ▼
SHAP TreeExplainer : quelles features pèsent dans la décision
```

## Structure

```
src/
  simulate.py      # GBM à changement de régime markovien (régime vrai connu)
  features.py       # Features glissantes sans fuite temporelle + cible (régime t+1)
  main.py            # Pipeline complet : split, RF, baseline, SHAP, graphiques
requirements.txt
outputs/               # Graphiques (committés : preuve visuelle pour le README)
```

## Usage

```bash
pip install -r requirements.txt
cd src && python main.py
```

## Limites connues

- Le régime simulé est un processus markovien SANS MÉMOIRE : c'est ce
  qui explique (et justifie) la faible accuracy sur le jour exact
  d'une transition. Un marché réel a probablement une composante plus
  prévisible (contagion, feedback loops, nouvelles macro
  programmées) qu'un simple modèle à 2 états markovien ne capture pas, 
  un vrai test sur données de marché réelles donnerait
  probablement un résultat différent, potentiellement plus favorable
  au ML sur les transitions.
- Un seul horizon de prédiction (t+1), un horizon plus long (t+5,
  t+20) changerait complètement l'équilibre entre baseline naïve et
  modèle ML.
- Pas de coût de transaction ni de stratégie de trading associée à la
  prédiction, ce projet évalue la qualité de la classification, pas
  sa rentabilité économique une fois exploitée.
- SHAP TreeExplainer est exact pour les modèles à base d'arbres
  (Random Forest) mais ne se généralise pas directement à un modèle de
  deep learning, qui nécessiterait KernelSHAP ou DeepSHAP (plus lents,
  approximatifs).
