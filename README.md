#  Projet Machine Learning – Régression Linéaire & Régression Logistique

##  Objectif du projet
Ce projet vise à appliquer des **algorithmes de Machine Learning Supervisé (MLS)** afin de résoudre :
- un problème de **régression** (prédiction d’une variable continue),
- un problème de **classification** (prédiction d’une classe),

en suivant une **démarche méthodologique complète** allant de l’exploration des données à l’évaluation des performances.

---

##  Algorithmes de Machine Learning utilisés

|------Partie------|----------Type--------|-----Algorithme--------|
|------------------|----------------------|-----------------------|
| Partie 1         | MLS – Régression     | Régression Linéaire   |
| Partie 2         | MLS – Classification | Régression Logistique |

```

## Contenu du projet



.
├── mini-projet-RLineair.ipynb  # Notebook Google Colab de  Régression Linéaire 
├── mini-projet-Rlogistique.ipynb  # Notebook Google Colab de Régression Logistique
└── README.md        # Documentation du projet
```




##  Partie 1 : Régression Linéaire  
### Dataset : Medical Insurance Cost

###  Description du jeu de données

Ce jeu de données contient des informations sur les **coûts d’assurance médicale pour 1 338 individus**.  
Il inclut des variables **démographiques** et **liées à la santé**, telles que l’âge, le sexe, l’indice de masse corporelle (IMC), le nombre d’enfants, le statut de fumeur et la région de résidence aux États-Unis.

La **variable cible** est `charges`, qui représente le **coût de l’assurance médicale facturé à l’individu**.

Ce dataset est couramment utilisé pour :
- la **modélisation par régression**,
- la **recherche en économie de la santé**,
- l’**analyse de la tarification des assurances**,
- l’**enseignement du machine learning** et des techniques d’ingénierie des caractéristiques.

####  Colonnes du dataset
- **age** : âge du bénéficiaire principal (entier)
- **sex** : sexe du bénéficiaire (male, female)
- **bmi** : indice de masse corporelle (IMC), mesure de la masse grasse basée sur la taille et le poids (réel)
- **children** : nombre d’enfants couverts par l’assurance santé (entier)
- **smoker** : statut de fumeur du bénéficiaire (yes, no)
- **region** : région de résidence aux États-Unis (northeast, northwest, southeast, southwest)
- **charges** : coût de l’assurance médicale facturé au bénéficiaire (réel)

####  Utilisations potentielles
- Construire des **modèles prédictifs** des coûts médicaux
- Étudier l’impact du **tabagisme** et de l’**IMC** sur les dépenses de santé
- Illustrer les concepts de **régression linéaire** et de **feature engineering**
- Analyser les tendances liées à l’accessibilité des soins

---

###  Étapes méthodologiques suivies

#### 1 Chargement et exploration des données
- Lecture du dataset
- Visualisation des premières lignes
- Analyse des types de variables

#### 2 Prétraitement des données
- Séparation des variables explicatives et de la variable cible
- Encodage des variables catégorielles (One-Hot Encoding)
- Vérification des valeurs manquantes

#### 3 Analyse exploratoire
- Calcul de la matrice de corrélation
- Visualisation via une **Heatmap**
- Analyse des relations entre variables

#### 4 Modélisation mathématique
Le modèle de régression linéaire est défini par :

\[
y = \beta_0 + \sum_{i=1}^{n} \beta_i x_i + \varepsilon
\]

#### 5 Entraînement du modèle
- Séparation des données en ensembles d’entraînement et de test
- Utilisation de **LinearRegression** (scikit-learn)

#### 6 Évaluation des performances
Les performances sont évaluées à l’aide de :
- **Mean Squared Error (MSE)**
- **Coefficient de détermination (R²)**

#### 7 Interprétation des résultats
- Analyse de l’influence de chaque variable explicative
- Identification des facteurs ayant le plus d’impact sur les coûts médicaux

---

##  Partie 2 : Régression Logistique  


Ce projet illustre l’application de la **régression logistique** sur le dataset **Iris** (disponible dans *scikit-learn*). Le problème est transformé en **classification binaire** :

* **Classe 1** : Iris *Setosa* (classe 0)
* **Classe 0** : Toutes les autres espèces (*Versicolor* et *Virginica*)

L’objectif est de prédire la probabilité d’appartenance à la classe *Setosa* à partir des caractéristiques des fleurs.

---


##  Données

Le dataset **Iris** contient 150 observations avec 4 variables explicatives :

* Longueur du sépale (sepal length)
* Largeur du sépale (sepal width)
* Longueur du pétale (petal length)
* Largeur du pétale (petal width)

La variable cible originale comporte 3 classes. Elle est transformée en une **variable binaire** :

* `1` : Setosa
* `0` : Autres

---

##  Étapes méthodologiques suivies

### 1 Préparation des données

* Chargement du dataset Iris depuis `sklearn.datasets`
* Conversion en DataFrame pour faciliter la manipulation
* Transformation de la variable cible en classification binaire

### 2 Séparation des données

* Division en **jeu d’entraînement (80%)** et **jeu de test (20%)**
* Objectif : entraîner le modèle sur une partie des données et évaluer sa performance sur des données jamais vues

### 3 Normalisation

* Application de la **standardisation** (moyenne = 0, écart-type = 1)
* Permet d’améliorer la convergence et la performance du modèle de régression logistique

### 4 Modélisation : Régression Logistique

La régression logistique estime la probabilité :

[ P(y = 1 | x) = \frac{1}{1 + e^{-z}} ]

avec :

[ z = w^T x + b ]

* `x` : vecteur des variables d’entrée
* `w` : poids du modèle
* `b` : biais

La fonction **sigmoïde** transforme la sortie en une probabilité comprise entre 0 et 1.

### 5 Prédiction

* Le modèle prédit la classe (0 ou 1) pour chaque observation du jeu de test

### 6 Évaluation du modèle

#### 7 Matrice de confusion

La matrice de confusion permet d’analyser les erreurs de classification :

|            | Prédit 0 | Prédit 1 |
| ---------- | -------- | -------- |
| **Réel 0** | TN       | FP       |
| **Réel 1** | FN       | TP       |

* **TP (True Positives)** : Setosa correctement détectées
* **TN (True Negatives)** : Autres classes correctement détectées
* **FP (False Positives)** : Autres classes prédites à tort comme Setosa
* **FN (False Negatives)** : Setosa non détectées

#### 8 Métriques

* **Accuracy (Exactitude)**
  [ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} ]

* **Precision (Précision)**
  [ Precision = \frac{TP}{TP + FP} ]

* **Recall (Rappel)**
  [ Recall = \frac{TP}{TP + FN} ]

Ces indicateurs permettent d’évaluer la qualité globale du modèle ainsi que sa capacité à détecter correctement la classe positive (Setosa).

---

##  Exécution

1. Ouvrir les notebooks dans **Google Colab**
2. Exécuter les cellules dans l’ordre :

   * Importation des bibliothèques
   * Chargement des données
   * Transformation en classification binaire
   * Séparation des données
   * Normalisation
   * Entraînement du modèle
   * Prédictions
   * Évaluation (matrice de confusion et métriques)

---

##  Résultats attendus

* Une **forte accuracy** indiquant une bonne performance globale
* Une **précision élevée** montrant que les prédictions de Setosa sont fiables
* Un **recall élevé** indiquant que la majorité des Setosa sont correctement identifiées

---

##  Outils et technologies utilisés
- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Google Colab

---
Auteur : medabdellahi elbah


