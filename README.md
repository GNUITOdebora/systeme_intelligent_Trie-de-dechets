# ♻️ Computer Vision — Classification des Déchets

> Projet de vision par ordinateur pour la classification automatique des déchets organiques et recyclables.  
> **Élaboré par** : Débora GNUITO & Moheddine BEN ABDALLAH

---

## 📋 Table des matières

1. [Contexte et problématique](#contexte-et-problématique)
2. [Structure du projet](#structure-du-projet)
3. [Partie 1 — Descripteurs Hand-crafted](#partie-1--descripteurs-hand-crafted)
4. [Partie 2 — Descripteurs de Haut Niveau (Deep Learning)](#partie-2--descripteurs-de-haut-niveau-deep-learning)
5. [Partie 3 — Segmentation avec YOLO](#partie-3--segmentation-avec-yolo)
6. [Frontend](#frontend)
7. [Résultats](#résultats)
8. [Perspectives](#perspectives)

---

## 🌍 Contexte et problématique

La gestion des déchets est un défi environnemental crucial. Une grande partie des déchets dirigée vers des décharges engendre des impacts écologiques graves : pollution des sols, de l'eau, de l'air, intoxication des écosystèmes et accumulation de toxines.

**Problématique :**  
> Comment concevoir un système intelligent capable de distinguer efficacement les déchets organiques des déchets recyclables afin de réduire leur impact environnemental et d'encourager leur valorisation ?

---

## 📁 Structure du projet

```
.
├── PART1_Project.ipynb       # Approche 1 : Descripteurs hand-crafted
├── PART2_Project.ipynb       # Approche 2 : Descripteurs de haut niveau (Deep Learning)
├── PART3_Project.ipynb       # Segmentation YOLO + Classification
└── frontend/
    └── main.py               # Interface utilisateur (Streamlit)
```

---

## 🔧 Partie 1 — Descripteurs Hand-crafted

**Fichier :** `PART1_Project.ipynb`

Cette approche consiste à extraire manuellement des caractéristiques spécifiques des images à l'aide de descripteurs "hand-crafted", qui sont ensuite utilisées pour entraîner un modèle de classification.

### Méthodes testées

| Méthode | Texture | Contours | Couleur |
|--------|---------|----------|---------|
| Méthode 1 | LBP | Sobel | — |
| Méthode 2 | Gabor | Scharr | — |
| Méthode 3 | GLCM | Canny | HSV |

### Sélection des caractéristiques
- **Embedded Method** : Sélection par modèle
- **Filter Method** : Sélection par test statistique

### Classifieurs utilisés
`Random Forest`, `XGBoost`, `KNN`, `SVM`

### ✅ Meilleur résultat
**XGBoost** avec la Méthode 3 (GLCM + Canny + HSV) et sélection de features par modèle :

| Classe | Précision | Rappel | F1-score |
|--------|-----------|--------|----------|
| Recyclable | 0.84 | 0.80 | 0.82 |
| Organique | 0.86 | 0.89 | 0.87 |
| **Accuracy** | | | **0.85** |

---

## 🧠 Partie 2 — Descripteurs de Haut Niveau (Deep Learning)

**Fichier :** `PART2_Project.ipynb`

Cette approche repose sur l'extraction automatique de caractéristiques pertinentes à l'aide de réseaux de neurones profonds (Deep Learning) pré-entraînés, utilisés comme extracteurs de features.

### Méthodes testées

| Méthode | Extracteur de features |
|--------|------------------------|
| Méthode 1 | ResNet50 |
| Méthode 2 | VGG |

### Classifieurs utilisés
`Random Forest`, `XGBoost`, `KNN`, `SVM`

### ✅ Meilleur résultat
**SVM + ResNet50** :

| Classe | Précision | Rappel | F1-score |
|--------|-----------|--------|----------|
| Recyclable | 0.97 | 0.96 | 0.96 |
| Organique | 0.97 | 0.98 | 0.97 |
| **Accuracy** | | | **0.97** |

> 📈 Amélioration significative (+12%) par rapport à l'approche hand-crafted, grâce à l'utilisation de caractéristiques de haut niveau extraites via un réseau de neurones profond.

---

## 🎯 Partie 3 — Segmentation avec YOLO

**Fichier :** `PART3_Project.ipynb`

Cette partie combine la détection/segmentation d'objets avec la classification pour traiter des images contenant plusieurs déchets dans des contextes variés.

### Pipeline

```
Collecte d'images  →  Segmentation (YOLOv8)  →  Classification (SVM + ResNet50)
```

1. **Collecte d'images** : Réunir un ensemble d'images contenant plusieurs types de déchets dans des contextes variés.
2. **Segmentation** : Utilisation de **YOLOv8** (Ultralytics) pour isoler les objets "déchets" dans l'image.
3. **Classification** : Application du meilleur modèle (SVM + ResNet50) sur chaque objet isolé pour classifier son type.

---

## 🖥️ Frontend

**Fichier :** `frontend/main.py`

Interface web développée avec **Streamlit**, offrant trois fonctionnalités :

- **Classification avec SVM** : Upload d'une ou plusieurs images → classification directe (Organique / Recyclable)
- **Segmentation avec YOLO et SVM** : Upload d'une image → détection des objets + classification de chaque déchet détecté
- **Team** : Informations sur l'équipe du projet

### Lancement du frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run main.py
```

---

## 📊 Comparaison des approches

| Critère | Hand-crafted | Deep Learning (Haut Niveau) |
|--------|-------------|------------------------------|
| Extraction des features | Manuelle | Automatique |
| Besoin en données | Modéré | Important |
| Précision (meilleur modèle) | 85% | **97%** |
| Temps d'exécution | ~0.96s | ~18.67s |
| Avantages | Peu de données, interprétable | Haute précision, automatique |
| Inconvénients | Performances limitées sur données complexes | Coûteux en ressources |

---

## 🔭 Perspectives

- Amélioration de la précision du modèle avec des données supplémentaires.
- Intégration d'un modèle de détection et segmentation personnalisé pour notifier la municipalité en temps réel, facilitant la collecte et l'analyse des données pour une gestion optimale des déchets.

---

## 👥 Équipe

| Nom | Filière |
|-----|---------|
| Moheddine BEN ABDALLAH | I3-FSS |
| Débora GNUITO | I3-FSS |
