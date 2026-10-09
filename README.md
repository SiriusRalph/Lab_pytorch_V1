# Lab PyTorch — Les bases du framework

Travaux pratiques d'introduction à **PyTorch** : tenseurs, autograd, construction de modèles avec `nn.Module`, boucle d'apprentissage, puis quatre exercices de régression et de classification (dont un réseau convolutif sur MNIST).

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `Activité pratique Bases de Pytorch.ipynb` | Notebook principal : cours (tenseurs, autograd, `nn.Module`, fonctions de coût, optimiseurs, `train()`/`eval()`, `DataLoader`, sauvegarde) et les quatre exercices résolus |
| `lab1.ipynb` | Premiers essais : création et manipulation de tenseurs, `device` CPU/GPU, calcul de gradients |
| `reg.ipynb` | Régression linéaire simple sur des données synthétiques `y = 3x + 2 + bruit`, avec score R² |
| `src/lab_pytorch_v1/` | Paquet Python minimal généré par `uv` |
| `pyproject.toml`, `uv.lock` | Dépendances du projet |

## Exercices

1. **Régression — prédiction du salaire** : régression linéaire multiple (`nn.Linear(2, 1)`) à partir de l'ancienneté et du niveau d'expertise, avec standardisation des entrées, calcul de la MAE et reconstruction des coefficients en unités d'origine.
2. **Classification MNIST (MLP)** : perceptron multicouche entraîné par mini-lots, métriques Accuracy / Recall / Precision / F1-score, matrice de confusion, images mal classées, sauvegarde et rechargement du modèle.
3. **Classification des fleurs Iris** : petit réseau 4 → 16 → 3, courbe de perte, matrice de confusion, comparaison avec un modèle linéaire sans couche cachée.
4. **Classification MNIST (CNN)** : réseau convolutif à deux blocs Conv → ReLU → MaxPool, avec Dropout, évaluation sur le jeu de test et sauvegarde dans `mnist_cnn.pth`.

### Résultats obtenus

| Exercice | Résultat |
|---|---|
| 1 — Salaire | MAE ≈ 380 DH ; coefficients retrouvés ≈ 593 / 1491 / 4090 (attendus : 600 / 1500 / 4000) |
| 2 — MNIST MLP | Accuracy ≈ 97,4 % après 5 époques |
| 3 — Iris | Accuracy 100 % sur le test (90 % pour le modèle linéaire) |
| 4 — MNIST CNN | Accuracy ≈ 98,9 % après 3 époques |

Les valeurs peuvent légèrement varier d'une exécution à l'autre.

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) et Python 3.11.

```bash
git clone https://github.com/<votre-utilisateur>/lab_pytorch_v1.git
cd lab_pytorch_v1
uv sync
```

`uv sync` crée l'environnement virtuel `.venv` et installe les dépendances : `torch`, `torchvision`, `torcheval`, `torchmetrics`, `scikit-learn`, `matplotlib`, `numpy` et `ipykernel`.

## Utilisation

Ouvrez les notebooks dans VS Code (ou Jupyter) et sélectionnez le noyau Python de `.venv`.

- Le jeu de données MNIST est téléchargé automatiquement dans `data/` lors de la première exécution.
- Le code fonctionne sur CPU ; un GPU CUDA est utilisé automatiquement s'il est disponible.
- La cellule de visualisation du graphe de calcul utilise `torchviz`, qui n'est pas installé par défaut. Pour l'exécuter : `uv add torchviz` (nécessite aussi [Graphviz](https://graphviz.org/download/)).

Les dossiers `data/`, `.venv/` et les modèles sauvegardés (`*.pth`, `*.pt`) sont exclus du dépôt via `.gitignore`.
