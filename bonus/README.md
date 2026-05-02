# Bonus GPU (CUDA)

Ce dossier contient une implementation GPU du MLP en **CUDA** via **CuPy**.

## Fichiers

- `mlp_cuda_cupy.py` : entrainement MLP sur GPU (2 couches cachees), evaluation et benchmark.

## Prerequis

- GPU NVIDIA + driver CUDA fonctionnel
- Python 3.10+
- Environnement virtuel actif

Installer les dependances (exemple CUDA 12.x):

```bash
pip install cupy-cuda12x numpy pandas scikit-learn imbalanced-learn
```

> Si votre machine utilise une autre version CUDA, adaptez le package CuPy:
> `cupy-cuda11x`, `cupy-cuda12x`, etc.

## Lancement

Depuis la racine du projet:

```bash
python3 bonus/mlp_cuda_cupy.py
```

## Sorties

Le script affiche:

- le device CUDA utilise,
- le temps d'entrainement,
- l'accuracy test (`threshold=0.5`),
- l'accuracy test avec threshold optimise sur validation.
