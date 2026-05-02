# RedTeam_T16 - MLP Parallel (OpenMP)

Ce projet entraine un MLP en C (OpenMP) et pilote les experiences avec Python.

## Structure de rendu recommandee

- `serial/` - implementation serie standalone.
- `parallel/` - implementation OpenMP ou MPI (obligatoire).
- `bonus/` - implementation GPU: CUDA, ROCm ou OpenCL (optionnel, +10%).
- `data/` - script de telechargement du dataset ou instructions.
- `README.md` - commandes de compilation et instructions d'execution.

## Structure actuelle du projet

- `parallel/mlp_openmp.c` : moteur de calcul (forward, backward, entrainement parallele).
- `parallel/compile.sh` : compile la librairie partagee `mlp_openmp.so`.
- `parallel/mlp_parallel.py` : script principal (benchmark threads + metriques).
- `data/preprocess.py` : preprocessing complet du dataset Adult.
- `bonus/mlp_cuda_cupy.py` : implementation GPU CUDA (bonus) avec CuPy.
- `bonus/README.md` : prerequis et instructions de lancement du bonus.
- `results/` : sortie des resultats.

## Prerequis

- Windows + WSL (Ubuntu) **ou** Linux natif
- Python 3.10+
- `gcc` avec OpenMP
- `pip`

## Lancement (recommande: terminal Ubuntu/WSL)

Depuis Ubuntu (WSL), lancer:

```bash
cd /mnt/c/Nacer/DevProject/projethpc/RedTeam_T16
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn imbalanced-learn matplotlib
cd parallel
bash compile.sh
python3 mlp_parallel.py
```

## Resultats generes

Apres execution:

- `results/benchmark_results.csv`
- `results/speedup_plot.png`

## Lancer seulement le preprocessing (optionnel)

```bash
cd /mnt/c/Nacer/DevProject/projethpc/RedTeam_T16
source .venv/bin/activate
python3 data/preprocess.py
```

Ce mode genere aussi des graphes EDA dans `results/`:

- `results/eda_overview.png`
- `results/eda_correlation.png`

## Bonus GPU (optionnel, +10%)

Si vous avez un GPU NVIDIA, vous pouvez lancer l'implementation CUDA dans `bonus/`.

Installation (exemple CUDA 12.x):

```bash
pip install cupy-cuda12x numpy pandas scikit-learn imbalanced-learn
```

Execution:

```bash
python3 bonus/mlp_cuda_cupy.py
```

## Depannage rapide

- Si `source` ne marche pas: tu es probablement en PowerShell, pas en Ubuntu.
- Si `bash compile.sh` echoue sous PowerShell: lance via `wsl -e bash -lc "..."`
- Si `ModuleNotFoundError` apparait dans WSL: reinstalle les paquets avec `pip` dans l'environnement WSL active.
