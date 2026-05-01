# RedTeam_T16 - MLP Parallel (OpenMP)

Ce projet entraine un MLP en C (OpenMP) et pilote les experiences avec Python.

## Structure utile

- `parallel/mlp_openmp.c` : moteur de calcul (forward, backward, entrainement parallele).
- `parallel/compile.sh` : compile la librairie partagee `mlp_openmp.so`.
- `parallel/mlp_parallel.py` : script principal (benchmark threads + metriques).
- `data/preprocess.py` : preprocessing complet du dataset Adult.
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

## Depannage rapide

- Si `source` ne marche pas: tu es probablement en PowerShell, pas en Ubuntu.
- Si `bash compile.sh` echoue sous PowerShell: lance via `wsl -e bash -lc "..."`
- Si `ModuleNotFoundError` apparait dans WSL: reinstalle les paquets avec `pip` dans l'environnement WSL active.
