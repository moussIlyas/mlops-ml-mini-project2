# mlops-ml-project — baseline ML reproductible

Baseline de classification (dataset Iris, intégré à scikit-learn) : configuration YAML,
modules réutilisables, scripts d'entraînement / évaluation, tests et artefacts.

## Prérequis

- Python 3.10+ (testé avec Python 3.13)
- Git

## Installation

```bash
python -m venv .venv
# Linux/macOS :
source .venv/bin/activate
# Windows (PowerShell) :
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> Les versions ne sont pas figées dans `requirements.txt`. Pour figer un environnement
> reproductible : `pip freeze > requirements.txt`.

## Entraînement

```bash
python scripts/train.py
```

Sorties dans `artifacts/` : `model.joblib`, `metrics.json`, `confusion_matrix.png`.

## Évaluation

```bash
python scripts/evaluate.py
```

Évalue le modèle sur le **jeu de test** (même découpage que `train.py`) et écrit
`artifacts/report.json`.

## Tests

```bash
python -m pytest -q
```

## Artefacts (générés dans artifacts/, non versionnés)

- `model.joblib` : pipeline entraîné (prétraitement + modèle)
- `metrics.json` : accuracy et f1_macro sur le jeu de test
- `confusion_matrix.png` : matrice de confusion
- `report.json` : rapport de classification détaillé

## Structure

```
config/train.yaml       # configuration de l'expérience
src/data.py             # chargement (Iris ou CSV)
src/features.py         # prétraitement (imputation + standardisation [+ clipping])
src/model.py            # construction du modèle
scripts/train.py        # entraînement + artefacts
scripts/evaluate.py     # évaluation sur le jeu de test
tests/                  # tests pytest
data/                   # données locales (ignorées par Git)
artifacts/              # sorties de l'entraînement (ignorées par Git)
```
