# TabularShenanigans

Config-driven Python workflows for semi-automated participation in tabular Kaggle competitions.

## Scope

- Focus area: tabular machine learning competitions.
- Primary target: Kaggle Playground Series tabular competitions.
- Task scope: regression and binary classification.
- Canonical experiment store: MLflow.

## Current Capabilities

- Load and validate one repository-root `config.yaml`.
- Fetch Kaggle competition data into `data/<competition_slug>/` when the zip is missing.
- Infer the Playground-style submission schema from `train.csv`, `test.csv`, and `sample_submission.csv`, with optional config overrides for `id_column` and `label_column`.
- Build deterministic competition-level fold assignments in memory from the configured CV settings.
- Run deterministic feature recipes for model candidates.
- Train one configured candidate or one ordered batch of configured candidates in a single process.
- Train blend candidates by downloading compatible base candidates from MLflow and combining their saved predictions.
- Run Optuna inside `train` for model candidates when optimization is enabled.
- Validate prediction artifacts against `sample_submission.csv` before submission.
- Submit an explicitly selected candidate to Kaggle from MLflow-backed candidate artifacts.
- Refresh Kaggle submission outcomes back onto the same MLflow candidate runs.

## Quickstart

```bash
uv sync --extra boosters
cp config.binary.example.yaml config.yaml
# edit config.yaml: set experiment.tracking.tracking_uri
uv run python main.py
```

See [USAGE.md](USAGE.md) for prerequisites, full setup, and command reference.

## Core Assumptions

- Kaggle CLI authentication is preconfigured.
- An MLflow tracking server is available.
- `config.yaml` is the only runtime config source and is Git-ignored.
- RAPIDS GPU acceleration is a Linux-only concern; macOS runs stay on CPU.

## Tooling

- Python for orchestration
- Kaggle CLI for competition data and submissions
- Optuna for hyperparameter tuning
- MLflow for canonical run metadata and artifact storage
- `uv` for environment management

## Documentation

- [USAGE.md](USAGE.md) — setup, commands, config reference, outputs, and operational notes.
- [docs/TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md) — architecture, runtime contracts, module responsibilities, and extension notes.
