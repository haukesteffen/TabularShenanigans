# TabularShenanigans

Vibe-coded MVP scaffold for reusable Kaggle tabular competitions with profile-based runtime parity across:
- local Apple Silicon development (CPU/MPS)
- remote cloud GPU training

## Quick Start

1. Create environment and install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[train,kaggle,notebook]
```
(`train` extra now includes Optuna for `ts tune-stage` and `ts stack-stage`.)

2. Copy Kaggle credentials if you plan to fetch from API:
```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

3. Validate setup:
```bash
ts validate-env --profile local_arm64
```

4. Run local baseline flow:
```bash
ts fetch-data --competition titanic
ts train --competition titanic --profile local_arm64
ts predict --competition titanic --profile local_arm64
ts make-submission --competition titanic --profile local_arm64
```

## Notebook-First Workflow

- Keep experimentation in `notebooks/`.
- Keep all production logic in `src/` and call modules from notebooks.
- Move stable notebook logic into `src/` before running remote jobs.

Open notebook environment:
```bash
make notebook
```

## Profiles

Runtime-specific settings live in `configs/profiles/`:
- `local_arm64.yaml`: local CPU/MPS defaults
- `remote_gpu.yaml`: cloud GPU defaults

Competition settings live in `configs/competitions/`.
Set competition-specific schema keys here:
- `schema.target` (training label column)
- `schema.id_column` (submission id column)
- `schema.train_file` / `schema.test_file` (defaults: `train.csv`, `test.csv`)
- optional `schema.task_type`: `classification` or `regression` (otherwise inferred)
- examples: `titanic.yaml` (binary classification), `house_prices.yaml` (regression)

Training model selection is config-driven:
- `training.model_family`: `sklearn`, `lightgbm`, `xgboost`, or `catboost`
- optional `training.model_params`: dict passed to the selected estimator
- optional random-search tuning:
  - `training.tune.n_trials`
  - `training.tune.search_space` with param specs (`int`, `float`, `categorical`, or list)
- optional multi-learner tuning and auto-stacking:
  - `training.ensemble.learners[]` with per-learner `model_family`, `model_params`, `tune`
  - `training.ensemble.stack.enabled` + `method` (`linear`/`mean`)

Evaluation is config-driven:
- `evaluation.metric`: classification (`accuracy`, `f1`, `roc_auc`, `logloss`)
- `evaluation.metric`: regression (`rmse`, `mae`)
- `evaluation.direction`: `maximize` or `minimize`
- Invalid metric/task combinations fail fast at train start
- each run stores all available metrics in `results.metrics`
- configured `evaluation.metric` remains the primary optimization/comparison metric

Leakage-safe CV:
- folds are persisted once at `data/<competition>/splits/<split_version>/folds.csv`
- all stages reuse these immutable folds (`cv.split_version`, default `v1`)
- fold transforms are fit on train-fold only and applied to val/test

Submission output is also config-driven:
- `submission.prediction_type`: `raw` or `label`
- `submission.classification_threshold`: threshold when `prediction_type: label`
- `submission.positive_label` / `submission.negative_label`

Experiment versioning:
- Train creates `artifacts/<competition>/runs/<run_id>/`
- `artifacts/<competition>/latest` is synced from the selected run
- Downstream commands support `--run-id`; default is latest tracked run

## MVP Commands

- `ts fetch-data`: Kaggle API download + unzip into `data/<competition>/raw`
- `ts prepare-data`: builds processed train/test artifacts from raw files
- `ts train`: builds processed data, runs CV baseline, saves fold models + OOF
- `ts baseline`: stage 1 automation (train baseline model set end-to-end)
- `ts tune-stage`: stage 2 automation (Optuna tuning across baseline model classes)
- `ts stack-stage`: stage 3 automation (meta-learner tuning/selection on OOF stack)
- `ts tune`: runs random-search trials and promotes best trial run
- `ts predict`: loads fold models and predicts test set
- `ts stack`: builds stacked OOF/test predictions from multiple run ids
- `ts make-submission`: creates submission CSV from processed IDs + predictions
- `ts submit`: uploads `submission.csv` to Kaggle and logs metadata
- `ts submissions`: prints Kaggle submission history for the competition
- `ts runs`: lists stored run ids and key metrics
- `ts compare-runs`: ranks run scores for a selected metric and shows deltas
- `ts best-run`: prints the top run for configured metric (or `--metric`)
- `ts promote-run`: marks an existing run as latest without retraining
- `ts clean`: removes stored raw/processed/artifact outputs (safe confirmation + dry-run)
- `ts check`: validates config combinations and runs unit tests
- `ts validate-env`: prints active runtime/device details

Submission validation gate:
- runs automatically in `ts make-submission` and `ts submit`
- checks required columns, row count, nulls, duplicate ids, id/order match
- checks label-domain for `submission.prediction_type: label`

Prepare-data report:
- `ts prepare-data` writes `data/<competition>/processed/prepare_report.json`
- includes schema mismatches, dtype mismatches, missing-value summary, and basic leakage flag (`target_present_in_test`)

Run comparison examples:
```bash
ts compare-runs --competition titanic --profile local_arm64
ts compare-runs --competition titanic --profile local_arm64 --metric accuracy
ts best-run --competition titanic --profile local_arm64
ts promote-run --competition titanic --profile local_arm64 --run-id <run_id>
ts stack --competition titanic --profile local_arm64 --run-id <run_a> --run-id <run_b> --method linear
ts tune --competition titanic --profile local_arm64 --n-trials 10
# with training.ensemble.learners configured:
ts tune --competition house_prices --profile local_arm64
# 3-stage workflow:
ts baseline --competition house_prices --profile local_arm64
ts tune-stage --competition house_prices --profile local_arm64 --n-trials 20
ts stack-stage --competition house_prices --profile local_arm64 --n-trials 20
```

Cleanup examples:
```bash
ts clean --competition titanic --scope all --dry-run
ts clean --competition titanic --scope artifacts --yes
ts clean --all-competitions --scope all --dry-run
```

## Make Targets

```bash
make train-local COMP=titanic
make train-remote COMP=titanic
make notebook
make test
make check
```

## Tests

Run lightweight reliability tests:
```bash
make test
```

## Regression Smoke Run

Example end-to-end flow for House Prices:
```bash
ts fetch-data --competition house_prices --profile local_arm64 --force
ts prepare-data --competition house_prices --profile local_arm64
ts train --competition house_prices --profile local_arm64
ts predict --competition house_prices --profile local_arm64
ts make-submission --competition house_prices --profile local_arm64
```

## Next Up

- Add model family presets per runtime profile (`local_arm64`, `remote_gpu`)
- Add competition-specific metric presets and validation checks
- Add stacking, experiment tracking, and artifact sync automation
