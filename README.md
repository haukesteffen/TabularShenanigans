# TabularShenanigans

Config-driven Python workflows for semi-automated participation in tabular Kaggle competitions.

## Project Overview
- Focus area: tabular machine learning competitions.
- Primary target: Kaggle Playground Series tabular competitions.
- Task scope: regression and binary classification.
- Scalability direction: CPU-first local workflow now, with a future path to cloud GPU execution (RAPIDS `cudf`/`cuml`) without rewriting the full pipeline.

## Development Defaults
The repository is developed and manually verified primarily against these Playground Series competitions:
- default classification development target: `playground-series-s5e12` with `primary_metric: roc_auc`
- classification production target: `playground-series-s6e3` with `primary_metric: roc_auc`
- default regression target: `playground-series-s5e10` with `primary_metric: mse`

## Current Capabilities
- Load and validate a single local repository-root `config.yaml`.
- Fetch Kaggle competition data into `data/<competition_slug>/` when the zip is missing.
- Require explicit `task_type` and `primary_metric` in config.
- Select one or more baseline or booster model recipes per run from config via `model_ids`, with `model_id` retained as the single-model shorthand.
- Ship tracked binary and regression example configs that can be copied into the local runtime config.
- Infer Playground-style submission schema from dataset files:
  - `id_column` as the only column shared by `train.csv`, `test.csv`, and `sample_submission.csv`
  - `label_column` as the only column shared by `train.csv` and `sample_submission.csv` but not `test.csv`
- Exclude the resolved `id_column` from modeled features by default; identifier columns are treated as metadata, not training signal.
- Generate terminal and CSV EDA summaries under `reports/<competition_slug>/`, including missingness, categorical cardinality, target summary, and feature-type counts.
- Run stage-specific CLI entrypoints for `fetch`, `eda`, `preprocess`, `train`, and submit-only flows against explicit run artifacts.
- Train one or more cross-validated model recipes with fold-local, model-specific preprocessing:
  - `onehot` preprocessing: `onehot_ridge`, `onehot_elasticnet`, `onehot_logreg`
  - `ordinal` preprocessing: `ordinal_randomforest`, `ordinal_extratrees`, `ordinal_hgb`, `ordinal_lightgbm`, `ordinal_xgboost`
  - `native` preprocessing: `native_catboost`
- Write a run-root `model_summary.csv`, task-aware run diagnostics, and a canonical `run_manifest.json` under `artifacts/<competition_slug>/train/<run_id>/`, with per-model prediction artifacts under `<run_id>/<model_id>/`.
- Validate predictions against `sample_submission.csv`, including exact ID content and order, with task-aware binary prediction checks, and optionally submit to Kaggle from the best model in the run or an explicitly selected model artifact.

## Tooling
- Python for orchestration
- Kaggle CLI for competition data and submissions
- `gh` CLI for repository management
- `uv` for environment management

## Quickstart
1. Ensure Kaggle CLI access is already configured for your user.
2. Install dependencies with `uv sync`.
3. If you want LightGBM, CatBoost, or XGBoost model recipes, install the optional booster dependencies with `uv sync --extra boosters`.
4. Copy a tracked example config to a local repository-root `config.yaml`.
5. Run `uv run python main.py`.

```bash
cp config.binary.example.yaml config.yaml
# or
cp config.regression.example.yaml config.yaml
```

`config.yaml` is the only runtime config source. It is intentionally ignored by Git so you can keep local competition-specific settings without committing them.

The current pipeline fetches competition data if needed, runs config-aware EDA, trains one or more CV model recipes with fold-local preprocessing and task-aware diagnostics, writes prediction artifacts, and prepares a validated submission file.

## Stage Commands
`uv run python main.py` still runs the full default pipeline: fetch, EDA, train, and submit.

Available stage-specific commands:
- `uv run python main.py fetch`
- `uv run python main.py eda`
- `uv run python main.py preprocess`
- `uv run python main.py train`
- `uv run python main.py submit --run-dir artifacts/<competition_slug>/train/<run_id>`
- `uv run python main.py submit --run-id <run_id>`

Stage behavior:
- `fetch`: ensures competition data is present locally
- `eda`: fetches if needed, then writes EDA report CSVs
- `preprocess`: fetches if needed, then writes preprocessing diagnostics under `reports/<competition_slug>/`
- `train`: fetches if needed, then trains and writes normal training artifacts
- `submit`: requires an explicit existing run selection and never retrains implicitly

The `preprocess` stage is a diagnostic/export path, not a separate required step in the normal runtime contract. It writes:
- `preprocess_summary.csv`
- `preprocess_features.csv`
- `preprocess_models.csv`

`submit` can take an optional `--model-id` when targeting a specific model artifact from a multi-model run.

## Config Overview
Tracked example configs:
- `config.binary.example.yaml`: binary-classification starting point using the default dev smoke-test target
- `config.regression.example.yaml`: regression starting point using the default regression smoke-test target

Runtime config workflow:
- copy one of the tracked examples to repository-root `config.yaml`
- edit only `config.yaml` for local runs
- keep `config.yaml` untracked; the application does not read the example files directly

Required keys:
- `competition_slug`
- `task_type`: `regression` or `binary`
- `primary_metric`: one of `rmse`, `mse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, `accuracy`

Optional binary-classification key:
- `positive_label`: explicit positive class for binary competitions; required unless the observed training labels follow one of the documented safe conventions: `[0, 1]`, `[False, True]`, or `["No", "Yes"]`

Optional model-selection key:
- `model_ids`: ordered list of baseline model recipes for the configured task
- `model_id`: single-model shorthand retained for backward compatibility; mutually exclusive with `model_ids`
  - regression: `onehot_ridge`, `onehot_elasticnet` (default), `ordinal_randomforest`, `ordinal_extratrees`, `ordinal_hgb`, `ordinal_lightgbm`, `native_catboost`, `ordinal_xgboost`
  - binary classification: `onehot_logreg` (default), `ordinal_randomforest`, `ordinal_extratrees`, `ordinal_hgb`, `ordinal_lightgbm`, `native_catboost`, `ordinal_xgboost`
  - compatibility aliases accepted during transition: `elasticnet`, `logistic_regression`, `random_forest`, `lightgbm`, `catboost`, `xgb`

Optional submission schema keys:
- `id_column`: override for the inferred identifier column; the resolved ID column is excluded from modeled features by default
- `label_column`: override for the inferred submission/target column

Optional preprocessing keys:
- `force_categorical`: list of feature names to force into the categorical pipeline
- `force_numeric`: list of feature names to force into the numeric pipeline
- `drop_columns`: additional feature names to remove before preprocessing after the ID column is already excluded by default
- `low_cardinality_int_threshold`: integer columns at or below this unique-count threshold are treated as categorical by default

Model preprocessing notes:
- `native_catboost` preserves a pandas feature frame and uses CatBoost native categorical handling.
- `ordinal_lightgbm` and `ordinal_xgboost` reuse the repository ordinal categorical path.

Optional CV keys:
- `cv_n_splits`: number of CV folds (default `7`)
- `cv_shuffle`: whether to shuffle before splitting (default `true`)
- `cv_random_state`: random seed for deterministic folds (default `42`)

Optional submission keys:
- `submit_enabled`: if `true`, submit to Kaggle after training (default `false`)
- `submit_message_prefix`: optional prefix used in auto-generated submission messages

Binary prediction artifact contract:
- `roc_auc` and `log_loss`: `test_predictions.csv` and `submission.csv` contain positive-class probabilities in `[0, 1]`
- `accuracy`: `test_predictions.csv` and `submission.csv` contain predicted class labels from the observed binary label set

If `id_column` or `label_column` are omitted, the training pipeline infers them from `train.csv`, `test.csv`, and `sample_submission.csv`. The resolved `id_column` is preserved for prediction outputs and in `run_manifest.json`, but it is not part of the model feature matrix. Submission preparation consumes the selected run manifest as the schema/task source of truth and uses `sample_submission.csv` only for validation. Invalid overrides, ambiguous inference, a `sample_submission.csv` shape that does not exactly match `[id_column, label_column]`, or a submission ID column that differs from `sample_submission.csv` in values or ordering are hard errors.

`model_ids` is the preferred config-driven model interface. If both `model_id` and `model_ids` are omitted, the workflow selects the current default recipe for the configured task: `onehot_elasticnet` for regression and `onehot_logreg` for binary classification. If `model_id` is provided, it resolves to a single-entry `model_ids` list. Backward-compatible aliases are normalized to the canonical preprocessing-first IDs during config loading, so run artifacts always use the canonical IDs.

`task_type` and `primary_metric` are always config-driven. The pipeline does not infer them from Kaggle metadata.

## Preferred Manual Verification Targets
Use these Playground competitions as the primary smoke tests:
- binary classification (dev): `playground-series-s5e12` with `task_type: binary` and `primary_metric: roc_auc`
- binary classification (prod target): `playground-series-s6e3` with `task_type: binary` and `primary_metric: roc_auc`
- regression: `playground-series-s5e10` with `task_type: regression` and `primary_metric: mse`

Tracked example configs for those targets:
- `config.binary.example.yaml`
- `config.regression.example.yaml`

Manual verification for each target:
- copy the corresponding example file to `config.yaml`
- confirm the competition archive includes `train.csv`, `test.csv`, and `sample_submission.csv`
- confirm the pipeline infers `id_column` and `label_column` without overrides
- confirm `artifacts/<competition_slug>/train/<run_id>/model_summary.csv` is written
- confirm `artifacts/<competition_slug>/train/<run_id>/<model_id>/test_predictions.csv` is written for each selected model
- confirm `artifacts/<competition_slug>/train/<run_id>/<model_id>/submission.csv` is written and validated against `sample_submission.csv`, including exact ID values and order, for the submitted model
- confirm binary outputs match the configured metric contract: probabilities for `roc_auc`/`log_loss`, labels for `accuracy`

## Outputs
- Competition data: `data/<competition_slug>/`
- EDA reports: `reports/<competition_slug>/`
  - when `preprocess` stage is run, also includes `preprocess_summary.csv`, `preprocess_features.csv`, and `preprocess_models.csv`
- Training artifacts: `artifacts/<competition_slug>/train/<run_id>/`
  - includes `run_diagnostics.csv`, `model_summary.csv`, and `run_manifest.json`
  - includes per-model subdirectories `artifacts/<competition_slug>/train/<run_id>/<model_id>/`
  - each model subdirectory includes `fold_metrics.csv`, `oof_predictions.csv`, `test_predictions.csv`, and `submission.csv` when prepared or submitted
  - `run_manifest.json` is the canonical per-run metadata source
  - each model summary/manifest entry records the resolved `preprocessing_scheme_id`
- Run ledger: `artifacts/<competition_slug>/train/runs.csv` as a compact comparison/history table
- Submission ledger: `artifacts/<competition_slug>/train/submissions.csv` as an append-only submission event table

## Current Assumptions
- Kaggle CLI is installed, authenticated, and has access to the configured competition.
- Competition zip contents include `train.csv`, `test.csv`, and `sample_submission.csv`.
- The competition follows a simple two-column Playground submission contract: `sample_submission.csv` must be exactly `[id_column, label_column]`.
- The resolved `id_column` is identifier metadata and is excluded from preprocessing and model fitting by default.
- Model IDs resolve to preprocessing-aware training recipes; run artifacts record the canonical `model_id` and `preprocessing_scheme_id`.
- Submission uses `run_manifest.json` as the canonical source for `competition_slug`, `task_type`, `id_column`, and `label_column`.
- Submission metadata includes the selected `model_id`; when no model is selected explicitly, submission defaults to the run manifest `best_model_id`.
- Submission validation requires the selected model artifact `test_predictions.csv[id_column]` to match `sample_submission.csv[id_column]` exactly in both values and row order.
- Binary classification supports any two-class labels accepted by scikit-learn.
- For binary `roc_auc` and `log_loss`, prediction artifacts use probabilities aligned to the resolved positive class.
- For binary `accuracy`, prediction artifacts use class labels from the observed binary label set.
- Binary classification requires an explicit positive-class contract. If `positive_label` is omitted, the workflow only auto-resolves the positive class for labels `[0, 1]`, `[False, True]`, or `["No", "Yes"]`; other two-class label pairs fail fast.
- `task_type` and `primary_metric` are explicitly configured for every run.
- Runtime config comes from local repository-root `config.yaml` only; tracked example files are just starting points, and there are no CLI or environment overrides.
- The current workflow is CPU-first and optimized for iteration speed over production hardening.
