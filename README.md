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
- Separate stable competition settings from the current experiment candidate in config via top-level `competition` and `experiment` sections.
- Ship tracked binary and regression example configs that can be copied into the local runtime config.
- Infer Playground-style submission schema from dataset files:
  - `id_column` as the only column shared by `train.csv`, `test.csv`, and `sample_submission.csv`
  - `label_column` as the only column shared by `train.csv` and `sample_submission.csv` but not `test.csv`
- Exclude the resolved `id_column` from modeled features by default; identifier columns are treated as metadata, not training signal.
- Generate terminal and CSV EDA summaries under `reports/<competition_slug>/`, including missingness, categorical cardinality, target summary, and feature-type counts.
- Run stage-specific CLI entrypoints for `fetch`, `eda`, `preprocess`, `train`, and submit-only flows against explicit run artifacts.
- Run an explicit `tune` stage that evaluates Optuna trials for the current experiment candidate when `experiment.candidate.optimization.enabled=true`, writes study artifacts, and retrains the best trial into the standard training artifact layout.
- Train one cross-validated model candidate at a time using config-selected preprocessing and model-family choices that resolve internally to the current canonical recipe IDs.
- Write a run-root `model_summary.csv`, task-aware run diagnostics, and a canonical `run_manifest.json` under `artifacts/<competition_slug>/train/<run_id>/`, with per-model prediction artifacts under `<run_id>/<model_id>/`.
- Write tuning artifacts under `artifacts/<competition_slug>/tune/<study_id>/`, including `study_manifest.json`, `study_summary.csv`, `trials.csv`, and `best_params.json`.
- Validate predictions against `sample_submission.csv`, including exact ID content and order, with task-aware binary prediction checks, and optionally submit to Kaggle from the best model in the run or an explicitly selected model artifact.

## Tooling
- Python for orchestration
- Kaggle CLI for competition data and submissions
- Optuna for local hyperparameter tuning
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

The current pipeline fetches competition data if needed, runs config-aware EDA, trains the current experiment candidate with fold-local preprocessing and task-aware diagnostics, writes prediction artifacts, and prepares a validated submission file.

## Stage Commands
`uv run python main.py` still runs the full default pipeline: fetch, EDA, train, and submit.

Available stage-specific commands:
- `uv run python main.py fetch`
- `uv run python main.py eda`
- `uv run python main.py preprocess`
- `uv run python main.py train`
- `uv run python main.py tune`
- `uv run python main.py submit --run-dir artifacts/<competition_slug>/train/<run_id>`
- `uv run python main.py submit --run-id <run_id>`

Stage behavior:
- `fetch`: ensures competition data is present locally
- `eda`: fetches if needed, then writes EDA report CSVs
- `preprocess`: fetches if needed, then writes preprocessing diagnostics under `reports/<competition_slug>/`
- `train`: fetches if needed, then trains and writes normal training artifacts
- `tune`: fetches if needed, runs an Optuna study for the current experiment candidate when `experiment.candidate.optimization.enabled=true`, writes tuning artifacts, then retrains the best trial into the normal training artifact layout
- `submit`: requires an explicit existing run selection and never retrains implicitly

The `preprocess` stage is a diagnostic/export path, not a separate required step in the normal runtime contract. It writes:
- `preprocess_summary.csv`
- `preprocess_features.csv`
- `preprocess_models.csv`

`submit` can take an optional `--model-id` when targeting a specific model artifact from a multi-model run.
The default submit path supports current manifest-backed run artifacts only. Older local artifact layouts are unsupported and fail with direct errors.

## Config Overview
Tracked example configs:
- `config.binary.example.yaml`: binary-classification starting point using the default dev smoke-test target
- `config.regression.example.yaml`: regression starting point using the default regression smoke-test target

Runtime config workflow:
- copy one of the tracked examples to repository-root `config.yaml`
- edit only `config.yaml` for local runs
- keep `config.yaml` untracked; the application does not read the example files directly

The flat config layout is no longer supported. `config.yaml` must now contain top-level `competition` and `experiment` sections.

Required top-level sections:
- `competition`
- `experiment`

`competition` keys:
- `slug`
- `task_type`: `regression` or `binary`
- `primary_metric`: one of `rmse`, `mse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, `accuracy`
- optional `positive_label`: explicit positive class for binary competitions; required unless the observed training labels follow one of the documented safe conventions: `[0, 1]`, `[False, True]`, or `["No", "Yes"]`
- optional schema overrides:
  - `id_column`
  - `label_column`
- `cv`:
  - `n_splits` (default `7`)
  - `shuffle` (default `true`)
  - `random_state` (default `42`)
- optional `features` block:
  - `force_categorical`
  - `force_numeric`
  - `drop_columns`
  - `low_cardinality_int_threshold`

`experiment` keys:
- `name`
- optional `notes`
- required `candidate` block
- optional `submit` block

Current `experiment.candidate` contract:
- `candidate_type`: currently only `model` is supported
- `candidate_id`
- `preprocessor`: `onehot`, `ordinal`, or `native`
- `model_family`
  - regression: `ridge`, `elasticnet`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `lightgbm`, `catboost`, `xgboost`
  - binary classification: `logistic_regression`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `lightgbm`, `catboost`, `xgboost`
- optional `model_params`
- optional `optimization`:
  - `enabled`
  - `method`: currently only `optuna`
  - `n_trials`
  - `timeout_seconds`
  - `random_state`

Supported `model_family + preprocessor` combinations:
- regression: `ridge + onehot`, `elasticnet + onehot`, `random_forest + ordinal`, `extra_trees + ordinal`, `hist_gradient_boosting + ordinal`, `lightgbm + ordinal`, `catboost + native`, `xgboost + ordinal`
- binary classification: `logistic_regression + onehot`, `random_forest + ordinal`, `extra_trees + ordinal`, `hist_gradient_boosting + ordinal`, `lightgbm + ordinal`, `catboost + native`, `xgboost + ordinal`

Current `experiment.submit` keys:
- `enabled`: if `true`, submit to Kaggle after training (default `false`)
- `message_prefix`: optional prefix used in auto-generated submission messages

Binary prediction artifact contract:
- `roc_auc` and `log_loss`: `test_predictions.csv` and `submission.csv` contain positive-class probabilities in `[0, 1]`
- `accuracy`: `test_predictions.csv` and `submission.csv` contain predicted class labels from the observed binary label set

If `id_column` or `label_column` are omitted, the training pipeline infers them from `train.csv`, `test.csv`, and `sample_submission.csv`. The resolved `id_column` is preserved for prediction outputs and in `run_manifest.json`, but it is not part of the model feature matrix. Submission preparation consumes the selected run manifest as the schema/task source of truth and uses `sample_submission.csv` only for validation. Invalid overrides, ambiguous inference, a `sample_submission.csv` shape that does not exactly match `[id_column, label_column]`, or a submission ID column that differs from `sample_submission.csv` in values or ordering are hard errors.

The current runtime resolves `experiment.candidate.model_family + experiment.candidate.preprocessor` into one internal canonical `model_id` so the existing train, tune, and submit paths can continue working during the transition to the broader candidate-centric workflow.

`tune` uses the current experiment candidate only. When `experiment.candidate.optimization.enabled=true`, the tune stage evaluates that candidate, writes study artifacts, and retrains the best trial into the normal single-model training artifact layout.

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
- confirm `artifacts/<competition_slug>/train/<run_id>/<model_id>/test_predictions.csv` is written for the resolved internal model artifact
- confirm `artifacts/<competition_slug>/train/<run_id>/<model_id>/submission.csv` is written and validated against `sample_submission.csv`, including exact ID values and order, for the submitted model
- confirm binary outputs match the configured metric contract: probabilities for `roc_auc`/`log_loss`, labels for `accuracy`

Manual verification for tuning:
- run `uv run python main.py tune` with `experiment.candidate.optimization.enabled: true` and at least one stopping condition
- supported tunable combinations are:
  - binary: `logistic_regression + onehot`, `random_forest + ordinal`, `extra_trees + ordinal`, `hist_gradient_boosting + ordinal`, `lightgbm + ordinal`, `catboost + native`, `xgboost + ordinal`
  - regression: `random_forest + ordinal`, `extra_trees + ordinal`, `hist_gradient_boosting + ordinal`, `lightgbm + ordinal`, `catboost + native`, `xgboost + ordinal`
- confirm `artifacts/<competition_slug>/tune/<study_id>/study_manifest.json` is written
- confirm `artifacts/<competition_slug>/tune/<study_id>/trials.csv` records trial state, score, and params
- confirm `artifacts/<competition_slug>/train/<run_id>/run_manifest.json` records `tuning_provenance`

## Outputs
- Competition data: `data/<competition_slug>/`
- EDA reports: `reports/<competition_slug>/`
  - when `preprocess` stage is run, also includes `preprocess_summary.csv`, `preprocess_features.csv`, and `preprocess_models.csv`
- Tuning artifacts: `artifacts/<competition_slug>/tune/<study_id>/`
  - includes `study_manifest.json`, `study_summary.csv`, `trials.csv`, and `best_params.json`
- Training artifacts: `artifacts/<competition_slug>/train/<run_id>/`
  - includes `run_diagnostics.csv`, `model_summary.csv`, and `run_manifest.json`
  - includes per-model subdirectories `artifacts/<competition_slug>/train/<run_id>/<model_id>/`
  - each model subdirectory includes `fold_metrics.csv`, `oof_predictions.csv`, `test_predictions.csv`, and `submission.csv` when prepared or submitted
  - `run_manifest.json` is the canonical per-run metadata source
  - each model summary/manifest entry records the resolved `preprocessing_scheme_id`
  - tuned retrain runs also record `tuning_provenance`
- Run ledger: `artifacts/<competition_slug>/train/runs.csv` as a compact comparison/history table
- Submission ledger: `artifacts/<competition_slug>/train/submissions.csv` as an append-only submission event table

## Current Assumptions
- Kaggle CLI is installed, authenticated, and has access to the configured competition.
- Competition zip contents include `train.csv`, `test.csv`, and `sample_submission.csv`.
- The competition follows a simple two-column Playground submission contract: `sample_submission.csv` must be exactly `[id_column, label_column]`.
- The resolved `id_column` is identifier metadata and is excluded from preprocessing and model fitting by default.
- `config.yaml` must use top-level `competition` and `experiment` sections; the old flat layout is unsupported.
- The current runtime resolves `experiment.candidate.model_family + experiment.candidate.preprocessor` to one canonical internal `model_id`; run artifacts still record that resolved `model_id` and `preprocessing_scheme_id`.
- The `tune` stage uses the current experiment candidate only, and the tuned best-trial retrain writes a standard single-model train artifact with `tuning_provenance`.
- Submission uses `run_manifest.json` as the canonical source for `competition_slug`, `task_type`, `id_column`, and `label_column`.
- Submission metadata includes the selected `model_id`; when no model is selected explicitly, submission defaults to the run manifest `best_model_id`.
- Submission validation requires the selected model artifact `test_predictions.csv[id_column]` to match `sample_submission.csv[id_column]` exactly in both values and row order.
- Submission requires the current per-model artifact layout under `artifacts/<competition_slug>/train/<run_id>/<model_id>/`.
- Binary classification supports any two-class labels accepted by scikit-learn.
- For binary `roc_auc` and `log_loss`, prediction artifacts use probabilities aligned to the resolved positive class.
- For binary `accuracy`, prediction artifacts use class labels from the observed binary label set.
- Binary classification requires an explicit positive-class contract. If `positive_label` is omitted, the workflow only auto-resolves the positive class for labels `[0, 1]`, `[False, True]`, or `["No", "Yes"]`; other two-class label pairs fail fast.
- `task_type` and `primary_metric` are explicitly configured for every run.
- Runtime config comes from local repository-root `config.yaml` only; tracked example files are just starting points, and there are no CLI or environment overrides.
- Tuning search spaces live in code next to model definitions; there is no YAML search-space DSL.
- The current workflow is CPU-first and optimized for iteration speed over production hardening.
