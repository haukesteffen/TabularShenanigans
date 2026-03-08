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
- Load and validate a single repository-root `config.yaml`.
- Fetch Kaggle competition data into `data/<competition_slug>/` when the zip is missing.
- Require explicit `task_type` and `primary_metric` in config.
- Infer Playground-style submission schema from dataset files:
  - `id_column` as the only column shared by `train.csv`, `test.csv`, and `sample_submission.csv`
  - `label_column` as the only column shared by `train.csv` and `sample_submission.csv` but not `test.csv`
- Exclude the resolved `id_column` from modeled features by default; identifier columns are treated as metadata, not training signal.
- Generate terminal and CSV EDA summaries under `reports/<competition_slug>/`, including missingness, categorical cardinality, target summary, and feature-type counts.
- Train baseline cross-validated models with fold-local preprocessing:
  - regression: `ElasticNet`
  - binary classification: `LogisticRegression`
- Write fold metrics, CV summary, task-aware run diagnostics, OOF predictions, test predictions, and a run manifest under `artifacts/<competition_slug>/train/<run_id>/`.
- Validate predictions against `sample_submission.csv`, including exact ID content and order, and optionally submit to Kaggle.

## Tooling
- Python for orchestration
- Kaggle CLI for competition data and submissions
- `gh` CLI for repository management
- `uv` for environment management

## Quickstart
1. Ensure Kaggle CLI access is already configured for your user.
2. Install dependencies with `uv sync`.
3. Keep a project `config.yaml` at repository root with explicit `competition_slug`, `task_type`, and `primary_metric`.
4. Run `uv run python main.py`.

The current pipeline fetches competition data if needed, runs config-aware EDA, trains a baseline CV model with fold-local preprocessing and task-aware diagnostics, writes prediction artifacts, and prepares a validated submission file.

## Config Overview
Required keys:
- `competition_slug`
- `task_type`: `regression` or `binary`
- `primary_metric`: one of `rmse`, `mse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, `accuracy`

Optional binary-classification key:
- `positive_label`: explicit positive class for binary competitions; required unless the observed training labels follow one of the documented safe conventions: `[0, 1]`, `[False, True]`, or `["No", "Yes"]`

Optional submission schema keys:
- `id_column`: override for the inferred identifier column; the resolved ID column is excluded from modeled features by default
- `label_column`: override for the inferred submission/target column

Optional preprocessing keys:
- `force_categorical`: list of feature names to force into the categorical pipeline
- `force_numeric`: list of feature names to force into the numeric pipeline
- `drop_columns`: additional feature names to remove before preprocessing after the ID column is already excluded by default
- `low_cardinality_int_threshold`: integer columns at or below this unique-count threshold are treated as categorical by default

Optional CV keys:
- `cv_n_splits`: number of CV folds (default `7`)
- `cv_shuffle`: whether to shuffle before splitting (default `true`)
- `cv_random_state`: random seed for deterministic folds (default `42`)

Optional submission keys:
- `submit_enabled`: if `true`, submit to Kaggle after training (default `false`)
- `submit_message_prefix`: optional prefix used in auto-generated submission messages

If `id_column` or `label_column` are omitted, the pipeline infers them from `train.csv`, `test.csv`, and `sample_submission.csv`. The resolved `id_column` is preserved for prediction outputs and submission validation, but it is not part of the model feature matrix. Invalid overrides, ambiguous inference, a `sample_submission.csv` shape that does not exactly match `[id_column, label_column]`, or a submission ID column that differs from `sample_submission.csv` in values or ordering are hard errors.

`task_type` and `primary_metric` are always config-driven. The pipeline does not infer them from Kaggle metadata.

## Preferred Manual Verification Targets
Use these Playground competitions as the primary smoke tests:
- binary classification (dev): `playground-series-s5e12` with `task_type: binary` and `primary_metric: roc_auc`
- binary classification (prod target): `playground-series-s6e3` with `task_type: binary` and `primary_metric: roc_auc`
- regression: `playground-series-s5e10` with `task_type: regression` and `primary_metric: mse`

Example binary config:

```yaml
competition_slug: playground-series-s5e12
task_type: binary
primary_metric: roc_auc
```

Example regression config:

```yaml
competition_slug: playground-series-s5e10
task_type: regression
primary_metric: mse
```

Manual verification for each target:
- confirm the competition archive includes `train.csv`, `test.csv`, and `sample_submission.csv`
- confirm the pipeline infers `id_column` and `label_column` without overrides
- confirm `artifacts/<competition_slug>/train/<run_id>/test_predictions.csv` is written
- confirm `artifacts/<competition_slug>/train/<run_id>/submission.csv` is written and validated against `sample_submission.csv`, including exact ID values and order

## Outputs
- Competition data: `data/<competition_slug>/`
- EDA reports: `reports/<competition_slug>/`
- Training artifacts: `artifacts/<competition_slug>/train/<run_id>/`
  - includes `fold_metrics.csv`, `cv_summary.csv`, `run_diagnostics.csv`, `oof_predictions.csv`, `test_predictions.csv`, and `run_manifest.json`
- Run ledger: `artifacts/<competition_slug>/train/runs.csv`
- Submission ledger: `artifacts/<competition_slug>/train/submissions.csv`

## Current Assumptions
- Kaggle CLI is installed, authenticated, and has access to the configured competition.
- Competition zip contents include `train.csv`, `test.csv`, and `sample_submission.csv`.
- The competition follows a simple two-column Playground submission contract: `sample_submission.csv` must be exactly `[id_column, label_column]`.
- The resolved `id_column` is identifier metadata and is excluded from preprocessing and model fitting by default.
- Submission validation requires `test_predictions.csv[id_column]` to match `sample_submission.csv[id_column]` exactly in both values and row order.
- Binary classification supports any two-class labels accepted by scikit-learn; probability outputs are aligned to the resolved positive class.
- Binary classification requires an explicit positive-class contract. If `positive_label` is omitted, the workflow only auto-resolves the positive class for labels `[0, 1]`, `[False, True]`, or `["No", "Yes"]`; other two-class label pairs fail fast.
- `task_type` and `primary_metric` are explicitly configured for every run.
- Runtime config comes from `config.yaml` only; there are no CLI or environment overrides.
- The current workflow is CPU-first and optimized for iteration speed over production hardening.
