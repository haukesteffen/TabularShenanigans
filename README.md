# TabularShenanigans

Config-driven Python workflows for semi-automated participation in tabular Kaggle competitions.

## Project Overview
- Focus area: tabular machine learning competitions.
- Task scope: regression and binary classification.
- Scalability direction: CPU-first local workflow now, with a future path to cloud GPU execution (RAPIDS `cudf`/`cuml`) without rewriting the full pipeline.

## Current Capabilities
- Load and validate a single repository-root `config.yaml`.
- Fetch Kaggle competition data into `data/<competition_slug>/` when the zip is missing.
- Require explicit `task_type` and `primary_metric` in config.
- Infer Playground-style submission schema from dataset files:
  - `id_column` as the only column shared by `train.csv`, `test.csv`, and `sample_submission.csv`
  - `label_column` as the only column shared by `train.csv` and `sample_submission.csv` but not `test.csv`
- Generate terminal and CSV EDA summaries under `reports/<competition_slug>/`, including missingness, categorical cardinality, target summary, and feature-type counts.
- Build preprocessing artifacts under `artifacts/<competition_slug>/preprocess/`.
- Train baseline cross-validated models with fold-local preprocessing:
  - regression: `ElasticNet`
  - binary classification: `LogisticRegression`
- Write fold metrics, CV summary, task-aware run diagnostics, OOF predictions, test predictions, and a run manifest under `artifacts/<competition_slug>/train/<run_id>/`.
- Validate predictions against `sample_submission.csv` and optionally submit to Kaggle.

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

The current pipeline fetches competition data if needed, runs config-aware EDA, writes preprocessing artifacts, trains a baseline CV model with task-aware diagnostics, writes prediction artifacts, and prepares a validated submission file.

## Config Overview
Required keys:
- `competition_slug`
- `task_type`: `regression` or `binary`
- `primary_metric`: one of `rmse`, `mse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, `accuracy`

Optional submission schema keys:
- `id_column`: override for the inferred identifier column
- `label_column`: override for the inferred submission/target column

Optional preprocessing keys:
- `force_categorical`: list of feature names to force into the categorical pipeline
- `force_numeric`: list of feature names to force into the numeric pipeline
- `drop_columns`: list of feature names to remove before preprocessing
- `low_cardinality_int_threshold`: integer columns at or below this unique-count threshold are treated as categorical by default

Optional CV keys:
- `cv_n_splits`: number of CV folds (default `7`)
- `cv_shuffle`: whether to shuffle before splitting (default `true`)
- `cv_random_state`: random seed for deterministic folds (default `42`)

Optional submission keys:
- `submit_enabled`: if `true`, submit to Kaggle after training (default `false`)
- `submit_message_prefix`: optional prefix used in auto-generated submission messages

If `id_column` or `label_column` are omitted, the pipeline infers them from `train.csv`, `test.csv`, and `sample_submission.csv`. Invalid overrides, ambiguous inference, or a `sample_submission.csv` shape that does not exactly match `[id_column, label_column]` are hard errors.

`task_type` and `primary_metric` are always config-driven. The pipeline does not infer them from Kaggle metadata.

## Preferred Manual Verification Targets
Use these Playground competitions as the primary smoke tests:
- binary classification: `playground-series-s5e12` with `task_type: binary` and `primary_metric: roc_auc`
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
- confirm `artifacts/<competition_slug>/train/<run_id>/submission.csv` is written and validated against `sample_submission.csv`

## Outputs
- Competition data: `data/<competition_slug>/`
- EDA reports: `reports/<competition_slug>/`
- Preprocessing artifacts: `artifacts/<competition_slug>/preprocess/`
- Training artifacts: `artifacts/<competition_slug>/train/<run_id>/`
  - includes `fold_metrics.csv`, `cv_summary.csv`, `run_diagnostics.csv`, `oof_predictions.csv`, `test_predictions.csv`, and `run_manifest.json`
- Run ledger: `artifacts/<competition_slug>/train/runs.csv`
- Submission ledger: `artifacts/<competition_slug>/train/submissions.csv`

## Current Assumptions
- Kaggle CLI is installed, authenticated, and has access to the configured competition.
- Competition zip contents include `train.csv`, `test.csv`, and `sample_submission.csv`.
- The competition follows a simple two-column Playground submission contract: `sample_submission.csv` must be exactly `[id_column, label_column]`.
- `task_type` and `primary_metric` are explicitly configured for every run.
- Runtime config comes from `config.yaml` only; there are no CLI or environment overrides.
- The current workflow is CPU-first and optimized for iteration speed over production hardening.
