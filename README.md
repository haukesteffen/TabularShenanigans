# TabularShenanigans

Config-driven Python workflows for semi-automated participation in tabular Kaggle competitions.

## Project Overview
- Focus area: tabular machine learning competitions.
- Task scope: regression and binary classification.
- Scalability direction: CPU-first local workflow now, with a future path to cloud GPU execution (RAPIDS `cudf`/`cuml`) without rewriting the full pipeline.

## Current Capabilities
- Load and validate a single repository-root `config.yaml`.
- Fetch Kaggle competition data into `data/<competition_slug>/` when the zip is missing.
- Infer `task_type` and `primary_metric` from config or Kaggle metadata.
- Generate terminal and CSV EDA summaries under `reports/<competition_slug>/`.
- Build preprocessing artifacts under `artifacts/<competition_slug>/preprocess/`.
- Train baseline cross-validated models with fold-local preprocessing:
  - regression: `ElasticNet`
  - binary classification: `LogisticRegression`
- Write fold metrics, CV summary, OOF predictions, test predictions, and a run manifest under `artifacts/<competition_slug>/train/<run_id>/`.
- Validate predictions against `sample_submission.csv` and optionally submit to Kaggle.

## Tooling
- Python for orchestration
- Kaggle CLI for competition data and submissions
- `gh` CLI for repository management
- `uv` for environment management

## Quickstart
1. Ensure Kaggle CLI access is already configured for your user.
2. Install dependencies with `uv sync`.
3. Keep a project `config.yaml` at repository root.
4. Run `uv run python main.py`.

The current pipeline fetches competition data if needed, runs EDA, writes preprocessing artifacts, trains a baseline CV model, writes prediction artifacts, and prepares a validated submission file.

## Config Overview
Required key:
- `competition_slug`

Optional competition metadata keys:
- `task_type`: `regression` or `binary`
- `primary_metric`: one of `rmse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, `accuracy`

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

If competition metadata keys are omitted, the pipeline attempts inference from Kaggle metadata. Partial or ambiguous inference fails fast and requires explicit values in `config.yaml`.

## Outputs
- Competition data: `data/<competition_slug>/`
- EDA reports: `reports/<competition_slug>/`
- Preprocessing artifacts: `artifacts/<competition_slug>/preprocess/`
- Training artifacts: `artifacts/<competition_slug>/train/<run_id>/`
- Run ledger: `artifacts/<competition_slug>/train/runs.csv`
- Submission ledger: `artifacts/<competition_slug>/train/submissions.csv`

## Current Assumptions
- Kaggle CLI is installed, authenticated, and has access to the configured competition.
- Competition zip contents include `train.csv`, `test.csv`, and `sample_submission.csv`.
- A single target column can be inferred from the train/test schema difference.
- Runtime config comes from `config.yaml` only; there are no CLI or environment overrides.
- The current workflow is CPU-first and optimized for iteration speed over production hardening.
