# TabularShenanigans

Config-driven Python workflows for semi-automated participation in tabular Kaggle competitions.

## Project Overview
- Focus area: tabular machine learning competitions.
- Task scope: regression and binary classification.
- Development style: small, incremental iterations with detailed explanations.
- Scalability direction: CPU-first local workflow now, with a future path to cloud GPU execution (RAPIDS `cudf`/`cuml`) without rewriting the full pipeline.

## Current Development Mode (Functionality First)
- This is an active development process focused on shipping working behavior quickly.
- Engineering polish is intentionally deprioritized in this phase.
- Unit and integration tests are explicitly out of scope for now.
- Avoid broad defensive `try/except` wrapping; let failures surface during development unless handling is required to keep core flow usable.
- Refactoring for style, architecture hardening, and production-grade robustness are deferred to a later stabilization phase.

## Current MVP Status
- Step 1 (configuration pipeline) is complete.
- Step 2 (Kaggle competition data fetch) is complete.
- Step 3 (script-based exploratory data analysis) is complete.
- Step 4 (preprocessing and feature engineering) is complete.
- Step 5 (baseline training with CV) is complete.
- Current implementation priority is submission workflow integration.

## Tooling
- Python for orchestration
- Kaggle CLI for competition data and submissions
- `gh` CLI for repository management
- `uv` for environment management

## Quickstart (Current Stage)
1. Keep a project `config.yaml` at repository root.
2. Run the current Python entrypoint scripts directly from the repo.
3. Ensure Kaggle CLI access is already configured for your user.
4. Current run behavior: fetch competition zip if missing, generate EDA report CSVs, write preprocessing artifacts, run linear baseline CV training, then build a validated submission file from the latest run (validated against `sample_submission.csv`).
5. Follow iteration notes and current development-mode rules in [`docs/TECHNICAL_GUIDE.md`](docs/TECHNICAL_GUIDE.md).

### Optional preprocessing config
`config.yaml` supports optional feature typing overrides used by preprocessing:
- `force_categorical`: list of feature names to force into the categorical pipeline.
- `force_numeric`: list of feature names to force into the numeric pipeline.
- `drop_columns`: list of feature names to remove before preprocessing.
- `low_cardinality_int_threshold`: if set, integer columns with unique values at or below this threshold are treated as categorical by default.

### Optional competition metadata config
`config.yaml` also supports optional overrides for competition task + scoring:
- `task_type`: `regression` or `binary`
- `primary_metric`: one of `rmse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, `accuracy`

If either key is missing, the pipeline tries to infer missing values from Kaggle competition metadata.
If inference is partial or ambiguous, the run fails and requires explicit config values.

### Optional CV config
`config.yaml` also supports:
- `cv_n_splits`: number of CV folds (default `7`)
- `cv_shuffle`: whether to shuffle before splitting (default `true`)
- `cv_random_state`: random seed for deterministic folds (default `42`)

### Optional submission config
`config.yaml` also supports:
- `submit_enabled`: if `true`, submit to Kaggle after training (default `false`)
- `submit_message_prefix`: optional prefix used in auto-generated submission messages

When `submit_enabled` is `false`, the pipeline still validates submission schema and writes `submission.csv`, then logs a dry-run status.

## Roadmap
1. Robust config pipeline
2. Kaggle data fetch
3. Exploratory data analysis
4. Preprocessing and feature engineering
5. Baseline models
6. Model stacking for stronger CV/LB performance
