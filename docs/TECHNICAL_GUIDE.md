# Technical Guide

Technical reference for the current repository design. Use GitHub issues and pull requests for active implementation tracking; this document describes the system as it exists and the contracts it is expected to preserve.

## System Flow
1. Load and validate the repository-root `config.yaml`.
2. Resolve `task_type` and `primary_metric` from config or Kaggle competition metadata.
3. Download the competition zip into `data/<competition_slug>/` when it is missing.
4. Read `train.csv`, `test.csv`, and `sample_submission.csv` from the zip as needed.
5. Run EDA and write report CSVs under `reports/<competition_slug>/`.
6. Prepare raw feature frames from the train/test data and infer the target column from the train/test schema difference.
7. Build preprocessing for the selected feature types:
   - numeric: median imputation + `StandardScaler`
   - categorical: most-frequent imputation + `OneHotEncoder`
8. Train the baseline model with fold-local preprocessing:
   - regression: `ElasticNet`
   - binary classification: `LogisticRegression`
9. Write fold metrics, CV summary, task-aware run diagnostics, OOF predictions, test predictions, and a run manifest under `artifacts/<competition_slug>/train/<run_id>/`.
10. Validate predictions against `sample_submission.csv`, write `submission.csv`, and optionally submit to Kaggle.

## Module Responsibilities
- `main.py`: orchestration entrypoint for config loading, data fetch, EDA, preprocessing, training, and submission.
- `src/tabular_shenanigans/config.py`: Pydantic-backed config schema and loading.
- `src/tabular_shenanigans/data.py`: Kaggle metadata lookup, competition download, zip access, and target inference.
- `src/tabular_shenanigans/eda.py`: competition-scan EDA summaries written to CSV, including missingness, categorical cardinality, target summary, and feature-type counts.
- `src/tabular_shenanigans/preprocess.py`: feature frame preparation, column typing, and sklearn preprocessing pipelines.
- `src/tabular_shenanigans/cv.py`: task-aware CV splitters and metric scoring helpers.
- `src/tabular_shenanigans/train.py`: baseline model selection, fold training, artifact writing, and run ledger updates.
- `src/tabular_shenanigans/submit.py`: submission schema validation, submission message creation, Kaggle submission, and submission ledger updates.

## Configuration Contract
Input:
- One config file: `config.yaml` (single source of truth)
- Required key: `competition_slug`
- Optional keys for competition metadata:
  - `task_type` (`regression` or `binary`)
  - `primary_metric` (`rmse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, `accuracy`)
- Optional keys for preprocessing:
  - `force_categorical` (list of column names)
  - `force_numeric` (list of column names)
  - `drop_columns` (list of column names)
  - `low_cardinality_int_threshold` (positive integer)
- Optional keys for CV:
  - `cv_n_splits` (integer >= 2, default 7)
  - `cv_shuffle` (boolean, default true)
  - `cv_random_state` (integer, default 42)
- Optional keys for submission:
  - `submit_enabled` (boolean, default false)
  - `submit_message_prefix` (string, optional)

The config is validated by Pydantic with `extra="forbid"`. Unknown keys, schema mismatches, and missing required fields are hard errors.

## Artifact Contract
- A validated in-memory config object from Pydantic
- Competition files downloaded under `data/<competition_slug>/`
- EDA summary printed to terminal
- EDA report CSV files under `reports/<competition_slug>/`
  - `columns_train.csv`
  - `columns_test.csv`
  - `missingness_summary.csv`
  - `categorical_cardinality_summary.csv`
  - `target_summary.csv`
  - `feature_type_counts.csv`
  - `run_summary.csv`
- Preprocessed feature/target CSV files under `artifacts/<competition_slug>/preprocess/`
- Training artifacts under `artifacts/<competition_slug>/train/<run_id>/`:
  - `fold_metrics.csv`
  - `cv_summary.csv`
  - `run_diagnostics.csv`
  - `oof_predictions.csv`
  - `test_predictions.csv`
  - `run_manifest.json`
- Training ledger at `artifacts/<competition_slug>/train/runs.csv` with task, metric, CV metadata, task-aware target summary fields, and artifact paths
- Submission artifact in each run dir:
  - `submission.csv`
- Append-only submission ledger at `artifacts/<competition_slug>/train/submissions.csv`

## Runtime Invariants And Failure Behavior
- One runtime config source only: `config.yaml`
- No config overrides via CLI or environment variables
- Kaggle CLI and authentication are expected to be preconfigured
- Competition zip contents are expected to include `train.csv`, `test.csv`, and `sample_submission.csv`
- Target inference must resolve to exactly one column
- Feature override columns must exist and cannot overlap between forced numeric and forced categorical sets
- Metric resolution must produce a supported metric compatible with the resolved task type
- CV splitter construction must support both `cv_shuffle=true` and `cv_shuffle=false`
- Submission output must match the schema and row count of `sample_submission.csv`
- Fail fast is the default behavior; errors are surfaced directly with minimal wrapping

Hard-error cases include:
- Missing config file -> hard error
- Schema/type violation -> hard error
- Any attempt to use additional config sources -> hard error
- Kaggle command failure -> hard error (bubble up with minimal wrapping)
- Missing exact competition metadata match for slug -> hard error
- Unknown/unsupported configured `primary_metric` -> hard error
- Partial task/metric inference from Kaggle metadata -> hard error
- Invalid task/metric pairing (for example `binary` + `rmse`) -> hard error
- Missing/invalid competition zip contents -> hard error
- Target inference not exactly one column -> hard error
- Unknown columns in `force_categorical`, `force_numeric`, or `drop_columns` -> hard error
- Any overlap between `force_categorical` and `force_numeric` -> hard error
- No feature columns remaining after `drop_columns` -> hard error
- Preprocessing fit/transform failure -> hard error
- Unsupported task type for CV/model selection -> hard error
- Unsupported metric for chosen task -> hard error
- Any CV/training fit or scoring failure -> hard error
- Fold assignment gaps in OOF generation -> hard error
- Submission schema mismatch against `sample_submission.csv` -> hard error
- Kaggle submission command failure when `submit_enabled=true` -> hard error

## Design Guardrails
- Keep implementation simple and avoid speculative abstractions.
- Stay CPU-first by default; treat GPU support as an optional later backend.
- Keep data and modeling logic behind small internal interfaces so backend swaps stay localized.
- Avoid scattering direct pandas/sklearn calls across many modules.
- Prefer columnar and vectorized transformations over row-wise Python loops.
- Use portable artifacts such as `csv` and `json`.
- Keep runs reproducible from config and append-only ledgers.
- Keep EDA script-driven and CSV-oriented rather than notebook- or plot-first.

## Extension Notes
- New config keys should be added to `AppConfig` in `config.py` and documented in both this file and `README.md` when user-facing.
- New metrics should be normalized and validated in `data.py`, then scored in `cv.py`.
- New model families should be introduced in `train.py` with explicit task compatibility and matching artifact outputs.
- New preprocessing modes should be added in `preprocess.py` without breaking the existing feature-frame contract.
- New run or submission artifacts should be reflected in both the artifact contract above and the corresponding ledger rows.
