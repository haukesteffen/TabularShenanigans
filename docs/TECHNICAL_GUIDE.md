# Technical Guide

Technical reference for the current repository design. Use GitHub issues and pull requests for active implementation tracking; this document describes the system as it exists and the contracts it is expected to preserve.

## System Flow
1. Load and validate the repository-root `config.yaml`.
2. Use explicit `task_type` and `primary_metric` from config.
3. Download the competition zip into `data/<competition_slug>/` when it is missing.
4. Read `train.csv`, `test.csv`, and `sample_submission.csv` from the zip as needed.
5. Run EDA and write report CSVs under `reports/<competition_slug>/`.
6. Resolve `id_column` and `label_column` from `train.csv`, `test.csv`, and `sample_submission.csv`, then prepare raw feature frames from the train/test data.
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
- `src/tabular_shenanigans/config.py`: Pydantic-backed config schema, metric normalization, and runtime contract validation.
- `src/tabular_shenanigans/data.py`: competition download, zip access, metric helpers, and dataset schema resolution.
- `src/tabular_shenanigans/eda.py`: competition-scan EDA summaries written to CSV, including missingness, categorical cardinality, target summary, and feature-type counts.
- `src/tabular_shenanigans/preprocess.py`: feature frame preparation, column typing, and sklearn preprocessing pipelines.
- `src/tabular_shenanigans/cv.py`: task-aware CV splitters and metric scoring helpers.
- `src/tabular_shenanigans/train.py`: baseline model selection, fold training, artifact writing, and run ledger updates.
- `src/tabular_shenanigans/submit.py`: submission schema validation, submission message creation, Kaggle submission, and submission ledger updates.

## Configuration Contract
Input:
- One config file: `config.yaml` (single source of truth)
- Required keys:
  - `competition_slug`
  - `task_type` (`regression` or `binary`)
  - `primary_metric` (`rmse`, `mse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, `accuracy`)
- Optional submission schema keys:
  - `id_column` (optional override for the inferred identifier column)
  - `label_column` (optional override for the inferred submission/target column)
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
Configured metrics are normalized to the internal metric names during config validation.

## Preferred Verification Targets
- `playground-series-s5e12`: binary smoke test with `task_type: binary` and `primary_metric: roc_auc`
- `playground-series-s5e10`: regression smoke test with `task_type: regression` and `primary_metric: mse`

Manual verification steps for each target:
- verify the competition assets include `train.csv`, `test.csv`, and `sample_submission.csv`
- run the workflow from a clean repo state with explicit `task_type` and `primary_metric`
- confirm inferred `id_column` and `label_column`
- confirm `test_predictions.csv` is generated in the run directory
- confirm `submission.csv` validates against `sample_submission.csv`

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
- `task_type` and `primary_metric` must be present in config for every run
- Kaggle CLI and authentication are expected to be preconfigured
- Competition zip contents are expected to include `train.csv`, `test.csv`, and `sample_submission.csv`
- `id_column` inference must resolve to exactly one column present in `train.csv`, `test.csv`, and `sample_submission.csv`
- `label_column` inference must resolve to exactly one column present in `train.csv` and `sample_submission.csv` but not `test.csv`
- `sample_submission.csv` must match the resolved schema exactly as `[id_column, label_column]`
- Feature override columns must exist and cannot overlap between forced numeric and forced categorical sets
- Configured metric must normalize to a supported metric compatible with the configured task type
- CV splitter construction must support both `cv_shuffle=true` and `cv_shuffle=false`
- Submission output must match the schema and row count of `sample_submission.csv`
- Fail fast is the default behavior; errors are surfaced directly with minimal wrapping

Hard-error cases include:
- Missing config file -> hard error
- Schema/type violation -> hard error
- Any attempt to use additional config sources -> hard error
- Kaggle command failure -> hard error (bubble up with minimal wrapping)
- Missing `task_type` or `primary_metric` -> hard error
- Unknown/unsupported configured `primary_metric` -> hard error
- Invalid task/metric pairing (for example `binary` + `rmse`) -> hard error
- Missing/invalid competition zip contents -> hard error
- `id_column` inference not exactly one column -> hard error
- `label_column` inference not exactly one column -> hard error
- Invalid `id_column` or `label_column` override -> hard error
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
- New metrics should be normalized and validated during config loading, then scored in `cv.py`.
- New model families should be introduced in `train.py` with explicit task compatibility and matching artifact outputs.
- New preprocessing modes should be added in `preprocess.py` without breaking the existing feature-frame contract.
- New run or submission artifacts should be reflected in both the artifact contract above and the corresponding ledger rows.
