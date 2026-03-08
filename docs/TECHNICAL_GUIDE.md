# Technical Guide

Technical reference for the current repository design. Use GitHub issues and pull requests for active implementation tracking; this document describes the system as it exists and the contracts it is expected to preserve.

The intended operating scope is Kaggle Playground Series tabular competitions. Current default development targets are `playground-series-s5e12` for binary classification with `primary_metric: roc_auc` and `playground-series-s5e10` for regression with `primary_metric: mse`. The current binary production target is `playground-series-s6e3` with `primary_metric: roc_auc`.

## System Flow
1. Load and validate the repository-root `config.yaml`.
2. Use explicit `task_type` and `primary_metric` from config.
3. Download the competition zip into `data/<competition_slug>/` when it is missing.
4. Read `train.csv`, `test.csv`, and `sample_submission.csv` from the zip as needed.
5. Run EDA and write report CSVs under `reports/<competition_slug>/`.
6. Resolve `id_column` and `label_column` from `train.csv`, `test.csv`, and `sample_submission.csv`, then prepare raw feature frames from the train/test data with the resolved `id_column` excluded from modeled features.
7. During training, build fold-local preprocessing for the selected feature types:
   - numeric: median imputation + `StandardScaler`
   - categorical: most-frequent imputation + `OneHotEncoder`
8. Train the configured baseline model or models from the resolved `model_ids` list:
   - regression: `elasticnet` (`ElasticNet`) or `random_forest` (`RandomForestRegressor`)
   - binary classification: `logistic_regression` (`LogisticRegression`) or `random_forest` (`RandomForestClassifier`)
9. Write run-level diagnostics, `model_summary.csv`, and a canonical `run_manifest.json` under `artifacts/<competition_slug>/train/<run_id>/`.
10. Write per-model fold metrics, OOF predictions, and test predictions under `artifacts/<competition_slug>/train/<run_id>/<model_id>/`.
11. Validate predictions against `sample_submission.csv`, including exact ID content and order, using `run_manifest.json` as the submission metadata contract, write `submission.csv` in the selected model directory, and optionally submit to Kaggle.

## Module Responsibilities
- `main.py`: orchestration entrypoint for config loading, data fetch, EDA, training, and submission.
- `src/tabular_shenanigans/config.py`: Pydantic-backed config schema, metric normalization, and runtime contract validation.
- `src/tabular_shenanigans/data.py`: competition download, zip access, metric helpers, dataset schema resolution, and sample-submission template loading.
- `src/tabular_shenanigans/eda.py`: competition-scan EDA summaries written to CSV, including missingness, categorical cardinality, target summary, and feature-type counts.
- `src/tabular_shenanigans/models.py`: single-model registry and estimator construction for supported baseline presets.
- `src/tabular_shenanigans/preprocess.py`: feature frame preparation, column typing, and sklearn preprocessing pipelines.
- `src/tabular_shenanigans/cv.py`: task-aware CV splitters and metric scoring helpers.
- `src/tabular_shenanigans/train.py`: config-selected multi-model training, shared split handling, artifact writing, and run ledger updates.
- `src/tabular_shenanigans/submit.py`: submission schema validation, model-artifact selection, submission message creation, Kaggle submission, and submission ledger updates.

## Configuration Contract
Input:
- One config file: `config.yaml` (single source of truth)
- Required keys:
  - `competition_slug`
  - `task_type` (`regression` or `binary`)
  - `primary_metric` (`rmse`, `mse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, `accuracy`)
- Optional model-selection key:
  - `model_ids` (ordered list of baseline model presets for the configured task)
  - `model_id` (single-model shorthand; mutually exclusive with `model_ids`)
    - regression: `elasticnet`, `random_forest`
    - binary classification: `logistic_regression`, `random_forest`
- Optional binary-classification key:
  - `positive_label` (explicit positive class for binary competitions; required unless observed labels match one of the documented safe conventions `[0, 1]`, `[False, True]`, or `["No", "Yes"]`)
- Optional submission schema keys:
  - `id_column` (optional override for the inferred identifier column; resolved IDs are kept as metadata and excluded from modeled features)
  - `label_column` (optional override for the inferred submission/target column)
- Optional keys for preprocessing:
  - `force_categorical` (list of column names)
  - `force_numeric` (list of column names)
  - `drop_columns` (list of additional modeled-feature columns to remove after the ID column is excluded by default)
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
- `playground-series-s5e12`: binary development smoke test with `task_type: binary` and `primary_metric: roc_auc`
- `playground-series-s6e3`: binary production-target smoke test with `task_type: binary` and `primary_metric: roc_auc`
- `playground-series-s5e10`: regression smoke test with `task_type: regression` and `primary_metric: mse`

Manual verification steps for each target:
- verify the competition assets include `train.csv`, `test.csv`, and `sample_submission.csv`
- run the workflow from a clean repo state with explicit `task_type`, `primary_metric`, and one or more selected models
- confirm inferred `id_column` and `label_column`
- confirm `model_summary.csv` is generated in the run directory
- confirm `test_predictions.csv` is generated in each model directory
- confirm `submission.csv` validates against `sample_submission.csv`, including exact ID values and order, for the selected model directory

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
- Training artifacts under `artifacts/<competition_slug>/train/<run_id>/`:
  - `run_diagnostics.csv`
  - `model_summary.csv`
  - `run_manifest.json`
  - per-model subdirectories `artifacts/<competition_slug>/train/<run_id>/<model_id>/` containing:
    - `fold_metrics.csv`
    - `oof_predictions.csv`
    - `test_predictions.csv`
    - `submission.csv` when prepared or submitted
- `run_manifest.json` is the canonical per-run metadata source
- Training ledger at `artifacts/<competition_slug>/train/runs.csv` with compact comparison fields and task-aware target summary fields
- Append-only submission ledger at `artifacts/<competition_slug>/train/submissions.csv` with submission event metadata only

## Runtime Invariants And Failure Behavior
- One runtime config source only: `config.yaml`
- No config overrides via CLI or environment variables
- `task_type` and `primary_metric` must be present in config for every run
- `model_ids` is the preferred model-selection interface; `model_id` remains the single-model shorthand and the two keys are mutually exclusive
- `model_ids` must resolve to one or more supported presets for the configured task; if omitted, the task default is used
- Kaggle CLI and authentication are expected to be preconfigured
- Competition zip contents are expected to include `train.csv`, `test.csv`, and `sample_submission.csv`
- Binary classification supports any two-class target labels; the positive class is resolved from the training target and used consistently for diagnostics, scoring, and probability extraction
- Binary classification must have an explicit positive-class contract; when `positive_label` is omitted, the workflow only auto-resolves the positive class for `[0, 1]`, `[False, True]`, or `["No", "Yes"]`
- `id_column` inference must resolve to exactly one column present in `train.csv`, `test.csv`, and `sample_submission.csv`
- The resolved `id_column` is identifier metadata and must be excluded from preprocessing and model fitting by default
- `label_column` inference must resolve to exactly one column present in `train.csv` and `sample_submission.csv` but not `test.csv`
- Submission must resolve `competition_slug`, `task_type`, `id_column`, and `label_column` from `run_manifest.json` rather than re-inferring them from raw train/test data
- Multi-model submission must default to `best_model_id` unless a specific `model_id` is requested explicitly
- `sample_submission.csv` must match the resolved schema exactly as `[id_column, label_column]`
- The selected model artifact `test_predictions.csv[id_column]` must match `sample_submission.csv[id_column]` exactly in both values and row order
- Feature override columns must exist and cannot overlap between forced numeric and forced categorical sets
- Configured metric must normalize to a supported metric compatible with the configured task type
- CV splitter construction must support both `cv_shuffle=true` and `cv_shuffle=false`
- Submission output must match the schema, row count, and ID ordering/content of `sample_submission.csv`
- Fail fast is the default behavior; errors are surfaced directly with minimal wrapping

Hard-error cases include:
- Missing config file -> hard error
- Schema/type violation -> hard error
- Any attempt to use additional config sources -> hard error
- Kaggle command failure -> hard error (bubble up with minimal wrapping)
- Missing `task_type` or `primary_metric` -> hard error
- Unknown/unsupported configured `primary_metric` -> hard error
- Invalid task/metric pairing (for example `binary` + `rmse`) -> hard error
- Invalid `model_id` or `model_ids` for the configured task -> hard error
- Empty or duplicate `model_ids` -> hard error
- Missing/invalid competition zip contents -> hard error
- `id_column` inference not exactly one column -> hard error
- `label_column` inference not exactly one column -> hard error
- Invalid `id_column` or `label_column` override -> hard error
- Unknown columns in `force_categorical`, `force_numeric`, or `drop_columns` -> hard error
- Any overlap between `force_categorical` and `force_numeric` -> hard error
- No modeled feature columns remaining after excluding `id_column` and applying `drop_columns` -> hard error
- Preprocessing fit/transform failure -> hard error
- Unsupported task type for CV/model selection -> hard error
- Unsupported metric for chosen task -> hard error
- Any CV/training fit or scoring failure -> hard error
- Fold assignment gaps in OOF generation -> hard error
- Requested submission `model_id` not present in the run manifest -> hard error
- Submission schema or ID mismatch against `sample_submission.csv` -> hard error
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
- New model families should be introduced in `models.py` with explicit task compatibility and matching artifact outputs.
- New preprocessing modes should be added in `preprocess.py` without breaking the existing feature-frame contract.
- New run or submission artifacts should be reflected in both the artifact contract above and the corresponding ledger rows.
