# Technical Guide

Technical reference for the current repository design. Use GitHub issues and pull requests for active implementation tracking; this document describes the system as it exists and the contracts it is expected to preserve.

The intended operating scope is Kaggle Playground Series tabular competitions. Current default development targets are `playground-series-s5e12` for binary classification with `primary_metric: roc_auc` and `playground-series-s5e10` for regression with `primary_metric: mse`. The current binary production target is `playground-series-s6e3` with `primary_metric: roc_auc`.

## System Flow
1. Load and validate the local repository-root `config.yaml`.
2. Use explicit `task_type` and `primary_metric` from config.
3. Download the competition zip into `data/<competition_slug>/` when it is missing.
4. Load one shared dataset context from `train.csv`, `test.csv`, and `sample_submission.csv`.
5. Run EDA against that shared dataset context and write report CSVs under `reports/<competition_slug>/`.
6. Resolve `id_column` and `label_column` from `train.csv`, `test.csv`, and `sample_submission.csv`, then prepare raw feature frames from the train/test data with the resolved `id_column` excluded from modeled features.
7. During training, build fold-local preprocessing from the selected model recipe:
   - `onehot`: numeric median imputation + `StandardScaler`; categorical most-frequent imputation + `OneHotEncoder`
   - `ordinal`: numeric median imputation; categorical most-frequent imputation + `OrdinalEncoder`
   - `native`: numeric median imputation inside a pandas frame; categorical missing-value fill with native categorical columns preserved for CatBoost
8. Resolve the current `experiment.candidate.model_family + preprocessor` combination to one internal canonical model recipe, then train that one configured candidate:
   - regression: `ridge + onehot`, `elasticnet + onehot`, `random_forest + ordinal`, `extra_trees + ordinal`, `hist_gradient_boosting + ordinal`, `lightgbm + ordinal`, `catboost + native`, `xgboost + ordinal`
   - binary classification: `logistic_regression + onehot`, `random_forest + ordinal`, `extra_trees + ordinal`, `hist_gradient_boosting + ordinal`, `lightgbm + ordinal`, `catboost + native`, `xgboost + ordinal`
9. Write run-level diagnostics, `model_summary.csv`, and a canonical `run_manifest.json` under `artifacts/<competition_slug>/train/<run_id>/`.
10. Write per-model fold metrics, OOF predictions, and test predictions under `artifacts/<competition_slug>/train/<run_id>/<model_id>/`.
11. Validate predictions against `sample_submission.csv`, including exact ID content and order, using `run_manifest.json` as the submission metadata contract, apply metric-aware binary prediction validation, write `submission.csv` in the selected model directory, and optionally submit to Kaggle.

When the explicit `tune` stage is selected, the workflow additionally runs an Optuna study for the current experiment candidate when `experiment.candidate.optimization.enabled=true`, writes tuning artifacts under `artifacts/<competition_slug>/tune/<study_id>/`, and retrains the best trial into the standard train artifact layout with `tuning_provenance` recorded in `run_manifest.json`.

## CLI Stages
- `uv run python main.py`: default full pipeline (`fetch` -> `eda` -> `train` -> `submit`)
- `uv run python main.py fetch`: ensure competition data is present
- `uv run python main.py eda`: fetch if needed, load the shared dataset context, and write EDA reports
- `uv run python main.py preprocess`: fetch if needed, load the shared dataset context, validate model-specific preprocessing paths, and write preprocessing diagnostics
- `uv run python main.py train`: fetch if needed, load the shared dataset context, and write training artifacts
- `uv run python main.py tune`: fetch if needed, load the shared dataset context, run an Optuna study for the current experiment candidate when `experiment.candidate.optimization.enabled=true`, write tuning artifacts, and retrain the best trial into normal training artifacts
- `uv run python main.py submit --run-dir artifacts/.../train/<run_id>`: prepare or submit from an explicit existing run artifact
- `uv run python main.py submit --run-id <run_id>`: resolve the run under `artifacts/<competition_slug>/train/<run_id>` using the configured competition slug

The `preprocess` stage is intentionally diagnostic. It is not part of the default runtime contract and it does not create a second training artifact layout.

The default `submit` path supports current manifest-backed run artifacts only. Unsupported older local artifact layouts fail directly instead of using compatibility fallbacks.

## Module Responsibilities
- `main.py`: orchestration entrypoint for config loading plus stage-specific CLI dispatch across fetch, EDA, preprocess diagnostics, training, tuning, and submission.
- `src/tabular_shenanigans/config.py`: Pydantic-backed nested config schema for `competition` plus `experiment`, metric normalization, candidate-to-model resolution, and runtime contract validation.
- `src/tabular_shenanigans/data.py`: competition download, zip access, metric helpers, dataset schema resolution, and sample-submission template loading.
- `src/tabular_shenanigans/eda.py`: competition-scan EDA summaries written to CSV from the shared dataset context, including missingness, categorical cardinality, target summary, and feature-type counts.
- `src/tabular_shenanigans/models.py`: model-recipe registry, candidate `model_family + preprocessor` resolution, tunable-model search spaces, optional booster loading, and estimator construction for supported presets.
- `src/tabular_shenanigans/preprocess.py`: feature frame preparation, column typing, scheme-specific preprocessing pipelines, native-frame support for CatBoost, and preprocess-stage diagnostics.
- `src/tabular_shenanigans/cv.py`: task-aware CV splitters and metric scoring helpers.
- `src/tabular_shenanigans/train.py`: config-selected training from the shared dataset context, shared split handling, artifact writing, and run ledger updates.
- `src/tabular_shenanigans/tune.py`: Optuna study execution for the current experiment candidate, study artifact writing, and best-trial retraining into the standard train artifact layout.
- `src/tabular_shenanigans/submit.py`: submission schema validation, model-artifact selection, submission message creation, Kaggle submission, and submission ledger updates.

## Configuration Contract
Input:
- One local runtime config file: `config.yaml` (single source of truth)
- Tracked example configs: `config.binary.example.yaml` and `config.regression.example.yaml`
- Expected workflow: copy one example file to `config.yaml`, then edit `config.yaml` for the local run
- Required top-level keys:
  - `competition`
  - `experiment`
- `competition` keys:
  - `slug`
  - `task_type` (`regression` or `binary`)
  - `primary_metric` (`rmse`, `mse`, `rmsle`, `mae`, `roc_auc`, `log_loss`, `accuracy`)
  - optional `positive_label` (explicit positive class for binary competitions; required unless observed labels match one of the documented safe conventions `[0, 1]`, `[False, True]`, or `["No", "Yes"]`)
  - optional schema overrides:
    - `id_column`
    - `label_column`
  - `cv`:
    - `n_splits` (integer >= 2, default 7)
    - `shuffle` (boolean, default true)
    - `random_state` (integer, default 42)
  - optional `features` block:
    - `force_categorical` (list of column names)
    - `force_numeric` (list of column names)
    - `drop_columns` (list of additional modeled-feature columns to remove after the ID column is excluded by default)
    - `low_cardinality_int_threshold` (positive integer)
- `experiment` keys:
  - `name`
  - optional `notes`
  - required `candidate`
  - optional `submit`
- Current `experiment.candidate` keys:
  - `candidate_type` (currently only `model`)
  - `candidate_id`
  - `preprocessor` (`onehot`, `ordinal`, or `native`)
  - `model_family`
    - regression: `ridge`, `elasticnet`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `lightgbm`, `catboost`, `xgboost`
    - binary classification: `logistic_regression`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `lightgbm`, `catboost`, `xgboost`
  - optional `model_params`
  - optional `optimization`:
    - `enabled` (boolean, default false)
    - `method` (currently only `optuna`)
    - `n_trials` (integer >= 1, optional)
    - `timeout_seconds` (integer >= 1, optional)
    - `random_state` (integer, default 42)
- Current `experiment.submit` keys:
  - `enabled` (boolean, default false)
  - `message_prefix` (string, optional)

Binary prediction artifact contract:
- `roc_auc` and `log_loss` write positive-class probabilities to `test_predictions.csv` and `submission.csv`
- `accuracy` writes predicted class labels to `test_predictions.csv` and `submission.csv`

The config is validated by Pydantic with `extra="forbid"`. Unknown keys, schema mismatches, and missing required fields are hard errors.
Configured metrics are normalized to the internal metric names during config validation.
The old flat config layout is unsupported and fails fast.
The current runtime resolves `experiment.candidate.model_family + experiment.candidate.preprocessor` to one internal canonical `model_id`.
optimization requires at least one stopping condition: `experiment.candidate.optimization.n_trials` or `experiment.candidate.optimization.timeout_seconds`.
the `tune` stage uses the current experiment candidate only.
LightGBM, CatBoost, and XGBoost require the optional booster dependencies installed via `uv sync --extra boosters`.

## Preferred Verification Targets
- `playground-series-s5e12`: binary development smoke test with `task_type: binary` and `primary_metric: roc_auc`
- `playground-series-s6e3`: binary production-target smoke test with `task_type: binary` and `primary_metric: roc_auc`
- `playground-series-s5e10`: regression smoke test with `task_type: regression` and `primary_metric: mse`

Manual verification steps for each target:
- copy the corresponding tracked example config to `config.yaml`
- verify the competition assets include `train.csv`, `test.csv`, and `sample_submission.csv`
- run the workflow from a clean repo state with explicit `competition` plus one current `experiment.candidate`
- confirm inferred `id_column` and `label_column`
- confirm `model_summary.csv` is generated in the run directory
- confirm `test_predictions.csv` is generated in each model directory
- confirm `submission.csv` validates against `sample_submission.csv`, including exact ID values and order, for the selected model directory
- confirm binary outputs match the configured metric contract: probabilities for `roc_auc`/`log_loss`, labels for `accuracy`
- when tuning is enabled, run `uv run python main.py tune` and confirm `study_manifest.json`, `study_summary.csv`, `trials.csv`, and `best_params.json` are generated under `artifacts/<competition_slug>/tune/<study_id>/`
- when tuning is enabled, confirm the best-trial retrain writes a standard train artifact and records `tuning_provenance` in `run_manifest.json`

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
  - when the `preprocess` stage is run:
    - `preprocess_summary.csv`
    - `preprocess_features.csv`
    - `preprocess_models.csv`
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
- Each model entry in `model_summary.csv` and `run_manifest.json` records the resolved `preprocessing_scheme_id`
- tuned retrain runs record `tuning_provenance` in `run_manifest.json`
- Tuning artifacts under `artifacts/<competition_slug>/tune/<study_id>/`:
  - `study_manifest.json`
  - `study_summary.csv`
  - `trials.csv`
  - `best_params.json`
- Training ledger at `artifacts/<competition_slug>/train/runs.csv` with compact comparison fields and task-aware target summary fields
- Append-only submission ledger at `artifacts/<competition_slug>/train/submissions.csv` with submission event metadata only

## Runtime Invariants And Failure Behavior
- One runtime config source only: local repository-root `config.yaml`
- No config overrides via CLI or environment variables
- Tracked example config files are documentation and starting points only; they are never read automatically at runtime
- top-level `competition` and `experiment` sections must be present in config for every run
- the old flat config layout is unsupported
- `competition.task_type` and `competition.primary_metric` must be present in config for every run
- the current runtime supports one `experiment.candidate` of type `model`
- `experiment.candidate.model_family + experiment.candidate.preprocessor` must resolve to one supported canonical recipe for the configured task
- the `tune` stage requires `experiment.candidate.optimization.enabled=true`, uses the current experiment candidate only, and retrains exactly one tuned candidate into the normal train artifact layout
- enabled optimization must have at least one stopping condition: `experiment.candidate.optimization.n_trials` or `experiment.candidate.optimization.timeout_seconds`
- Submit-time `model_id` remains the trained-model selector for choosing one artifact from a run
- `native_catboost` must preserve categorical feature positions through preprocessing so CatBoost can receive `cat_features`
- Kaggle CLI and authentication are expected to be preconfigured
- Competition zip contents are expected to include `train.csv`, `test.csv`, and `sample_submission.csv`
- Binary classification supports any two-class target labels
- The positive class is resolved from the training target and used consistently for diagnostics and scoring
- Binary `roc_auc` and `log_loss` artifacts use positive-class probabilities
- Binary `accuracy` artifacts use class labels from the observed binary label pair
- Binary classification must have an explicit positive-class contract; when `positive_label` is omitted, the workflow only auto-resolves the positive class for `[0, 1]`, `[False, True]`, or `["No", "Yes"]`
- `id_column` inference must resolve to exactly one column present in `train.csv`, `test.csv`, and `sample_submission.csv`
- The resolved `id_column` is identifier metadata and must be excluded from preprocessing and model fitting by default
- `label_column` inference must resolve to exactly one column present in `train.csv` and `sample_submission.csv` but not `test.csv`
- Submission must resolve `competition_slug`, `task_type`, `id_column`, and `label_column` from `run_manifest.json` rather than re-inferring them from raw train/test data
- Multi-model submission must default to `best_model_id` unless a specific `model_id` is requested explicitly
- `sample_submission.csv` must match the resolved schema exactly as `[id_column, label_column]`
- The selected model artifact `test_predictions.csv[id_column]` must match `sample_submission.csv[id_column]` exactly in both values and row order
- Submission requires the current per-model prediction layout at `artifacts/<competition_slug>/train/<run_id>/<model_id>/test_predictions.csv`
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
- Missing required top-level `competition` or `experiment` sections -> hard error
- Any use of the old flat config layout -> hard error
- Unsupported `experiment.candidate.candidate_type` -> hard error
- Invalid configured `experiment.candidate.model_family + preprocessor` combination for the configured task -> hard error
- enabled optimization without `experiment.candidate.optimization.n_trials` or `experiment.candidate.optimization.timeout_seconds` -> hard error
- enabled optimization for an unsupported candidate combination -> hard error
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
- Missing or legacy-shaped submit artifacts outside the current manifest-backed per-model contract -> hard error
- Submission schema or ID mismatch against `sample_submission.csv` -> hard error
- Binary probability artifact outside `[0, 1]` for `roc_auc` or `log_loss` -> hard error
- Binary label artifact containing values outside the observed label pair for `accuracy` -> hard error
- Kaggle submission command failure when `experiment.submit.enabled=true` -> hard error

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
- New user-facing config keys should be added to the relevant nested config model in `config.py` and documented in both this file and `README.md`.
- If the user-facing config workflow changes, update `config.binary.example.yaml` and `config.regression.example.yaml` alongside the docs.
- New metrics should be normalized and validated during config loading, then scored in `cv.py`.
- New model families should be introduced in `models.py` with explicit task compatibility, canonical recipe IDs, and matching artifact outputs.
- New preprocessing modes should be added in `preprocess.py` without breaking the existing feature-frame contract or the preprocessing-first recipe naming convention.
- New run or submission artifacts should be reflected in both the artifact contract above and the corresponding ledger rows.
